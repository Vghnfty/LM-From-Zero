from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import regex as re
import torch
import torch.nn.functional as F
from torch import Tensor

# GPT-2 预分词正则：先把文本切成小块，BPE 合并在每个小块里独立进行，不跨块。
# 这就是 tiktoken 给 GPT-2（r50k_base / p50k_base）用的那个正则。
_GPT2_PRE_TOKENIZE = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
)


def _pretokenize_gpt2(text: str) -> list[str]:
    """用 GPT-2 正则把文本切成预分词小块。"""
    return _GPT2_PRE_TOKENIZE.findall(text)


# ═══════════════════════════════════════════════════════════════
# 1. Softmax —— 把 logits 变成概率
# ═══════════════════════════════════════════════════════════════

def run_softmax(in_features: Tensor, dim: int) -> Tensor:
    x_max = in_features.max(dim=dim, keepdim=True).values
    shifted = in_features - x_max
    exp_shifted = shifted.exp()
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)


# ═══════════════════════════════════════════════════════════════
# 2. Cross Entropy —— 交叉熵损失
# ═══════════════════════════════════════════════════════════════

def run_cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(inputs, targets, reduction='mean')


# ═══════════════════════════════════════════════════════════════
# 3. 梯度裁剪
# ═══════════════════════════════════════════════════════════════

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total_norm_sq = torch.tensor(0.0)
    for p in parameters:
        if p.grad is not None:
            total_norm_sq = total_norm_sq + (p.grad.norm() ** 2)
    total_norm = total_norm_sq.sqrt()
    scale = max_l2_norm / (total_norm + 1e-6)
    if scale < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)


# ═══════════════════════════════════════════════════════════════
# 4. SiLU —— x * sigmoid(x)
# ═══════════════════════════════════════════════════════════════

def run_silu(in_features: Tensor) -> Tensor:
    return F.silu(in_features)


# ═══════════════════════════════════════════════════════════════
# 5. RMSNorm —— 不用减均值的归一化
# ═══════════════════════════════════════════════════════════════

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    x_float = in_features.float()
    rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + eps)
    x_normed = x_float / rms
    return (x_normed * weights).type_as(in_features)


# ═══════════════════════════════════════════════════════════════
# 6. RoPE —— 旋转位置编码
# ═══════════════════════════════════════════════════════════════

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
) -> Tensor:
    device = in_query_or_key.device
    dtype = in_query_or_key.dtype
    seq_len = in_query_or_key.shape[-2]

    if token_positions.dim() == 1:
        token_positions = token_positions.to(device=device, dtype=dtype)
    else:
        token_positions = token_positions.to(device=device, dtype=dtype)

    # freqs: base^(-2i/d_k) for i = 0, 2, 4, ..., d_k-2
    indices = torch.arange(0, d_k, 2, device=device, dtype=dtype)
    freqs = 1.0 / (theta ** (indices / d_k))

    # angles[m, i] = pos[m] * freq[i]
    angles = token_positions.unsqueeze(-1) * freqs.unsqueeze(0)
    cos = angles.cos()
    sin = angles.sin()

    # repeat for even/odd interleave
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)

    # broadcast to match x dims
    while cos.dim() < in_query_or_key.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]
    cos_2d = cos[..., 0::2]
    sin_2d = sin[..., 0::2]

    rotated_even = x_even * cos_2d - x_odd * sin_2d
    rotated_odd = x_even * sin_2d + x_odd * cos_2d

    result = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(start_dim=-2)
    return result.type_as(in_query_or_key)


# ═══════════════════════════════════════════════════════════════
# 7. 缩放点积注意力
# ═══════════════════════════════════════════════════════════════

def run_scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask.bool() if mask.dtype != torch.bool else ~mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    return torch.matmul(attn_weights, V)


# ═══════════════════════════════════════════════════════════════
# 8. SwiGLU —— 带门控的 FFN
# ═══════════════════════════════════════════════════════════════

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    # w1=门控(d_ff,d_model), w3=上投影(d_ff,d_model), w2=下投影(d_model,d_ff)
    gate = F.silu(torch.matmul(in_features, w1_weight.T))
    up = torch.matmul(in_features, w3_weight.T)
    return torch.matmul(gate * up, w2_weight.T)


# ═══════════════════════════════════════════════════════════════
# 9. 线性层
# ═══════════════════════════════════════════════════════════════

def run_linear(
    d_in: int,
    d_out: int,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    return torch.matmul(in_features, weights.T)


# ═══════════════════════════════════════════════════════════════
# 10. Embedding —— 词嵌入查表
# ═══════════════════════════════════════════════════════════════

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Tensor,
    token_ids: Tensor,
) -> Tensor:
    return F.embedding(token_ids, weights)


# ═══════════════════════════════════════════════════════════════
# 11. 多头自注意力（不带 RoPE）
# ═══════════════════════════════════════════════════════════════

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    head_dim = d_model // num_heads
    batch_shape = in_features.shape[:-1]  # (..., seq_len)
    seq_len = batch_shape[-1]

    # Q、K、V 分头投影
    q = torch.matmul(in_features, q_proj_weight.T)  # (..., d_model)
    k = torch.matmul(in_features, k_proj_weight.T)
    v = torch.matmul(in_features, v_proj_weight.T)

    # 重整为 (..., num_heads, seq_len, head_dim)
    q = q.view(*batch_shape, num_heads, head_dim).transpose(-3, -2)
    k = k.view(*batch_shape, num_heads, head_dim).transpose(-3, -2)
    v = v.view(*batch_shape, num_heads, head_dim).transpose(-3, -2)

    # 因果 mask：每个位置只看自己和前面的
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)
    )

    # 带因果 mask 的缩放点积注意力
    attn_output = run_scaled_dot_product_attention(q, k, v, mask=causal_mask)

    # 拼回所有头
    attn_output = attn_output.transpose(-3, -2).contiguous()
    attn_output = attn_output.view(*batch_shape, d_model)

    # 输出投影
    return torch.matmul(attn_output, o_proj_weight.T)


# ═══════════════════════════════════════════════════════════════
# 12. 多头自注意力（带 RoPE）
# ═══════════════════════════════════════════════════════════════

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor | None = None,
) -> Tensor:
    head_dim = d_model // num_heads
    batch_shape = in_features.shape[:-1]
    seq_len = batch_shape[-1]

    # 投影
    q = torch.matmul(in_features, q_proj_weight.T)
    k = torch.matmul(in_features, k_proj_weight.T)
    v = torch.matmul(in_features, v_proj_weight.T)

    # 重整为 (..., num_heads, seq_len, head_dim)
    q = q.view(*batch_shape, num_heads, head_dim).transpose(-3, -2)
    k = k.view(*batch_shape, num_heads, head_dim).transpose(-3, -2)
    v = v.view(*batch_shape, num_heads, head_dim).transpose(-3, -2)

    # Q 和 K 加 RoPE
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device)

    q = run_rope(head_dim, theta, max_seq_len, q, token_positions)
    k = run_rope(head_dim, theta, max_seq_len, k, token_positions)

    # 因果 mask
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)
    )

    # 带因果 mask 的注意力
    attn_output = run_scaled_dot_product_attention(q, k, v, mask=causal_mask)

    # 拼回
    attn_output = attn_output.transpose(-3, -2).contiguous()
    attn_output = attn_output.view(*batch_shape, d_model)

    # 输出投影
    return torch.matmul(attn_output, o_proj_weight.T)


# ═══════════════════════════════════════════════════════════════
# 13. Transformer Block（Pre-LN + RoPE）
# ═══════════════════════════════════════════════════════════════

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    seq_len = in_features.shape[-2]

    # 子层 1：带 RoPE 的多头自注意力
    residual = in_features
    x_normed = run_rmsnorm(d_model, 1e-5, weights["ln1.weight"], in_features)
    token_positions = torch.arange(seq_len, device=in_features.device)
    attn_out = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=x_normed,
        token_positions=token_positions,
    )
    x = residual + attn_out

    # 子层 2：SwiGLU 前馈网络
    residual = x
    x_normed = run_rmsnorm(d_model, 1e-5, weights["ln2.weight"], x)
    ffn_out = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=x_normed,
    )
    x = residual + ffn_out

    return x


# ═══════════════════════════════════════════════════════════════
# 14. 完整 Transformer 语言模型
# ═══════════════════════════════════════════════════════════════

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    batch_size, seq_len = in_indices.shape
    device = in_indices.device

    # Token 嵌入
    x = F.embedding(in_indices, weights["token_embeddings.weight"])

    # 位置 ID
    token_positions = torch.arange(seq_len, device=device)

    # 逐层过 Transformer
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        block_weights = {
            "attn.q_proj.weight": weights[f"{prefix}attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"{prefix}attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"{prefix}attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"{prefix}attn.output_proj.weight"],
            "ln1.weight": weights[f"{prefix}ln1.weight"],
            "ffn.w1.weight": weights[f"{prefix}ffn.w1.weight"],
            "ffn.w2.weight": weights[f"{prefix}ffn.w2.weight"],
            "ffn.w3.weight": weights[f"{prefix}ffn.w3.weight"],
            "ln2.weight": weights[f"{prefix}ln2.weight"],
        }
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=block_weights,
            in_features=x,
        )

    # 最终 RMSNorm
    x = run_rmsnorm(d_model, 1e-5, weights["ln_final.weight"], x)

    # LM head 输出投影
    logits = torch.matmul(x, weights["lm_head.weight"].T)
    return logits


# ═══════════════════════════════════════════════════════════════
# 15. 获取训练 batch
# ═══════════════════════════════════════════════════════════════

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Tensor, Tensor]:
    max_start = len(dataset) - context_length
    starts = torch.randint(0, max_start, (batch_size,), dtype=torch.long)

    inputs = []
    targets = []
    for start in starts:
        inputs.append(dataset[start : start + context_length])
        targets.append(dataset[start + 1 : start + 1 + context_length])

    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


# ═══════════════════════════════════════════════════════════════
# 16. AdamW 优化器
# ═══════════════════════════════════════════════════════════════

class _AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                m, v = state["m"], state["v"]
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                p.mul_(1 - lr * weight_decay)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss


def get_adamw_cls() -> Any:
    return _AdamW


# ═══════════════════════════════════════════════════════════════
# 17. 余弦学习率调度
# ═══════════════════════════════════════════════════════════════

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / max(1, warmup_iters)
    elif it >= cosine_cycle_iters:
        return min_learning_rate
    else:
        progress = (it - warmup_iters) / max(1, cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay


# ═══════════════════════════════════════════════════════════════
# 18. 保存 Checkpoint
# ═══════════════════════════════════════════════════════════════

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


# ═══════════════════════════════════════════════════════════════
# 19. 加载 Checkpoint
# ═══════════════════════════════════════════════════════════════

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


# ═══════════════════════════════════════════════════════════════
# 20. Tokenizer
# ═══════════════════════════════════════════════════════════════

class _Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.vocab_bytes_to_id: dict[bytes, int] = {}
        for tid, tbytes in vocab.items():
            self.vocab_bytes_to_id[tbytes] = tid

        # 合并优先级字典：{(a,b): rank}，rank 越小越优先
        self.merge_rank: dict[tuple[bytes, bytes], int] = {}
        for rank, (a, b) in enumerate(merges):
            self.merge_rank[(a, b)] = rank

        # 特殊 token 映射
        self.special_tokens: dict[str, int] = {}
        if special_tokens:
            for tok in special_tokens:
                tok_bytes = tok.encode("utf-8")
                # 在词表里找到这个特殊 token 的 ID
                for tid, tbytes in vocab.items():
                    if tbytes == tok_bytes:
                        self.special_tokens[tok] = tid
                        break

    def _encode_bytes(self, text_bytes: bytes) -> list[int]:
        """BPE 编码一段纯字节（单个预分词块，不含特殊 token）。"""
        if not text_bytes:
            return []

        tokens: list[bytes] = [bytes([b]) for b in text_bytes]

        while len(tokens) >= 2:
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

        ids = []
        for t in tokens:
            tid = self.vocab_bytes_to_id.get(t)
            if tid is not None:
                ids.append(tid)
            else:
                for b in t:
                    ids.append(b)
        return ids

    def _encode_text(self, text: str) -> list[int]:
        """GPT-2 预分词编码：先切成块，每块独立 BPE。"""
        chunks = _pretokenize_gpt2(text)
        ids = []
        for chunk in chunks:
            ids.extend(self._encode_bytes(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        if not self.special_tokens:
            return self._encode_text(text)

        # 先按特殊 token 把文本切开，普通部分走正常编码，特殊部分直接用预留 ID。
        special_patterns = list(self.special_tokens.keys())
        special_patterns.sort(key=len, reverse=True)  # longest first

        result_ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for sp in special_patterns:
                if text.startswith(sp, i):
                    result_ids.append(self.special_tokens[sp])
                    i += len(sp)
                    matched = True
                    break
            if not matched:
                end = len(text)
                for sp in special_patterns:
                    pos = text.find(sp, i)
                    if pos != -1 and pos < end:
                        end = pos
                if end > i:
                    result_ids.extend(self._encode_text(text[i:end]))
                i = end
        return result_ids

    def decode(self, ids: list[int]) -> str:
        token_bytes = bytearray()
        for tid in ids:
            tbytes = self.vocab.get(tid)
            if tbytes is not None:
                token_bytes.extend(tbytes)
            else:
                token_bytes.extend(f"[UNK:{tid}]".encode("utf-8"))
        return token_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable):
        """流式编码 —— 把所有文本拼起来一次性编码。"""
        all_text = "".join(iterable)
        yield from self.encode(all_text)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return _Tokenizer(vocab, merges, special_tokens)


# ═══════════════════════════════════════════════════════════════
# 21. BPE 训练
# ═══════════════════════════════════════════════════════════════

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 二进制读入，然后立即把 Windows 的 \r\n 转成 \n。
    # 不然预分词块边界跟 Linux 参考数据对不上。
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8")
    text = text.replace("\r\n", "\n")

    # GPT-2 预分词：先切块，BPE 只在块内合并，不跨块。
    # 特殊 token 不删，留在训练文本里 —— 参考实现就是这样做的，
    # 只是防止合并出含有特殊 token 子串的 token（下面有检查）。
    chunks = _pretokenize_gpt2(text)

    # 每个块编码成 UTF-8 字节，转成整数 ID 列表。
    # 用整数 ID 比较更快，需要查字节内容的时候再去 vocab 里找。
    chunk_tokens: list[list[int]] = []
    for chunk in chunks:
        chunk_tokens.append(list(chunk.encode("utf-8")))

    vocab: dict[int, bytes] = {b: bytes([b]) for b in range(256)}
    next_id = 256
    special_start = vocab_size - len(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # ── 建立初始 pair 计数 + pair→块 映射 ──
    pair_counts: dict[tuple[int, int], int] = {}
    pair_to_chunks: dict[tuple[int, int], set[int]] = {}
    for chunk_idx, tokens in enumerate(chunk_tokens):
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if pair not in pair_to_chunks:
                pair_to_chunks[pair] = set()
            pair_to_chunks[pair].add(chunk_idx)

    while next_id < special_start:
        if not pair_counts:
            break

        # 字节值打破平局：比较实际字节内容而不是整数 ID。
        # 这样合并顺序才能跟参考实现一模一样。
        b1, b2 = max(pair_counts, key=lambda x: (pair_counts[x], (vocab[x[0]], vocab[x[1]])))

        merged_bytes = vocab[b1] + vocab[b2]

        # 跳过那些合并后会产生含特殊 token 子串的 pair。
        # 保证 <|endoftext|> 这类特殊 token 在 BPE 训练过程中不会被意外拼出来。
        if b"<|" in merged_bytes:
            # 直接扔掉这个 pair，下一轮选次优的
            pair_counts.pop((b1, b2), None)
            pair_to_chunks.pop((b1, b2), None)
            continue

        new_id = next_id
        next_id += 1

        vocab[new_id] = merged_bytes
        merges.append((vocab[b1], vocab[b2]))

        # ── Incremental update: only process chunks that contain (b1, b2) ──
        affected = pair_to_chunks.get((b1, b2), set())
        if not affected:
            # pair 消失了（最高频的 pair 按理不会走这里，防御性代码）
            if (b1, b2) in pair_counts:
                del pair_counts[(b1, b2)]
            if (b1, b2) in pair_to_chunks:
                del pair_to_chunks[(b1, b2)]
            continue

        for chunk_idx in list(affected):
            old_tokens = chunk_tokens[chunk_idx]

            # 这个块可能已经被之前的合并改过了，不再包含当前 pair。
            # 快速扫一遍确认。
            has_pair = False
            for i in range(len(old_tokens) - 1):
                if old_tokens[i] == b1 and old_tokens[i + 1] == b2:
                    has_pair = True
                    break
            if not has_pair:
                if (b1, b2) in pair_to_chunks:
                    pair_to_chunks[(b1, b2)].discard(chunk_idx)
                continue

            # ── Remove old pair contributions from this chunk ──
            chunk_pairs: dict[tuple[int, int], int] = {}
            for i in range(len(old_tokens) - 1):
                p = (old_tokens[i], old_tokens[i + 1])
                chunk_pairs[p] = chunk_pairs.get(p, 0) + 1

            for p, cnt in chunk_pairs.items():
                pair_counts[p] = pair_counts.get(p, 0) - cnt
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                    pair_to_chunks.pop(p, None)
                elif p in pair_to_chunks:
                    pair_to_chunks[p].discard(chunk_idx)

            # ── Apply merge within this chunk ──
            new_tokens: list[int] = []
            i = 0
            while i < len(old_tokens):
                if i + 1 < len(old_tokens) and old_tokens[i] == b1 and old_tokens[i + 1] == b2:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1
            chunk_tokens[chunk_idx] = new_tokens

            # ── Add new pair contributions from this chunk ──
            new_chunk_pairs: dict[tuple[int, int], int] = {}
            for i in range(len(new_tokens) - 1):
                p = (new_tokens[i], new_tokens[i + 1])
                new_chunk_pairs[p] = new_chunk_pairs.get(p, 0) + 1

            for p, cnt in new_chunk_pairs.items():
                pair_counts[p] = pair_counts.get(p, 0) + cnt
                if p not in pair_to_chunks:
                    pair_to_chunks[p] = set()
                pair_to_chunks[p].add(chunk_idx)

    # 把特殊 token 加入词表
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    return vocab, merges
