"""
完整的 Transformer 语言模型（LLaMA 风格）。

前向传播就四步：
    token_ids → embedding → [Transformer Block × N] → RMSNorm → lm_head → logits

拿这个 logits 去做 cross_entropy 就是训练，做 softmax 采样就是生成。
lm_head 和 embedding 可以选择共享权重（weight tying），省参数还能稳定训练。
"""
import torch
import torch.nn.functional as F
from lmfromzero.block import transformer_block
from lmfromzero.norm import rms_norm


def transformer_lm(
    token_ids: torch.Tensor,
    weights: dict[str, torch.Tensor],
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    rope_theta: float,
    rms_norm_eps: float,
    mask: torch.Tensor | None = None,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transformer 语言模型的前向。

    weights 字典布局（假设有 N 层）：
    - "embedding"          [vocab_size, d_model]
    - "block_{i}_attn_qkv" 第 i 层的 QKV 合并投影
    - "block_{i}_attn_out" 第 i 层的注意力输出投影
    - "block_{i}_ffn_w1"   第 i 层的 SwiGLU 门控
    - "block_{i}_ffn_w2"   第 i 层的 SwiGLU 上投影
    - "block_{i}_ffn_w3"   第 i 层的 SwiGLU 下投影
    - "block_{i}_ln1_w"    第 i 层注意力前 RMSNorm
    - "block_{i}_ln2_w"    第 i 层 FFN 前 RMSNorm
    - "final_norm"         [d_model]
    - "lm_head"            [d_model, vocab_size]

    输入 token_ids [batch, seq_len]，输出 logits [batch, seq_len, vocab_size]。
    """
    batch, seq_len = token_ids.shape

    # 1. 嵌入查找
    x = F.embedding(token_ids, weights["embedding"])

    # 2. 逐层过 Transformer Block
    for layer_idx in range(num_layers):
        block_weights = {
            key.replace(f"block_{layer_idx}_", ""): value
            for key, value in weights.items()
            if key.startswith(f"block_{layer_idx}_")
        }
        assert len(block_weights) == 7, \
            f"第 {layer_idx} 层：期望 7 个权重，实际只找到 {len(block_weights)} 个"

        x = transformer_block(
            x,
            weights=block_weights,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            mask=mask,
            token_positions=token_positions,
        )

    # 3. 最后一道 RMSNorm
    x = rms_norm(x, weights["final_norm"], eps=rms_norm_eps)

    # 4. 输出投影到 vocab_size
    logits = torch.matmul(x, weights["lm_head"])

    return logits
