"""
语言模型训练主脚本。

流程：加载 tokenizer → 初始化模型 → 加载数据 → 训练循环（前向→loss→反向→梯度裁剪→优化）

CPU 快速试跑：
    python scripts/train.py --train_data data/train.txt --tokenizer_prefix checkpoints/tokenizer --max_steps 500 --batch_size 4

GPU 完整训练：
    python scripts/train.py --train_data data/train.txt --tokenizer_prefix checkpoints/tokenizer --device cuda --max_steps 50000
"""
import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lmfromzero.config import ModelConfig
from lmfromzero.tokenizer import Tokenizer
from lmfromzero.ops import cross_entropy, gradient_clipping
from lmfromzero.data import load_and_tokenize, get_batch
from lmfromzero.model import transformer_lm
from lmfromzero.optimizer import AdamW
from lmfromzero.scheduler import get_lr_cosine_schedule
from lmfromzero.checkpoint import save_checkpoint, load_checkpoint


# ═══════════════════════════════════════════════════════════════
# 权重初始化
# ═══════════════════════════════════════════════════════════════

def init_weights(config: ModelConfig) -> dict[str, torch.Tensor]:
    """初始化所有权重，返回扁平字典。

    初始化方式参考 LLaMA：
    - Embedding 和 lm_head：N(0, 0.02)
    - 残差层里的投影矩阵：N(0, 0.02/sqrt(2*num_layers))，越深越小心
    - RMSNorm 权重：全 1
    """
    d_model = config.d_model
    num_layers = config.num_layers
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    d_ff = config.d_ff
    vocab_size = config.vocab_size

    d_qkv = (num_heads + 2 * num_kv_heads) * head_dim

    weights: dict[str, torch.Tensor] = {}

    # Embedding
    weights["embedding"] = torch.randn(vocab_size, d_model) * 0.02

    # 残差层 std 按深度缩放
    residual_std = 0.02 / math.sqrt(2 * num_layers)

    for i in range(num_layers):
        prefix = f"block_{i}_"
        weights[f"{prefix}attn_qkv"] = torch.randn(d_model, d_qkv) * residual_std
        weights[f"{prefix}attn_out"] = torch.randn(d_model, d_model) * residual_std
        weights[f"{prefix}ffn_w1"] = torch.randn(d_model, d_ff) * residual_std
        weights[f"{prefix}ffn_w2"] = torch.randn(d_model, d_ff) * residual_std
        weights[f"{prefix}ffn_w3"] = torch.randn(d_ff, d_model) * residual_std
        weights[f"{prefix}ln1_w"] = torch.ones(d_model)
        weights[f"{prefix}ln2_w"] = torch.ones(d_model)

    weights["final_norm"] = torch.ones(d_model)
    weights["lm_head"] = torch.randn(d_model, vocab_size) * 0.02

    return weights


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """下三角矩阵，让每个 token 只能看到前面（包括自己）的 token。"""
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


def count_parameters(weights: dict[str, torch.Tensor]) -> int:
    """数一下总共有多少参数。"""
    return sum(p.numel() for p in weights.values())


def estimate_perplexity(loss: float) -> float:
    """交叉熵 → 困惑度：PPL = exp(loss)。

    直观理解：模型在每个位置平均要猜几个候选 token。
    PPL 越低越好，PPL=1 说明完美预测。
    """
    return math.exp(loss)


# ═══════════════════════════════════════════════════════════════
# 训练主循环
# ═══════════════════════════════════════════════════════════════

def train(config: ModelConfig, args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")
    if device.type == "cuda":
        print(f"       GPU: {torch.cuda.get_device_name(0)}")
        print(f"       显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Tokenizer ───────────────────────────────────
    print(f"\n[Tokenizer] 加载中...")
    with open(f"{args.tokenizer_prefix}_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(f"{args.tokenizer_prefix}_merges.pkl", "rb") as f:
        merges = pickle.load(f)

    special_map = {}
    for tid, tbytes in vocab.items():
        s = tbytes.decode("utf-8", errors="replace")
        if s in ("<|endoftext|>", "<|pad|>"):
            special_map[s] = tid

    tokenizer = Tokenizer(vocab, merges, special_tokens=special_map)
    config.vocab_size = tokenizer.vocab_size
    print(f"       词表大小: {config.vocab_size}")

    # ── 数据 ─────────────────────────────────────────
    print(f"\n[数据] 加载并 tokenize: {args.train_data}")
    token_ids = load_and_tokenize(args.train_data, tokenizer)
    data = np.array(token_ids, dtype=np.int64)
    print(f"       Token 总数: {len(data):,}")
    print(f"       约 {len(data) // (config.batch_size * config.context_length):,} 个 batch 可用")

    # ── 模型 ─────────────────────────────────────────
    print(f"\n[模型] 初始化权重...")
    weights = init_weights(config)
    weights = {k: v.to(device) for k, v in weights.items()}
    n_params = count_parameters(weights)
    print(f"       参数量: {n_params:,} (~{n_params / 1e6:.1f}M)")

    # ── 优化器 ───────────────────────────────────────
    params_list = [p for p in weights.values() if p.requires_grad]
    optimizer = AdamW(
        params_list,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # Causal mask 只算一次，后面复用
    causal_mask = make_causal_mask(config.context_length, device)

    # ── 恢复 checkpoint ──────────────────────────────
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n[恢复] 加载 checkpoint: {args.resume}")
        start_step = load_checkpoint(args.resume, weights, optimizer, map_location=str(device))
        print(f"       从 step {start_step} 继续")

    # ── 模型前向函数 ─────────────────────────────────
    def model_forward(token_ids_tensor: torch.Tensor) -> torch.Tensor:
        return transformer_lm(
            token_ids_tensor,
            weights=weights,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            mask=causal_mask[:token_ids_tensor.shape[1], :token_ids_tensor.shape[1]],
        )

    # ── 训练循环 ─────────────────────────────────────
    print(f"\n[训练] 开始! max_steps={config.max_steps}, warmup={config.warmup_steps}")
    print(f"       batch={config.batch_size}, ctx_len={config.context_length}")
    print(f"       {'步数':>8s}  {'loss':>8s}  {'ppl':>8s}  {'lr':>10s}  {'步/s':>8s}")

    step = start_step
    pbar = tqdm(total=config.max_steps - start_step, desc="训练进度")
    total_loss = 0.0
    loss_count = 0
    best_loss = float("inf")

    while step < config.max_steps:
        # 学习率
        lr = get_lr_cosine_schedule(
            step,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            peak_lr=config.learning_rate,
            min_lr=config.learning_rate * 0.1,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 取 batch
        inputs, targets = get_batch(data, config.batch_size, config.context_length)
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向 + loss
        t0 = time.time()
        logits = model_forward(inputs)
        loss = cross_entropy(logits, targets)

        # 反向 + 梯度裁剪 + 更新
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(params_list, max_l2_norm=config.grad_clip)
        optimizer.step()

        dt = time.time() - t0

        # 日志
        total_loss += loss.item()
        loss_count += 1
        ppl = estimate_perplexity(loss.item())

        if step % args.log_every == 0 or step == config.max_steps - 1:
            avg_loss = total_loss / max(loss_count, 1)
            avg_ppl = estimate_perplexity(avg_loss)
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ppl": f"{avg_ppl:.1f}",
                "lr": f"{lr:.2e}",
                "步/s": f"{1 / max(dt, 1e-6):.1f}",
            })
            total_loss = 0.0
            loss_count = 0

        # Checkpoint
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_path = f"{args.checkpoint_dir}/step_{step:07d}.pt"
            save_checkpoint(
                model_state=weights,
                optimizer_state=optimizer.state_dict(),
                step=step,
                filepath=save_path,
            )
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    model_state=weights,
                    optimizer_state=optimizer.state_dict(),
                    step=step,
                    filepath=f"{args.checkpoint_dir}/best.pt",
                )

        step += 1
        pbar.update(1)

    pbar.close()

    # 最终保存
    final_path = f"{args.checkpoint_dir}/final.pt"
    save_checkpoint(
        model_state=weights,
        optimizer_state=optimizer.state_dict(),
        step=step - 1,
        filepath=final_path,
    )
    print(f"\n[完成] 模型已保存到: {final_path}")
    print(f"       总步数: {step}")


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="训练 Transformer 语言模型")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据文件")
    parser.add_argument("--tokenizer_prefix", type=str, required=True, help="tokenizer 文件前缀")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="checkpoint 目录")
    parser.add_argument("--device", type=str, default="cpu", help="训练设备 (cpu/cuda)")
    parser.add_argument("--max_steps", type=int, default=50000, help="总步数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--context_length", type=int, default=512, help="上下文长度")
    parser.add_argument("--log_every", type=int, default=50, help="每 N 步记录一次")
    parser.add_argument("--save_every", type=int, default=2000, help="每 N 步保存 checkpoint")
    parser.add_argument("--resume", type=str, default="", help="从 checkpoint 恢复")
    parser.add_argument("--preset", type=str, default="default", choices=["default", "small"],
                        help="模型规模预设 (default: ~150M, small: ~55M)")
    args = parser.parse_args()

    if args.preset == "small":
        config = ModelConfig.small()
    else:
        config = ModelConfig()
    config.max_steps = args.max_steps
    config.batch_size = args.batch_size
    config.context_length = args.context_length

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train(config, args)


if __name__ == "__main__":
    main()
