"""
Colab 一键训练脚本 —— 在 Google Colab 的 T4 GPU 上 30-40 分钟完成端到端训练。

在 Colab 中依次运行每个 cell（以 # %% 分隔的行）即可。
也可以直接 `python notebooks/colab_train.py` 运行全部。

Cell 顺序：
  1. 安装依赖 + 克隆仓库（如果本地已有代码则跳过克隆）
  2. 下载 TinyStories 数据
  3. 训练 BPE tokenizer
  4. 训练模型（~55M, 5000 步, ~30min）
  5. 画 loss 曲线 + 生成文本
"""
import os
import sys
import time
import pickle
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

# ── 确保项目根目录在 path 里 ──────────────────────────
if os.path.exists("lmfromzero"):
    sys.path.insert(0, ".")
elif os.path.exists("../lmfromzero"):
    sys.path.insert(0, "..")

from lmfromzero.config import ModelConfig
from lmfromzero.tokenizer import Tokenizer
from lmfromzero.ops import cross_entropy, gradient_clipping
from lmfromzero.data import load_and_tokenize, get_batch
from lmfromzero.model import transformer_lm
from lmfromzero.optimizer import AdamW
from lmfromzero.scheduler import get_lr_cosine_schedule
from lmfromzero.checkpoint import save_checkpoint


# ═══════════════════════════════════════════════════════════════
# Cell 1: 安装依赖
# ═══════════════════════════════════════════════════════════════

def cell_setup():
    """在 Colab 上跑：先 pip install，再克隆仓库。
    如果已经在仓库目录里了就跳过。
    """
    if not os.path.exists("lmfromzero"):
        print("看起来不在项目目录。请在 Colab 第一个 cell 里运行：")
        print("  !git clone https://github.com/<your-username>/LMfromzero.git")
        print("  %cd LMfromzero")
        print("  !pip install -r requirements.txt")
        return

    print("✓ 项目文件已就绪")

    # 检查 GPU
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ 没有检测到 GPU，训练会很慢。请确认 Colab 运行时已选 T4 GPU。")


# ═══════════════════════════════════════════════════════════════
# Cell 2: 下载数据
# ═══════════════════════════════════════════════════════════════

def cell_download_data():
    """下载 TinyStories 数据（如果还没有的话）。"""
    os.makedirs("data", exist_ok=True)

    files = {
        "TinyStoriesV2-GPT4-train.txt":
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
        "TinyStoriesV2-GPT4-valid.txt":
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    }

    for fname, url in files.items():
        path = f"data/{fname}"
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"[跳过] {fname} ({size_mb:.0f} MB)")
        else:
            print(f"[下载] {fname} ...")
            import urllib.request
            urllib.request.urlretrieve(url, path)
            size_mb = os.path.getsize(path) / 1e6
            print(f"[完成] {fname} ({size_mb:.0f} MB)")

    print("✓ 数据就绪")


# ═══════════════════════════════════════════════════════════════
# Cell 3: 训练 Tokenizer
# ═══════════════════════════════════════════════════════════════

def cell_load_tokenizer():
    """加载仓库里预训练的 tokenizer（用优化版 BPE 训练在 TinyStories 样本上）。"""
    vocab_path = "checkpoints/tokenizer_vocab.pkl"
    merges_path = "checkpoints/tokenizer_merges.pkl"

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    print(f"[Tokenizer] 词表大小: {len(vocab)}")
    return vocab


# ═══════════════════════════════════════════════════════════════
# Cell 4: 训练模型
# ═══════════════════════════════════════════════════════════════

def cell_train_model():
    """用 small preset 训练模型，T4 上约 30-40 分钟。

    返回 (losses, perplexities, final_weights)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # ── 加载 tokenizer ──
    with open("checkpoints/tokenizer_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("checkpoints/tokenizer_merges.pkl", "rb") as f:
        merges = pickle.load(f)

    special_map = {}
    for tid, tbytes in vocab.items():
        s = tbytes.decode("utf-8", errors="replace")
        if s in ("<|endoftext|>",):
            special_map[s] = tid

    tokenizer = Tokenizer(vocab, merges, special_tokens=special_map)
    print(f"[Tokenizer] 词表大小: {tokenizer.vocab_size}")

    # ── 配置 ──
    config = ModelConfig.small()
    config.vocab_size = tokenizer.vocab_size
    print(f"[配置] {config.num_layers} 层, d_model={config.d_model}, "
          f"heads={config.num_heads}, kv_heads={config.num_kv_heads}, d_ff={config.d_ff}")
    print(f"[配置] max_steps={config.max_steps}, batch={config.batch_size}, "
          f"ctx={config.context_length}, lr={config.learning_rate}")

    # ── 加载数据 ──
    print(f"[数据] tokenize 中...")
    token_ids = load_and_tokenize("data/TinyStoriesV2-GPT4-train.txt", tokenizer)
    data = np.array(token_ids, dtype=np.int64)
    print(f"[数据] {len(data):,} tokens, "
          f"~{len(data) // (config.batch_size * config.context_length):,} batches")

    # ── 初始化权重 ──
    print(f"[模型] 初始化...")
    weights = _init_weights(config)
    weights = {k: v.to(device) for k, v in weights.items()}
    n_params = sum(p.numel() for p in weights.values())
    print(f"[模型] 参数量: {n_params:,} (~{n_params / 1e6:.1f}M)")

    # ── 优化器 ──
    params_list = [p for p in weights.values() if p.requires_grad]
    optimizer = AdamW(params_list, lr=config.learning_rate,
                      betas=(config.beta1, config.beta2),
                      weight_decay=config.weight_decay)

    # Causal mask
    causal_mask = torch.tril(torch.ones(config.context_length, config.context_length, device=device))

    # ── 训练循环 ──
    print(f"\n{'步数':>6s}  {'loss':>7s}  {'ppl':>7s}  {'lr':>8s}  {'耗时':>7s}")
    print("-" * 50)

    losses = []
    ppls = []
    step = 0
    total_loss = 0.0
    loss_count = 0
    log_every = 50
    t_start = time.time()

    while step < config.max_steps:
        lr = get_lr_cosine_schedule(step, config.warmup_steps, config.max_steps,
                                    config.learning_rate, config.learning_rate * 0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        inputs, targets = get_batch(data, config.batch_size, config.context_length)
        inputs, targets = inputs.to(device), targets.to(device)

        t0 = time.time()
        logits = transformer_lm(
            inputs,
            weights=weights,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            mask=causal_mask[:inputs.shape[1], :inputs.shape[1]],
        )
        loss = cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(params_list, config.grad_clip)
        optimizer.step()
        dt = time.time() - t0

        total_loss += loss.item()
        loss_count += 1

        if step % log_every == 0 or step == config.max_steps - 1:
            avg_loss = total_loss / loss_count
            avg_ppl = math.exp(avg_loss)
            losses.append((step, avg_loss))
            ppls.append((step, avg_ppl))
            elapsed = time.time() - t_start
            print(f"{step:6d}  {avg_loss:7.4f}  {avg_ppl:7.1f}  {lr:8.6f}  "
                  f"{elapsed:6.0f}s")
            total_loss = 0.0
            loss_count = 0

        step += 1

    total_time = time.time() - t_start
    print(f"\n[完成] 总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")

    # ── 保存模型 ──
    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint(weights, optimizer.state_dict(), step, "checkpoints/final.pt")
    print("[保存] checkpoints/final.pt")

    return losses, ppls, weights


def _init_weights(config: ModelConfig) -> dict[str, torch.Tensor]:
    """初始化权重 —— 跟 scripts/train.py 一样的逻辑。"""
    d_model = config.d_model
    num_layers = config.num_layers
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    d_ff = config.d_ff
    vocab_size = config.vocab_size

    d_qkv = (num_heads + 2 * num_kv_heads) * head_dim
    weights: dict[str, torch.Tensor] = {}
    weights["embedding"] = torch.randn(vocab_size, d_model) * 0.02

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
# Cell 5: 画 loss 曲线
# ═══════════════════════════════════════════════════════════════

def cell_plot_loss(losses, ppls):
    """画训练 loss/perplexity 曲线，保存到 assets/ 目录。"""
    os.makedirs("assets", exist_ok=True)

    steps_l, vals_l = zip(*losses)
    steps_p, vals_p = zip(*ppls)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps_l, vals_l, color="#2563eb", linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps_p, vals_p, color="#dc2626", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity (exp(loss))")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("assets/training_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[保存] assets/training_curve.png")


# ═══════════════════════════════════════════════════════════════
# Cell 6: 生成文本
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def cell_generate(weights, prompt="Once upon a time", max_tokens=150,
                  temperature=0.8, top_k=50, top_p=0.9):
    """用训练好的模型生成文本。"""
    import torch.nn.functional as F

    device = next(iter(weights.values())).device

    # 加载 tokenizer
    with open("checkpoints/tokenizer_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("checkpoints/tokenizer_merges.pkl", "rb") as f:
        merges = pickle.load(f)

    special_map = {}
    for tid, tbytes in vocab.items():
        s = tbytes.decode("utf-8", errors="replace")
        if s in ("<|endoftext|>",):
            special_map[s] = tid

    tokenizer = Tokenizer(vocab, merges, special_tokens=special_map)

    # 从 config.small() 推断架构
    config = ModelConfig.small()

    # Encode prompt
    token_ids = tokenizer.encode_special(prompt)
    context = torch.tensor([token_ids], dtype=torch.long, device=device)

    generated = []
    for _ in range(max_tokens):
        logits = transformer_lm(
            context,
            weights=weights,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
        )
        logits = logits[:, -1, :].squeeze(0)

        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        elif temperature == 0:
            next_id = logits.argmax(dim=-1).item()
            generated.append(next_id)
            context = torch.cat([context, torch.tensor([[next_id]], device=device)], dim=1)
            continue

        probs = F.softmax(logits, dim=-1)

        if top_k is not None and top_k > 0:
            topk_probs, topk_indices = torch.topk(probs, k=min(top_k, tokenizer.vocab_size))
            probs = torch.zeros_like(probs)
            probs.scatter_(0, topk_indices, topk_probs)

        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumulative > top_p
            cutoff_mask[0] = False
            cutoff_idx = cutoff_mask.int().argmax(dim=-1)
            if cutoff_idx < len(sorted_probs) - 1:
                sorted_probs[cutoff_idx + 1:] = 0
            probs = torch.zeros_like(probs)
            probs.scatter_(0, sorted_indices, sorted_probs)

        probs = probs / probs.sum()
        next_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_id)
        context = torch.cat([context, torch.tensor([[next_id]], device=device)], dim=1)

    output = tokenizer.decode(generated)
    return output


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cell_setup()
    cell_download_data()
    cell_load_tokenizer()
    losses, ppls, weights = cell_train_model()
    cell_plot_loss(losses, ppls)

    print("\n[生成示例]")
    for prompt in ["Once upon a time", "The little girl", "I like to"]:
        text = cell_generate(weights, prompt=prompt, max_tokens=100)
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {text}")
