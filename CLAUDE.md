# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 提供本仓库的工作指南。

## 项目概述

LMfromzero 是一个约 1.5 亿参数的 LLaMA 风格 Transformer 语言模型，仅使用 PyTorch 作为张量后端从零构建。所有核心组件（softmax、注意力、RoPE、RMSNorm、SwiGLU、AdamW、BPE tokenizer）均为手工实现。项目同时作为 CS336（Stanford）Assignment 1 测试套件的实现基座，测试代码位于 `cs336_repo/`。

## 常用命令

```bash
# 安装（可编辑模式 + 开发依赖）
pip install -e ".[dev]"

# 运行全部 87 个单元测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_attention.py -v

# 运行单个测试函数
pytest tests/test_attention.py::TestScaledDotProductAttention::test_causal_mask -v

# 训练 tokenizer
python scripts/train_tokenizer.py --input data/TinyStoriesV2-GPT4-train.txt --output checkpoints/tokenizer --vocab_size 16384

# 训练模型（CPU 快速验证）
python scripts/train.py --train_data data/TinyStoriesV2-GPT4-train.txt --tokenizer_prefix checkpoints/tokenizer --device cpu --max_steps 500 --batch_size 4 --context_length 128

# 训练模型（GPU 完整训练）
python scripts/train.py --train_data data/TinyStoriesV2-GPT4-train.txt --tokenizer_prefix checkpoints/tokenizer --device cuda --max_steps 50000

# 文本生成（交互模式）
python scripts/generate.py --checkpoint checkpoints/final.pt --tokenizer_prefix checkpoints/tokenizer --interactive
```

## 架构

### 组件依赖关系（自底向上）

```
ops.py (softmax, cross_entropy, gradient_clipping)
  ├─ tokenizer.py (BPE 训练 + Tokenizer 类)
  ├─ norm.py (rms_norm)
  ├─ activations.py (silu, swiglu_ffn)
  ├─ attention.py (sdpa → rope → mha → mha_with_rope)
  │     依赖: ops.py
  ├─ block.py (transformer_block: Pre-LN MHA + SwiGLU FFN)
  │     依赖: attention.py, norm.py, activations.py
  ├─ model.py (transformer_lm: embedding → N×block → RMSNorm → lm_head)
  │     依赖: block.py, norm.py
  ├─ optimizer.py (AdamW, 继承自 torch.optim.Optimizer)
  ├─ scheduler.py (余弦学习率调度 + 线性预热)
  ├─ data.py (load_and_tokenize, get_batch)
  ├─ checkpoint.py (保存/加载训练状态)
  └─ generate.py (temperature / top-k / top-p 采样)
```

### 权重字典约定

所有模型组件均为**函数式实现**（非 `nn.Module` 子类）。每个函数接受 `weights: dict[str, Tensor]` 参数。扁平权重字典命名规则：

| 键名 | 形状 | 说明 |
|-----|-------|-------------|
| `embedding` | `[vocab_size, d_model]` | token 嵌入表 |
| `block_{i}_attn_qkv` | `[d_model, d_qkv]` | 第 i 层合并 Q/K/V 投影 |
| `block_{i}_attn_out` | `[d_model, d_model]` | 第 i 层注意力输出投影 |
| `block_{i}_ffn_w1` | `[d_model, d_ff]` | SwiGLU 门控投影 |
| `block_{i}_ffn_w2` | `[d_model, d_ff]` | SwiGLU 上投影 |
| `block_{i}_ffn_w3` | `[d_ff, d_model]` | SwiGLU 下投影 |
| `block_{i}_ln1_w` | `[d_model]` | 注意力前的 RMSNorm 权重 |
| `block_{i}_ln2_w` | `[d_model]` | FFN 前的 RMSNorm 权重 |
| `final_norm` | `[d_model]` | 最终 RMSNorm 权重 |
| `lm_head` | `[d_model, vocab_size]` | 输出投影 |

其中 `d_qkv = (num_heads + 2*num_kv_heads) * head_dim`。层号从 0 开始。

### 关键设计决策

- **GQA（分组查询注意力）**：`num_kv_heads=4`，`num_heads=12`，每 3 个 Q 头共享一组 K/V，KV cache 显存减少约 3 倍。
- **合并 QKV 投影**：单次矩阵乘法 `x @ w_qkv` 代替三次分别投影，GPU 利用率更好。
- **Pre-LN + RMSNorm**：归一化在子层之前（而非之后），RMSNorm 省去均值中心化以提速。
- **SwiGLU FFN**：`(SiLU(x @ W1) * (x @ W2)) @ W3`，门控激活优于普通 ReLU。
- **RoPE 仅施加于 Q 和 K**：位置信息通过注意力分数传递，对 V 加 RoPE 无效果。

### CS336 测试套件（`cs336_repo/`）

`cs336_repo/` 目录包含原始 CS336 assignment 测试框架。`cs336_repo/tests/adapters.py` 中的适配器桥接了 assignment 期望的函数签名与本地实现。`cs336_repo/CLAUDE.md` 包含教学助手指南——它指示 AI 代理不得为学生编写代码。

### 模型默认配置

| 参数 | 值 |
|-----------|-------|
| vocab_size | 16384 |
| d_model | 768 |
| num_layers | 12 |
| num_heads | 12 |
| num_kv_heads | 4 |
| d_ff | 3072 |
| max_seq_len / context_length | 512 |
| learning_rate | 3e-4 |
| weight_decay | 0.1 |
| warmup_steps | 1000 |
| max_steps | 50000 |
| batch_size | 32 |
| grad_clip | 1.0 |
