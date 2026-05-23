"""
注意力机制 —— Transformer 最核心的东西。

本模块从简单到复杂实现了四个函数：
1. scaled_dot_product_attention  —— 最基础的单头注意力
2. rope                           —— 旋转位置编码
3. multihead_self_attention       —— 多头注意力 + GQA（不带 RoPE）
4. multihead_self_attention_with_rope —— 完整版：MHA + GQA + RoPE

关于 GQA（分组查询注意力）：
- Query 有 num_heads 个头，但 Key/Value 只有 num_kv_heads 个头
- 每几个 Q 头共享一组 K/V，比如 num_heads=12, num_kv_heads=4，就是 3 个 Q 共享一组 KV
- 这样做 KV cache 显存能省好几倍，而模型质量几乎不掉
- LLaMA 2/3、Mistral 都在用这个技巧

参考：
- "Attention Is All You Need" (Vaswani et al., 2017)
- "RoFormer" (Su et al., 2021)
- LLaMA 2 (Touvron et al., 2023)
"""
import math
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1. 缩放点积注意力
# ═══════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """最基础的注意力：softmax(Q @ K^T / sqrt(d_k)) @ V。

    为啥要除以 sqrt(d_k)？因为 d_k 大了以后，Q 和 K 点积的值也会变大，
    softmax 会被推到饱和区（梯度接近零）。除以 sqrt(d_k) 让方差保持稳定，
    这也是"Scaled"这个名字的由来。

    mask 中 True/1 的位置保留，False/0 的位置用 -inf 屏蔽（比如不让看到未来 token）。
    """
    d_k = queries.shape[-1]
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask.bool() if mask.dtype == torch.bool else mask == 0, float("-inf"))

    # 如果某一行全被 mask 了（全是 -inf），softmax 会出 NaN，
    # 用 nan_to_num 把 NaN 变成 0，这些位置不参与输出。
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    return torch.matmul(attn_weights, values)


# ═══════════════════════════════════════════════════════════════
# 2. RoPE（旋转位置编码）
# ═══════════════════════════════════════════════════════════════

def rope(
    x: torch.Tensor,
    token_positions: torch.Tensor | None = None,
    theta: float = 10000.0,
) -> torch.Tensor:
    """旋转位置编码 —— 通过对向量做旋转来注入位置信息。

    大致思路（不太严格地说）：
    - 把 d_k 维向量拆成 d_k/2 对，每对看成 2D 平面上的一个点
    - 第 i 对旋转的角度 = 位置编号 * (theta 的某个指数衰减)
    - 离得近的 token 旋转差小，离得远的旋转差大
    - 两个向量的点积只跟它们的相对位置有关，跟绝对位置无关

    具体计算用的是三角恒等式而不是真的矩阵旋转：
        x_rot[2i]   = x[2i]*cos - x[2i+1]*sin
        x_rot[2i+1] = x[2i]*sin + x[2i+1]*cos
    """
    seq_len = x.shape[-2]
    d_k = x.shape[-1]

    if token_positions is None:
        token_positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    else:
        token_positions = token_positions.to(device=x.device, dtype=x.dtype)

    # 频率：freqs[i] = 1 / (theta^(2i/d_k))
    indices = torch.arange(0, d_k, 2, device=x.device, dtype=x.dtype)
    freqs = 1.0 / (theta ** (indices / d_k))

    # 每个位置的旋转角 = pos * freq
    angles = token_positions.unsqueeze(-1) * freqs.unsqueeze(0)
    cos = angles.cos()
    sin = angles.sin()

    # 每个 cos/sin 值要复制一份给相邻的奇偶维度（interleave）
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)

    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_2d = cos[..., 0::2]
    sin_2d = sin[..., 0::2]

    rotated_even = x_even * cos_2d - x_odd * sin_2d
    rotated_odd = x_even * sin_2d + x_odd * cos_2d

    # 把偶/奇拼回交替排列
    result = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(start_dim=-2)
    return result.type_as(x)


# ═══════════════════════════════════════════════════════════════
# 3. 多头自注意力（含 GQA，不含 RoPE）
# ═══════════════════════════════════════════════════════════════

def multihead_self_attention(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    w_out: torch.Tensor,
    num_heads: int = 12,
    num_kv_heads: int = 4,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """多头自注意力，Q/K/V 投影合并成一次矩阵乘法（性能优化）。

    普通做法要做三次矩阵乘法：x@W_q, x@W_k, x@W_v。
    我们把 W_q, W_k, W_v 拼成一个大矩阵 w_qkv，一次 x@w_qkv 搞定三个，
    GPU 上大矩阵乘法比三个小矩阵快很多。

    支持 GQA：如果 num_kv_heads < num_heads，K 和 V 会先按少头算，
    然后通过 expand+reshape 复制到跟 Q 一样多的头。

    输入 [batch, seq_len, d_model]，输出同形状。
    """
    batch, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    assert num_heads % num_kv_heads == 0, \
        f"num_heads ({num_heads}) 必须能被 num_kv_heads ({num_kv_heads}) 整除"
    groups_per_kv = num_heads // num_kv_heads

    # 一次搞定 Q/K/V：w_qkv 里按顺序排着 Q 部分、K 部分、V 部分
    qkv = torch.matmul(x, w_qkv)

    d_q = num_heads * head_dim
    d_kv = num_kv_heads * head_dim
    q = qkv[..., :d_q]
    k = qkv[..., d_q:d_q + d_kv]
    v = qkv[..., d_q + d_kv:]

    # 拆成多头形状 [batch, heads, seq_len, head_dim]
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # GQA 扩展：把 K/V 的头数从 num_kv_heads 复制到 num_heads
    if num_kv_heads < num_heads:
        k = k.unsqueeze(2).expand(batch, num_kv_heads, groups_per_kv, seq_len, head_dim)
        k = k.reshape(batch, num_heads, seq_len, head_dim)
        v = v.unsqueeze(2).expand(batch, num_kv_heads, groups_per_kv, seq_len, head_dim)
        v = v.reshape(batch, num_heads, seq_len, head_dim)

    attn_output = scaled_dot_product_attention(q, k, v, mask=mask)

    # 拼回头 → 输出投影
    attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, d_model)
    return torch.matmul(attn_output, w_out)


# ═══════════════════════════════════════════════════════════════
# 4. 多头自注意力（含 GQA + RoPE）—— 完整版
# ═══════════════════════════════════════════════════════════════

def multihead_self_attention_with_rope(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    w_out: torch.Tensor,
    num_heads: int = 12,
    num_kv_heads: int = 4,
    rope_theta: float = 10000.0,
    mask: torch.Tensor | None = None,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """完整版多头自注意力：GQA + RoPE 都在里面。

    跟上面那个不带 RoPE 的版本就多了一步：Q 和 K 投影完之后、算注意力之前，
    各自过一次 RoPE 旋转。

    为啥只对 Q 和 K 加 RoPE，不对 V 加？
    因为位置信息是通过注意力分数（Q·K^T）来传递的，V 只是被注意力权重加权平均，
    给它加位置信息没有额外好处 —— 数学上可以证明对 V 做旋转不会改变输出。
    """
    batch, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    groups_per_kv = num_heads // num_kv_heads

    qkv = torch.matmul(x, w_qkv)
    d_q = num_heads * head_dim
    d_kv = num_kv_heads * head_dim
    q = qkv[..., :d_q]
    k = qkv[..., d_q:d_q + d_kv]
    v = qkv[..., d_q + d_kv:]

    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # 关键一步：Q 和 K 加 RoPE
    q = rope(q, token_positions=token_positions, theta=rope_theta)
    k = rope(k, token_positions=token_positions, theta=rope_theta)

    if num_kv_heads < num_heads:
        k = k.unsqueeze(2).expand(batch, num_kv_heads, groups_per_kv, seq_len, head_dim)
        k = k.reshape(batch, num_heads, seq_len, head_dim)
        v = v.unsqueeze(2).expand(batch, num_kv_heads, groups_per_kv, seq_len, head_dim)
        v = v.reshape(batch, num_heads, seq_len, head_dim)

    attn_output = scaled_dot_product_attention(q, k, v, mask=mask)
    attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, d_model)
    return torch.matmul(attn_output, w_out)
