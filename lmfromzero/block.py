"""
Transformer Block —— LLaMA 风格的 Pre-LN Transformer 块。

每个 Block 里面就两个东西：
1. 带 RoPE 的多头自注意力（GQA）
2. SwiGLU 前馈网络

用的是 Pre-LN（预层归一化）结构：
    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))

Pre-LN vs Post-LN 的区别：
- Post-LN（原始论文）：x = LN(x + Sublayer(x))，LN 放在残差之后
- Pre-LN（LLaMA）：   x = x + Sublayer(LN(x))，LN 放在子层之前

Pre-LN 训练更稳定，因为梯度不需要穿过 LN 再回传，现代 LLM 基本都选 Pre-LN。
"""
import torch
from lmfromzero.attention import multihead_self_attention_with_rope
from lmfromzero.norm import rms_norm
from lmfromzero.activations import swiglu_ffn


def transformer_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    num_heads: int,
    num_kv_heads: int,
    rope_theta: float,
    rms_norm_eps: float,
    mask: torch.Tensor | None = None,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """一个完整的 Transformer Block。

    weights 字典里应该有这些键：
    - "attn_qkv": QKV 合并投影 [d_model, d_qkv]
    - "attn_out": 注意力输出投影 [d_model, d_model]
    - "ffn_w1":   SwiGLU 门控 [d_model, d_ff]
    - "ffn_w2":   SwiGLU 上投影 [d_model, d_ff]
    - "ffn_w3":   SwiGLU 下投影 [d_ff, d_model]
    - "ln1_w":    注意力前的 RMSNorm [d_model]
    - "ln2_w":    FFN 前的 RMSNorm [d_model]
    """
    # 子层 1：多头自注意力
    residual = x
    x_normed = rms_norm(x, weights["ln1_w"], eps=rms_norm_eps)
    attn_out = multihead_self_attention_with_rope(
        x_normed,
        w_qkv=weights["attn_qkv"],
        w_out=weights["attn_out"],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=rope_theta,
        mask=mask,
        token_positions=token_positions,
    )
    x = residual + attn_out

    # 子层 2：SwiGLU 前馈网络
    residual = x
    x_normed = rms_norm(x, weights["ln2_w"], eps=rms_norm_eps)
    ffn_out = swiglu_ffn(
        x_normed,
        w1=weights["ffn_w1"],
        w2=weights["ffn_w2"],
        w3=weights["ffn_w3"],
    )
    x = residual + ffn_out

    return x
