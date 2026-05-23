"""
激活函数与 SwiGLU 前馈网络。

SiLU = x * sigmoid(x)，也叫 Swish，是个平滑的非线性函数。
比 ReLU 好的地方在于：
- 不是硬截断，负数区域也有很小的输出（不是零）
- 处处可导，梯度流更顺畅

SwiGLU 就是"带门控的 Swish 版 FFN"：
    output = (SiLU(x @ W1) * (x @ W2)) @ W3
W1 那条路算出一个 0~x 的"门控信号"，控制 W2 那条路的信息能通过多少。
LLaMA、PaLM 这些模型都在用，比普通 ReLU FFN 明显更好。

参考：
- Shazeer (2020) "GLU Variants Improve Transformer"
- Touvron et al. (2023) "LLaMA"
"""
import torch
import torch.nn.functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU = x * sigmoid(x)，PyTorch 有现成的 F.silu，直接调用。"""
    return F.silu(x)


def swiglu_ffn(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """SwiGLU 前馈网络。

    三步走：
    1. gate = SiLU(x @ w1)  —— 门控值，决定哪些信息放行
    2. up   = x @ w2         —— 上投影，把 d_model 扩到 d_ff
    3. out  = (gate * up) @ w3 —— 门控后再投影回 d_model

    w1, w2 形状是 [d_model, d_ff]，w3 是 [d_ff, d_model]。
    输入 [..., d_model]，输出也是同形状。
    """
    gate = F.silu(torch.matmul(x, w1))
    up = torch.matmul(x, w2)
    return torch.matmul(gate * up, w3)
