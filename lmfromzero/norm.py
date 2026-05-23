"""
RMS 层归一化 —— LLaMA 系列的标配归一化方式。

跟标准 LayerNorm 的区别：
- LayerNorm：(x - mean) / std * weight + bias，又减均值又除方差
- RMSNorm： x / RMS(x) * weight，直接跳过均值中心化

省了一步计算，速度更快，而且实验证明对 Transformer 来说效果不差，
LLaMA 论文里甚至说某些任务上 RMSNorm 还能略优于 LayerNorm。

参考：Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
"""
import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对最后一维做 RMS 归一化。

    RMS(x) = sqrt(mean(x²) + eps)，然后 x / RMS(x) * weight。
    中间计算全部转 float32，避免 bf16/float16 下精度不够导致 NaN。
    """
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    x_normed = x.float() / rms
    return (x_normed * weight).type_as(x)


class RMSNorm(torch.nn.Module):
    """跟 nn.LayerNorm 用法一样的 RMSNorm，方便塞进 nn.Module 体系里。"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)
