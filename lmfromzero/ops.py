"""
一些最底层的运算 —— softmax、交叉熵、梯度裁剪。
虽然 PyTorch 都有现成的，但自己写一遍才知道里面发生了什么。
比如为啥 softmax 要先减最大值，交叉熵和 log_softmax 是什么关系。
"""
import math
import torch
import torch.nn.functional as F


def softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """对最后一维做 softmax，把 logits 变成概率分布。

    直接 exp 的话万一有个值特别大（比如 1000），直接就溢出了。
    所以先减去每行的最大值，把所有值拉到 <=0，再 exp 就安全了。
    """
    x_max = logits.max(dim=dim, keepdim=True).values
    shifted = logits - x_max
    exp_shifted = shifted.exp()
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """算交叉熵损失，语言模型训练就用这个。

    说白了就是：模型输出的 logits 经过 softmax 变成概率，
    然后看正确答案对应的那个概率够不够高。概率越高，loss 越小。

    输入可能是 [batch, vocab_size] 或 [batch, seq_len, vocab_size]，
    后者会先展平成 2D 再算。
    """
    if logits.dim() > 2:
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
    return F.cross_entropy(logits, targets, reduction='mean')


def cross_entropy_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """手写版交叉熵，纯粹为了看清楚里面到底算了什么。

    分两步：
    1. log_softmax —— 等于 x - max - log(sum(exp(x - max)))，都是数值稳定的写法
    2. NLL —— 拿出 target 位置的值，取负号，求平均
    """
    x_max = logits.max(dim=-1, keepdim=True).values
    shifted = logits - x_max
    log_sum_exp = shifted.exp().sum(dim=-1, keepdim=True).log()
    log_probs = shifted - log_sum_exp

    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


def gradient_clipping(params: list[torch.Tensor], max_l2_norm: float) -> None:
    """全局梯度裁剪，防止梯度爆炸。训练 LLM 基本都会加这个。

    怎么做的：把所有参数的梯度拼起来算一个总的 L2 范数，
    如果超过阈值，就等比缩放，让一步跨出去的距离不会太远。

    注意这是 in-place 的，直接改 param.grad，不返回值。
    """
    device = params[0].device
    total_norm_sq = torch.tensor(0.0, device=device)
    for p in params:
        if p.grad is not None:
            total_norm_sq = total_norm_sq + p.grad.norm() ** 2

    total_norm = total_norm_sq.sqrt()
    scale = max_l2_norm / (total_norm + 1e-6)
    if scale < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
