"""
学习率调度 —— 线性预热 + 余弦退火，绝大多数 LLM 训练的标配。

分两阶段：
1. 预热期：lr 从 0 线性涨到 peak_lr
   为什么要从 0 开始？刚初始化完的模型是随机的，大的 lr 会让梯度过大乱跑，
   先小步快走找方向，然后再加速。

2. 退火期：lr 沿余弦曲线从 peak_lr 衰减到 min_lr
   训练后期需要精细调参，大 lr 容易跳过最优解。余弦形状比线性下降更平滑。

公式很简单：
    lr(t) = peak_lr * t / warmup           (t < warmup)
    lr(t) = peak_lr * 0.5*(1+cos(π*r))    (t >= warmup)
    其中 r = (t - warmup) / (total - warmup)
"""
import math


def get_lr_cosine_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    peak_lr: float,
    min_lr: float = 0.0,
) -> float:
    """返回当前步的学习率。step 从 0 开始数。"""
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    elif step >= max_steps:
        return min_lr
    else:
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * cosine_decay
