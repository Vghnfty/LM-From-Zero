"""
AdamW 优化器 —— 把权重衰减从自适应学习率里拆出来。

Adam（Kingma & Ba, 2014）是个很好的优化器：
- 每个参数有自己的自适应学习率 = lr / (sqrt(v) + eps)
- 动量 m 平滑梯度方向，二阶矩 v 跟踪梯度大小

但 Adam + L2 正则化有个坑：对于自适应优化器来说，L2 正则化和权重衰减不是等价的
（而 SGD 里它们就是等价的）。Loshchilov & Hutter (2017) 指出这个问题后提出 AdamW，
把 weight decay 从梯度里分离出来，直接对参数做衰减：
    θ = θ - lr * (m_hat / sqrt(v_hat)) - lr * wd * θ

这个改法在实际训练中稳定多了，LLaMA/GPT 系列全在用。

继承 torch.optim.Optimizer，用法跟官方的 AdamW 一模一样。
"""
import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """AdamW 优化器，用法跟 torch.optim.AdamW 一样。

    默认的 beta2=0.95 是 LLaMA 的设置（普通 AdamW 一般是 0.999），
    memory 更短，更适合作文生成长序列的任务。
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """一步 AdamW 更新。

        对每个参数 p：
        1. m = β1*m + (1-β1)*grad       （一阶动量）
        2. v = β2*v + (1-β2)*grad²      （二阶矩）
        3. m_hat = m / (1-β1^t)          （偏差校正）
        4. v_hat = v / (1-β2^t)          （偏差校正）
        5. p = p - lr*wd*p               （解耦的权重衰减）
        6. p = p - lr * m_hat/(√v_hat+ε) （梯度更新）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 首次调用，初始化动量
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                m, v = state["m"], state["v"]
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # 先衰减权重，再更新参数（顺序不影响结果）
                p.mul_(1 - lr * weight_decay)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss
