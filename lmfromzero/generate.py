"""
文本生成 —— 自回归地一个 token 一个 token 地往外吐。

三种采样策略，可以混着用：
1. Temperature：控制输出有多"放飞"。=0 就是贪心（每次选概率最高的），
   >1 会让低概率 token 被选中的机会变大，输出更随机。
2. Top-K：只从概率最高的 K 个里面选，把那些长尾噪音全砍掉。
3. Top-P（Nucleus Sampling）：从概率从高到低累加，加到超过 P 就停，
   后面的全砍掉。集合大小是动态的，比 Top-K 更灵活。

实际用的时候一般 Top-P + Temperature 一起上。
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model_fn,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    vocab_size: int,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    seed: int | None = None,
) -> list[int]:
    """自回归生成。

    每次循环：
    1. 模型前向 → 拿到最后一个位置的 logits
    2. Temperature 缩放
    3. Top-K / Top-P 过滤
    4. 从过滤后的分布里采样一个 token
    5. 拼回去，下一轮

    返回新生成的 token ID 列表（不包含 prompt 那部分）。
    """
    if seed is not None:
        torch.manual_seed(seed)

    generated: list[int] = []
    context = token_ids

    for _ in range(max_new_tokens):
        logits = model_fn(context)
        logits = logits[:, -1, :]    # 只要最后一个位置 [1, vocab_size]
        logits = logits.squeeze(0)   # [vocab_size]

        # Temperature
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        elif temperature == 0:
            # 贪心，直接取 argmax
            next_id = logits.argmax(dim=-1).item()
            generated.append(next_id)
            context = torch.cat(
                [context, torch.tensor([[next_id]], device=context.device, dtype=context.dtype)],
                dim=1,
            )
            continue

        probs = F.softmax(logits, dim=-1)

        # Top-K：只保留概率最大的 K 个
        if top_k is not None and top_k > 0:
            topk_probs, topk_indices = torch.topk(probs, k=min(top_k, vocab_size))
            probs = torch.zeros_like(probs)
            probs.scatter_(0, topk_indices, topk_probs)

        # Top-P：贪心地取到累积概率达到 P 为止
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumulative > top_p
            cutoff_mask[0] = False  # 至少保留一个
            cutoff_idx = cutoff_mask.int().argmax(dim=-1)
            if cutoff_idx < len(sorted_probs) - 1:
                sorted_probs[cutoff_idx + 1:] = 0
            probs = torch.zeros_like(probs)
            probs.scatter_(0, sorted_indices, sorted_probs)

        # 重新归一化，然后按概率采样
        probs = probs / probs.sum()
        next_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_id)
        context = torch.cat(
            [context, torch.tensor([[next_id]], device=context.device, dtype=context.dtype)],
            dim=1,
        )

    return generated
