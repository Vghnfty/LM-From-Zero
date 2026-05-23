"""
数据加载与批处理。

语言模型训练的数据组织方式比较特殊：
1. 整个文本文件 tokenize 成一个超长的一维序列
2. 每个训练步从里面随机截取 batch_size 个长度为 context_length 的片段
3. 输入 x = tokens[i : i+L]，标签 y = tokens[i+1 : i+L+1]（错开一位预测下一个）

说白了就是"给你前面 L 个 token，猜下一个是什么"，标准的自回归语言模型做法。
"""
import torch
import numpy as np
from lmfromzero.tokenizer import Tokenizer


def load_and_tokenize(filepath: str, tokenizer: Tokenizer) -> list[int]:
    """读取文本文件，tokenize 成一维 token ID 列表。"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return tokenizer.encode_special(text) if tokenizer.special_tokens else tokenizer.encode(text)


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从一维 token 数组里随机采一批训练样本。

    每个样本不能跨越数组边界（保证 token i 到 i+L+1 都在范围内），
    起始位置均匀随机挑选。

    返回 (inputs, targets)，都是 LongTensor，形状 [batch_size, context_length]。
    targets[i, j] = data[start_i + j + 1]，也就是 inputs 对应的"下一个 token"。
    """
    max_start = len(data) - context_length - 1
    if max_start <= 0:
        raise ValueError(
            f"数据太短 ({len(data)} tokens)，至少需要 {context_length + 1} tokens"
        )

    starts = np.random.randint(0, max_start, size=(batch_size,))
    inputs = []
    targets = []
    for start in starts:
        inputs.append(data[start : start + context_length])
        targets.append(data[start + 1 : start + 1 + context_length])

    return (
        torch.tensor(np.array(inputs), dtype=torch.long),
        torch.tensor(np.array(targets), dtype=torch.long),
    )
