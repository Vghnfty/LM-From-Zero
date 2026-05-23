"""
Checkpoint 保存与恢复 —— 训练中断后能接着练。

保存的东西：
- 模型权重（扁平字典）
- 优化器状态（动量、二阶矩这些）
- 当前步数
- 配置（可选）

直接用 torch.save / torch.load，cpu/gpu 随便切换。
"""
import torch
from pathlib import Path
from typing import Optional


def save_checkpoint(
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict,
    step: int,
    filepath: str | Path,
    config: Optional[dict] = None,
) -> None:
    """保存训练状态到磁盘，自动建目录。"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "step": step,
    }
    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, str(filepath))


def load_checkpoint(
    filepath: str | Path,
    model_weights: dict[str, torch.Tensor],
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> int:
    """从 checkpoint 恢复训练状态。

    模型权重原地写入 model_weights 字典（in-place copy），
    优化器状态也一并恢复。返回之前保存的步数。
    """
    checkpoint = torch.load(str(filepath), map_location=map_location, weights_only=False)

    saved_state = checkpoint["model_state"]
    for key, value in saved_state.items():
        if key in model_weights:
            model_weights[key].copy_(value)
        else:
            raise KeyError(
                f"Checkpoint 里有 '{key}' 这个键，但当前模型权重里没有。"
                f"当前模型的键：{list(model_weights.keys())}"
            )

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint["step"]
