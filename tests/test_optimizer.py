"""训练设施模块的单元测试。"""
import math
import os
import tempfile
import pytest
import torch
import numpy as np
from lmfromzero.data import get_batch
from lmfromzero.optimizer import AdamW
from lmfromzero.scheduler import get_lr_cosine_schedule
from lmfromzero.checkpoint import save_checkpoint, load_checkpoint


# ═══════════════════════════════════════════════════════════════
# get_batch 测试
# ═══════════════════════════════════════════════════════════════

class TestGetBatch:

    def test_output_shapes(self):
        """输入和标签的形状应正确。"""
        data = np.arange(1000, dtype=np.int64)
        inputs, targets = get_batch(data, batch_size=8, context_length=32)
        assert inputs.shape == (8, 32)
        assert targets.shape == (8, 32)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long

    def test_targets_shifted_by_one(self):
        """标签应是输入向右偏移一位。"""
        data = np.arange(100, dtype=np.int64)
        inputs, targets = get_batch(data, batch_size=1, context_length=5)
        # targets[i] 应该等于 inputs[i+1]
        for i in range(4):
            assert targets[0, i].item() == inputs[0, i + 1].item()

    def test_all_within_bounds(self):
        """所有 token ID 都应该在 data 范围内。"""
        data = np.arange(500, dtype=np.int64)
        inputs, targets = get_batch(data, batch_size=16, context_length=20)
        assert inputs.min() >= 0 and inputs.max() < 500
        assert targets.min() >= 0 and targets.max() < 500

    def test_insufficient_data_raises(self):
        """数据不够长时应抛出错误。"""
        data = np.arange(10, dtype=np.int64)
        with pytest.raises(ValueError):
            get_batch(data, batch_size=4, context_length=100)

    def test_reproducible_with_seed(self):
        """相同随机种子应产生相同结果。"""
        data = np.arange(1000, dtype=np.int64)
        np.random.seed(42)
        inputs1, targets1 = get_batch(data, batch_size=4, context_length=8)
        np.random.seed(42)
        inputs2, targets2 = get_batch(data, batch_size=4, context_length=8)
        assert torch.equal(inputs1, inputs2)
        assert torch.equal(targets1, targets2)


# ═══════════════════════════════════════════════════════════════
# AdamW 测试
# ═══════════════════════════════════════════════════════════════

class TestAdamW:

    def test_init_and_step(self):
        """AdamW 应能正常初始化和执行一步更新。"""
        param = torch.randn(10, requires_grad=True)
        param_before = param.clone()
        optimizer = AdamW([param], lr=0.01, weight_decay=0.1)

        loss = param.sum()
        loss.backward()
        optimizer.step()

        # 参数值应该被更新了
        assert not torch.equal(param, param_before)

    def test_no_nan_update(self):
        """合理的梯度不应产生 NaN 参数。"""
        param = torch.randn(10, requires_grad=True)
        optimizer = AdamW([param], lr=0.001, weight_decay=0.01)

        for _ in range(10):
            optimizer.zero_grad()
            loss = param.sum()
            loss.backward()
            optimizer.step()
            assert not torch.isnan(param).any()

    def test_weight_decay_effect(self):
        """有 weight_decay 比没有时参数缩小更快。"""
        p1 = torch.ones(10, requires_grad=True)
        p2 = torch.ones(10, requires_grad=True)

        opt1 = AdamW([p1], lr=0.01, weight_decay=0.5)
        opt2 = AdamW([p2], lr=0.01, weight_decay=0.0)

        for _ in range(5):
            opt1.zero_grad()
            opt2.zero_grad()
            (p1 * 0).sum().backward()  # 零梯度，只有 weight decay 作用
            (p2 * 0).sum().backward()
            opt1.step()
            opt2.step()

        # 有 weight decay 的参数应该更小
        assert p1.abs().sum() < p2.abs().sum()

    def test_state_preserved(self):
        """迭代间动量状态应保留。"""
        param = torch.randn(10, requires_grad=True)
        optimizer = AdamW([param], lr=0.01)

        optimizer.zero_grad()
        param.sum().backward()
        optimizer.step()
        state = optimizer.state[param]
        assert "m" in state and "v" in state
        step1 = state["step"]

        optimizer.zero_grad()
        param.sum().backward()
        optimizer.step()
        assert state["step"] == step1 + 1


# ═══════════════════════════════════════════════════════════════
# get_lr_cosine_schedule 测试
# ═══════════════════════════════════════════════════════════════

class TestLRScheduler:

    def test_warmup_linear_increase(self):
        """预热阶段学习率应线性增长。"""
        lr0 = get_lr_cosine_schedule(0, warmup_steps=100, max_steps=1000, peak_lr=0.01)
        lr50 = get_lr_cosine_schedule(50, warmup_steps=100, max_steps=1000, peak_lr=0.01)
        lr100 = get_lr_cosine_schedule(100, warmup_steps=100, max_steps=1000, peak_lr=0.01)
        assert lr0 == 0.0
        assert lr50 == pytest.approx(0.005, rel=1e-5)
        assert lr100 == pytest.approx(0.01, rel=1e-5)

    def test_cosine_decay_end(self):
        """余弦退火结束时学习率应降到 min_lr。"""
        lr_end = get_lr_cosine_schedule(
            1000, warmup_steps=100, max_steps=1000, peak_lr=0.01, min_lr=1e-6
        )
        assert lr_end == pytest.approx(1e-6, rel=1e-3)

    def test_cosine_midpoint(self):
        """余弦退火中点（cos(π/2)=0）学习率应为 (peak+min)/2。"""
        warmup, max_steps = 100, 1100
        mid = warmup + (max_steps - warmup) // 2  # = 600
        lr_mid = get_lr_cosine_schedule(
            mid, warmup_steps=warmup, max_steps=max_steps, peak_lr=0.01, min_lr=0.0
        )
        assert lr_mid == pytest.approx(0.005, rel=1e-5)

    def test_beyond_max_steps(self):
        """超过 max_steps 后返回 min_lr。"""
        lr = get_lr_cosine_schedule(
            2000, warmup_steps=100, max_steps=1000, peak_lr=0.01, min_lr=1e-6
        )
        assert lr == pytest.approx(1e-6, rel=1e-3)

    def test_zero_warmup(self):
        """warmup_steps=0 时直接从余弦退火开始。"""
        lr = get_lr_cosine_schedule(0, warmup_steps=0, max_steps=100, peak_lr=0.01)
        assert lr == pytest.approx(0.01, rel=1e-5)  # cos(0) = 1 → 0.5*(1+1) = 1


# ═══════════════════════════════════════════════════════════════
# checkpoint 测试
# ═══════════════════════════════════════════════════════════════

class TestCheckpoint:

    def test_save_and_load(self):
        """保存后加载应恢复相同的模型权重和步数。"""
        model_state = {
            "w1": torch.randn(10, 20),
            "w2": torch.randn(20, 5),
        }
        param = torch.randn(10, requires_grad=True)
        optimizer = AdamW([param], lr=0.01)
        optimizer.zero_grad()
        param.sum().backward()
        optimizer.step()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        try:
            save_checkpoint(model_state, optimizer.state_dict(), step=42, filepath=tmp_path)

            # 加载到一个新字典
            new_state = {k: torch.zeros_like(v) for k, v in model_state.items()}
            new_optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=0.01)
            loaded_step = load_checkpoint(tmp_path, new_state, new_optimizer)

            assert loaded_step == 42
            for k in model_state:
                assert torch.equal(model_state[k], new_state[k])
        finally:
            os.unlink(tmp_path)

    def test_config_preserved(self):
        """配置字典应能随 checkpoint 保存和读取。"""
        model_state = {"w": torch.randn(5, 5)}
        config = {"lr": 0.001, "batch_size": 32}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        try:
            save_checkpoint(
                model_state, optimizer_state={}, step=0,
                filepath=tmp_path, config=config,
            )
            ckpt = torch.load(tmp_path, weights_only=False)
            assert ckpt["config"] == config
        finally:
            os.unlink(tmp_path)
