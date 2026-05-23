"""
基础运算模块的单元测试。
"""
import pytest
import torch
from lmfromzero.ops import softmax, cross_entropy, gradient_clipping


# ═══════════════════════════════════════════════════════════════
# softmax 测试
# ═══════════════════════════════════════════════════════════════

class TestSoftmax:
    """测试 softmax 的正确性和数值稳定性。"""

    def test_basic(self):
        """基本功能：输出应为概率分布（每行和为 1，值在 [0, 1]）。"""
        x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        out = softmax(x, dim=-1)

        # 每行和应该接近 1
        assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-6)
        # 所有值在 [0, 1]
        assert (out >= 0).all() and (out <= 1).all()

    def test_numerical_stability(self):
        """数值稳定性：大数值输入不应产生 NaN 或 inf。"""
        x = torch.tensor([[1000.0, 1000.0, 1000.0], [-1000.0, 1000.0, 0.0]])
        out = softmax(x, dim=-1)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-6)

    def test_equiv_to_pytorch(self):
        """结果应与 PyTorch 内置 softmax 一致。"""
        x = torch.randn(4, 10)
        out = softmax(x, dim=-1)
        expected = torch.softmax(x, dim=-1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_arbitrary_dim(self):
        """支持任意维度上的 softmax。"""
        x = torch.randn(2, 3, 4, 5)
        for dim in [0, 1, 2, 3]:
            out = softmax(x, dim=dim)
            assert torch.allclose(out.sum(dim=dim), torch.ones(1), atol=1e-6)

    def test_gradient(self):
        """反向传播是否正确。"""
        x = torch.randn(3, 5, requires_grad=True)
        out = softmax(x, dim=-1)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ═══════════════════════════════════════════════════════════════
# cross_entropy 测试
# ═══════════════════════════════════════════════════════════════

class TestCrossEntropy:
    """测试交叉熵损失。"""

    def test_perfect_prediction(self):
        """完美预测时损失应接近 0。"""
        logits = torch.tensor([[0.01, 100.0, 0.01]])  # 第 1 类得分极高
        targets = torch.tensor([1])
        loss = cross_entropy(logits, targets)
        assert loss.item() < 0.01

    def test_random_prediction(self):
        """随机均匀预测时，损失应接近 log(vocab_size)。"""
        vocab_size = 100
        logits = torch.zeros(1, vocab_size)  # 所有类别得分相同
        targets = torch.tensor([0])
        loss = cross_entropy(logits, targets)
        # softmax 后每类概率 = 1/vocab_size，loss = -log(1/vocab_size) = log(vocab_size)
        expected = torch.tensor(vocab_size).log()
        assert torch.allclose(loss, expected, atol=1e-3)

    def test_batch(self):
        """批处理模式：batch 维度上的平均值。"""
        logits = torch.randn(8, 50)
        targets = torch.randint(0, 50, (8,))
        loss = cross_entropy(logits, targets)
        assert loss.ndim == 0  # 标量
        assert loss.item() > 0

    def test_3d_input(self):
        """支持 3D 输入 [batch, seq_len, vocab_size]。"""
        logits = torch.randn(4, 16, 1000)
        targets = torch.randint(0, 1000, (4, 16))
        loss = cross_entropy(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gradient_flow(self):
        """确保梯度能正常通过损失函数回传。"""
        logits = torch.randn(4, 10, requires_grad=True)
        targets = torch.randint(0, 10, (4,))
        loss = cross_entropy(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


# ═══════════════════════════════════════════════════════════════
# gradient_clipping 测试
# ═══════════════════════════════════════════════════════════════

class TestGradientClipping:
    """测试全局 L2 范数梯度裁剪。"""

    def test_no_clip_when_under_threshold(self):
        """梯度范数小于阈值时，不应被修改。"""
        x = torch.randn(3, requires_grad=True)
        x.sum().backward()
        grad_before = x.grad.clone()
        gradient_clipping([x], max_l2_norm=100.0)
        assert torch.allclose(x.grad, grad_before, atol=1e-6)

    def test_clip_when_over_threshold(self):
        """梯度范数超过阈值时，应被缩放到刚好等于阈值。"""
        x = torch.randn(3, requires_grad=True)
        (x * 100).sum().backward()
        gradient_clipping([x], max_l2_norm=1.0)
        clipped_norm = x.grad.norm()
        assert torch.allclose(clipped_norm, torch.tensor(1.0), atol=1e-5)

    def test_multi_param(self):
        """多个参数的梯度应一起参与全局范数计算和缩放。"""
        a = torch.randn(5, requires_grad=True)
        b = torch.randn(5, requires_grad=True)
        (a.sum() + b.sum()).backward()
        norm_before = torch.sqrt(a.grad.norm() ** 2 + b.grad.norm() ** 2)
        gradient_clipping([a, b], max_l2_norm=norm_before.item() * 0.5)
        norm_after = torch.sqrt(a.grad.norm() ** 2 + b.grad.norm() ** 2)
        # 裁剪后范数应约为阈值
        assert torch.allclose(norm_after, norm_before * 0.5, atol=1e-5)

    def test_mixed_grad_none(self):
        """参数中有 grad 为 None 的情况不应报错。"""
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        x.sum().backward()  # 只有 x 有梯度
        gradient_clipping([x, y], max_l2_norm=1.0)  # 不应抛异常
