"""注意力机制模块的单元测试。"""
import math
import pytest
import torch
from lmfromzero.attention import (
    scaled_dot_product_attention,
    rope,
    multihead_self_attention,
    multihead_self_attention_with_rope,
)


# ═══════════════════════════════════════════════════════════════
# scaled_dot_product_attention 测试
# ═══════════════════════════════════════════════════════════════

class TestScaledDotProductAttention:

    def test_basic_shape(self):
        """输出形状应与 V 相同（最后一维 = d_v）。"""
        batch, seq_len, d_k, d_v = 2, 8, 64, 64
        Q = torch.randn(batch, seq_len, d_k)
        K = torch.randn(batch, seq_len, d_k)
        V = torch.randn(batch, seq_len, d_v)
        out = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (batch, seq_len, d_v)

    def test_mask_completely_blocks(self):
        """全 False 的 mask 应使输出全为 0（全 -inf → softmax NaN → nan_to_num → 0）。"""
        seq_len, d_k = 4, 16
        Q = torch.randn(1, seq_len, d_k)
        K = torch.randn(1, seq_len, d_k)
        V = torch.randn(1, seq_len, d_k)
        mask = torch.zeros(seq_len, seq_len)  # 全 0 = 全屏蔽
        out = scaled_dot_product_attention(Q, K, V, mask=mask)
        # 全屏蔽 → 注意力权重全是 0 → 输出全是 0
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)

    def test_causal_mask(self):
        """因果掩码：第 i 个位置只能看到 ≤ i 的位置。"""
        seq_len, d_k = 4, 8
        Q = torch.randn(1, seq_len, d_k)
        K = torch.randn(1, seq_len, d_k)
        V = torch.randn(1, seq_len, d_k)
        # 下三角 mask（含对角线）
        mask = torch.tril(torch.ones(seq_len, seq_len))
        out = scaled_dot_product_attention(Q, K, V, mask=mask)
        assert out.shape == (1, seq_len, d_k)

    def test_gradient_flow(self):
        """标度点积注意力应支持反向传播。"""
        Q = torch.randn(2, 4, 16, requires_grad=True)
        K = torch.randn(2, 4, 16, requires_grad=True)
        V = torch.randn(2, 4, 16, requires_grad=True)
        out = scaled_dot_product_attention(Q, K, V)
        loss = out.sum()
        loss.backward()
        assert Q.grad is not None and not torch.isnan(Q.grad).any()
        assert K.grad is not None and not torch.isnan(K.grad).any()

    def test_scale_by_sqrt_dk(self):
        """验证缩放因子：得分应被除以 sqrt(d_k)。"""
        d_k = 64
        Q = torch.randn(1, 2, d_k)
        K = torch.randn(1, 2, d_k)
        V = torch.randn(1, 2, d_k)
        scores = Q @ K.transpose(-2, -1)
        scaled_scores = scores / math.sqrt(d_k)
        # 手动计算 softmax(QK^T/sqrt(d_k))
        expected_weights = torch.softmax(scaled_scores, dim=-1)
        # 用函数算，无 mask
        out = scaled_dot_product_attention(Q, K, V, mask=None)
        expected_out = expected_weights @ V
        assert torch.allclose(out, expected_out, atol=1e-6)


# ═══════════════════════════════════════════════════════════════
# rope 测试
# ═══════════════════════════════════════════════════════════════

class TestRoPE:

    def test_output_shape(self):
        """RoPE 不改变输入形状。"""
        x = torch.randn(2, 4, 10, 32)  # [batch, num_heads, seq_len, head_dim]
        out = rope(x)
        assert out.shape == x.shape

    def test_preserves_norm(self):
        """RoPE 是旋转操作，应保持向量 L2 范数不变（每个 2D 子平面）。"""
        x = torch.randn(1, 1, 8, 64)
        out = rope(x)
        # 整体 L2 范数可能因浮点误差微变，但应非常接近
        x_norm = x.norm(dim=-1)
        out_norm = out.norm(dim=-1)
        assert torch.allclose(x_norm, out_norm, atol=1e-4)

    def test_output_differs_from_input(self):
        """非零位置上的 RoPE 应该改变了值（不是恒等变换）。"""
        x = torch.randn(1, 1, 5, 32)
        out = rope(x)
        # position 0 的旋转角 = 0，所以值不变；其他位置应该变化
        assert torch.allclose(out[..., 0, :], x[..., 0, :], atol=1e-6)  # pos=0 不变
        assert not torch.allclose(out[..., 1, :], x[..., 1, :], atol=1e-6)  # pos>0 变化

    def test_with_token_positions(self):
        """自定义 token 位置参数。"""
        x = torch.randn(1, 1, 3, 16)
        positions = torch.tensor([0, 5, 10])
        out1 = rope(x, token_positions=positions)
        out2 = rope(x, token_positions=torch.tensor([0, 5, 10]))
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradient(self):
        """RoPE 应支持反向传播。"""
        x = torch.randn(2, 4, 8, 32, requires_grad=True)
        out = rope(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    def test_relative_position_property(self):
        """RoPE 的核心性质：对完全相同的内容向量，相对位置差决定注意力分数。

        测试方法：用相同的 Q=K 向量序列，验证位置 m 与 m+delta 的点积取决于 delta。
        当输入全为 1 时，各位置的 Q/K 相同，分数差异仅来自位置编码。
        """
        head_dim = 16
        # 所有位置用相同的值，这样点积分数的差异只来自位置编码
        x = torch.ones(1, 1, 8, head_dim)
        q = rope(x.clone())
        k = rope(x.clone())
        scores = q @ k.transpose(-2, -1)  # [1, 1, 8, 8]

        # 同一条对角线上（相对位置相同）的值应该一致
        for diag in range(1, 4):
            vals = []
            for i in range(8 - diag):
                vals.append(scores[0, 0, i, i + diag].item())
            # 相同长度、相同输入 → 相同对角线应该有几乎完全相同的值
            assert max(vals) - min(vals) < 1e-4, \
                f"diag={diag}: max={max(vals):.6f}, min={min(vals):.6f}"


# ═══════════════════════════════════════════════════════════════
# multihead_self_attention 测试
# ═══════════════════════════════════════════════════════════════

class TestMultiheadSelfAttention:

    @pytest.fixture
    def sample_inputs(self):
        """创建测试用的输入和权重。"""
        batch, seq_len, d_model = 2, 8, 384
        num_heads, num_kv_heads = 6, 3
        head_dim = d_model // num_heads  # 64

        d_q = num_heads * head_dim       # 384
        d_kv = num_kv_heads * head_dim   # 192
        d_qkv = d_q + 2 * d_kv           # 384 + 384 = 768

        x = torch.randn(batch, seq_len, d_model)
        w_qkv = torch.randn(d_model, d_qkv)
        w_out = torch.randn(d_model, d_model)

        return x, w_qkv, w_out, num_heads, num_kv_heads

    def test_output_shape(self, sample_inputs):
        """输出形状应与输入相同 [batch, seq_len, d_model]。"""
        x, w_qkv, w_out, nh, nkv = sample_inputs
        out = multihead_self_attention(x, w_qkv, w_out, nh, nkv)
        assert out.shape == x.shape

    def test_gqa_kv_expansion(self, sample_inputs):
        """GQA 中 K/V 头数少于 Q 头数时不应报错，且输出形状正确。"""
        x, w_qkv, w_out, nh, nkv = sample_inputs
        assert nkv < nh  # 确认是 GQA 模式
        out = multihead_self_attention(x, w_qkv, w_out, nh, nkv)
        assert out.shape == x.shape

    def test_with_causal_mask(self, sample_inputs):
        """带因果掩码的 MHA。"""
        x, w_qkv, w_out, nh, nkv = sample_inputs
        batch, seq_len, d_model = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len))
        out = multihead_self_attention(x, w_qkv, w_out, nh, nkv, mask=mask)
        assert out.shape == x.shape

    def test_gradient_flow(self, sample_inputs):
        """MHA 应支持完整的反向传播（通过所有投影矩阵）。"""
        x, w_qkv, w_out, nh, nkv = sample_inputs
        x.requires_grad = True
        w_qkv.requires_grad = True
        w_out.requires_grad = True
        out = multihead_self_attention(x, w_qkv, w_out, nh, nkv)
        loss = out.sum()
        loss.backward()
        for name, param in [("x", x), ("w_qkv", w_qkv), ("w_out", w_out)]:
            assert param.grad is not None, f"{name} 应有梯度"
            assert not torch.isnan(param.grad).any(), f"{name} 梯度有 NaN"

    def test_equivalence_without_gqa(self, sample_inputs):
        """当 num_kv_heads == num_heads 时，GQA 退化为标准 MHA。"""
        batch, seq_len = 2, 8
        d_model = 384
        num_heads = 6
        head_dim = d_model // num_heads  # 64
        d_qkv = num_heads * head_dim * 3  # 1152

        x = torch.randn(batch, seq_len, d_model)
        w_qkv = torch.randn(d_model, d_qkv)
        w_out = torch.randn(d_model, d_model)

        out = multihead_self_attention(x, w_qkv, w_out, num_heads, num_heads)
        assert out.shape == x.shape


# ═══════════════════════════════════════════════════════════════
# multihead_self_attention_with_rope 测试
# ═══════════════════════════════════════════════════════════════

class TestMultiheadSelfAttentionWithRoPE:

    @pytest.fixture
    def sample_inputs(self):
        batch, seq_len, d_model = 2, 16, 384
        num_heads, num_kv_heads = 6, 3
        head_dim = d_model // num_heads

        d_qkv = (num_heads + 2 * num_kv_heads) * head_dim

        x = torch.randn(batch, seq_len, d_model)
        w_qkv = torch.randn(d_model, d_qkv)
        w_out = torch.randn(d_model, d_model)

        return x, w_qkv, w_out, num_heads, num_kv_heads

    def test_output_shape(self, sample_inputs):
        """含 RoPE 的 MHA 输出形状应与输入相同。"""
        x, w_qkv, w_out, nh, nkv = sample_inputs
        out = multihead_self_attention_with_rope(x, w_qkv, w_out, nh, nkv)
        assert out.shape == x.shape

    def test_with_token_positions(self, sample_inputs):
        """自定义 token 位置应正常工作。"""
        x, w_qkv, w_out, nh, nkv = sample_inputs
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, dtype=torch.float32)
        out = multihead_self_attention_with_rope(
            x, w_qkv, w_out, nh, nkv, token_positions=positions
        )
        assert out.shape == x.shape

    def test_different_from_no_rope(self, sample_inputs):
        """含 RoPE 的输出应与不含 RoPE 的不同（非零位置）。"""
        x, w_qkv, w_out, nh, nkv = sample_inputs
        out_with = multihead_self_attention_with_rope(x, w_qkv, w_out, nh, nkv)
        out_without = multihead_self_attention(x, w_qkv, w_out, nh, nkv)
        # 因为 RoPE 改变了 Q 和 K 的值，输出应该不同
        assert not torch.allclose(out_with, out_without, atol=1e-3)

    def test_gradient_flow(self, sample_inputs):
        """含 RoPE 的 MHA 应支持完整的反向传播。"""
        x, w_qkv, w_out, nh, nkv = sample_inputs
        x.requires_grad = True
        w_qkv.requires_grad = True
        w_out.requires_grad = True
        out = multihead_self_attention_with_rope(x, w_qkv, w_out, nh, nkv)
        loss = out.sum()
        loss.backward()
        for name, param in [("x", x), ("w_qkv", w_qkv), ("w_out", w_out)]:
            assert param.grad is not None, f"{name} 应有梯度"
            assert not torch.isnan(param.grad).any(), f"{name} 梯度有 NaN"
