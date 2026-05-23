"""模型组件模块的单元测试。"""
import pytest
import torch
from lmfromzero.norm import rms_norm, RMSNorm
from lmfromzero.activations import silu, swiglu_ffn
from lmfromzero.block import transformer_block
from lmfromzero.model import transformer_lm


# ═══════════════════════════════════════════════════════════════
# RMSNorm 测试
# ═══════════════════════════════════════════════════════════════

class TestRMSNorm:

    def test_output_shape(self):
        """RMSNorm 不改变输入形状。"""
        x = torch.randn(2, 8, 64)
        w = torch.ones(64)
        out = rms_norm(x, w)
        assert out.shape == x.shape

    def test_unit_norm_output(self):
        """权重为 1 时，输出的 RMS 应接近 1。"""
        x = torch.randn(4, 16, 128)
        w = torch.ones(128)
        out = rms_norm(x, w)
        rms_out = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert torch.allclose(rms_out, torch.ones_like(rms_out), atol=1e-4)

    def test_weight_scaling(self):
        """权重放大 2 倍，输出也应放大 2 倍。"""
        x = torch.randn(2, 5, 32)
        out1 = rms_norm(x, torch.ones(32))
        out2 = rms_norm(x, 2.0 * torch.ones(32))
        assert torch.allclose(out2, 2.0 * out1, atol=1e-6)

    def test_gradient(self):
        """RMSNorm 应支持反向传播。"""
        x = torch.randn(2, 4, 16, requires_grad=True)
        w = torch.ones(16, requires_grad=True)
        out = rms_norm(x, w)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()
        assert w.grad is not None

    def test_module_interface(self):
        """RMSNorm nn.Module 包装类。"""
        layer = RMSNorm(d_model=64)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == x.shape


# ═══════════════════════════════════════════════════════════════
# SiLU 测试
# ═══════════════════════════════════════════════════════════════

class TestSiLU:

    def test_output_shape(self):
        """SiLU 不改变输入形状。"""
        x = torch.randn(3, 7, 11)
        out = silu(x)
        assert out.shape == x.shape

    def test_zero_input(self):
        """x=0 时，SiLU(0) = 0。"""
        x = torch.zeros(5)
        out = silu(x)
        assert torch.allclose(out, torch.zeros(5), atol=1e-6)

    def test_large_positive(self):
        """x 很大时，SiLU(x) ≈ x。"""
        x = torch.tensor([10.0, 100.0])
        out = silu(x)
        # sigmoid(10) ≈ 0.99995, sigmoid(100) ≈ 1.0
        assert torch.allclose(out, x, atol=1e-3)

    def test_large_negative(self):
        """x 很负时，SiLU(x) ≈ 0。"""
        x = torch.tensor([-10.0, -100.0])
        out = silu(x)
        assert (out.abs() < 1e-3).all()

    def test_gradient(self):
        """SiLU 应支持反向传播。"""
        x = torch.randn(10, requires_grad=True)
        out = silu(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()


# ═══════════════════════════════════════════════════════════════
# SwiGLU FFN 测试
# ═══════════════════════════════════════════════════════════════

class TestSwiGLUFFN:

    def test_output_shape(self):
        """输出形状应与输入相同。"""
        batch, seq_len, d_model, d_ff = 2, 4, 64, 256
        x = torch.randn(batch, seq_len, d_model)
        w1 = torch.randn(d_model, d_ff)
        w2 = torch.randn(d_model, d_ff)
        w3 = torch.randn(d_ff, d_model)
        out = swiglu_ffn(x, w1, w2, w3)
        assert out.shape == (batch, seq_len, d_model)

    def test_gradient(self):
        """SwiGLU FFN 应支持反向传播。"""
        batch, seq_len, d_model, d_ff = 2, 4, 64, 256
        x = torch.randn(batch, seq_len, d_model, requires_grad=True)
        w1 = torch.randn(d_model, d_ff, requires_grad=True)
        w2 = torch.randn(d_model, d_ff, requires_grad=True)
        w3 = torch.randn(d_ff, d_model, requires_grad=True)
        out = swiglu_ffn(x, w1, w2, w3)
        loss = out.sum()
        loss.backward()
        for name, param in [("x", x), ("w1", w1), ("w2", w2), ("w3", w3)]:
            assert param.grad is not None, f"{name} 应有梯度"

    def test_batch_independence(self):
        """batch 中不同样本应独立计算。"""
        d_model, d_ff = 32, 128
        x1 = torch.randn(1, 2, d_model)
        x2 = torch.randn(1, 2, d_model)
        w1 = torch.randn(d_model, d_ff)
        w2 = torch.randn(d_model, d_ff)
        w3 = torch.randn(d_ff, d_model)

        out1 = swiglu_ffn(x1, w1, w2, w3)
        out2 = swiglu_ffn(x2, w1, w2, w3)
        # 不同输入应有不同输出
        assert not torch.allclose(out1, out2, atol=1e-4)


# ═══════════════════════════════════════════════════════════════
# Transformer Block 测试
# ═══════════════════════════════════════════════════════════════

class TestTransformerBlock:

    @pytest.fixture
    def setup(self):
        batch, seq_len, d_model = 2, 8, 384
        num_heads, num_kv_heads = 6, 3
        head_dim = d_model // num_heads  # 64
        d_ff = 1536

        d_qkv = (num_heads + 2 * num_kv_heads) * head_dim

        weights = {
            "attn_qkv": torch.randn(d_model, d_qkv),
            "attn_out": torch.randn(d_model, d_model),
            "ffn_w1": torch.randn(d_model, d_ff),
            "ffn_w2": torch.randn(d_model, d_ff),
            "ffn_w3": torch.randn(d_ff, d_model),
            "ln1_w": torch.ones(d_model),
            "ln2_w": torch.ones(d_model),
        }

        x = torch.randn(batch, seq_len, d_model)
        return x, weights, num_heads, num_kv_heads

    def test_output_shape(self, setup):
        """输出形状应与输入相同。"""
        x, weights, nh, nkv = setup
        out = transformer_block(x, weights, nh, nkv, 10000.0, 1e-6)
        assert out.shape == x.shape

    def test_residual_connection(self, setup):
        """残差连接：零权重时输出应等于输入。"""
        x, weights, nh, nkv = setup
        # 所有投影权重置零 → 残差连接保证输出 = 输入
        zero_weights = {k: torch.zeros_like(v) if k.startswith("attn") or k.startswith("ffn") else v
                       for k, v in weights.items()}
        # 注意：归一化权重置零会导致输出为 0，所以保留 ln 权重为 1
        out = transformer_block(x, zero_weights, nh, nkv, 10000.0, 1e-6)
        # 投影全零，残差连接使输出 ≈ 输入
        assert torch.allclose(out, x, atol=1e-5)

    def test_gradient(self, setup):
        """Transformer Block 应支持完整的反向传播。"""
        x, weights, nh, nkv = setup
        x.requires_grad = True
        for k in weights:
            weights[k].requires_grad = True

        out = transformer_block(x, weights, nh, nkv, 10000.0, 1e-6)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None and not torch.isnan(x.grad).any()
        for k in weights:
            assert weights[k].grad is not None, f"{k} 应有梯度"

    def test_with_causal_mask(self, setup):
        """带因果掩码的 Transformer Block。"""
        x, weights, nh, nkv = setup
        seq_len = x.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        out = transformer_block(x, weights, nh, nkv, 10000.0, 1e-6, mask=mask)
        assert out.shape == x.shape

    def test_no_nan(self, setup):
        """合理的输入不应产生 NaN。"""
        x, weights, nh, nkv = setup
        out = transformer_block(x, weights, nh, nkv, 10000.0, 1e-6)
        assert not torch.isnan(out).any()


# ═══════════════════════════════════════════════════════════════
# Transformer LM 测试
# ═══════════════════════════════════════════════════════════════

class TestTransformerLM:

    @pytest.fixture
    def setup(self):
        batch, seq_len = 2, 8
        vocab_size = 1000
        d_model = 384
        num_layers = 2
        num_heads, num_kv_heads = 6, 3
        head_dim = d_model // num_heads
        d_ff = d_model * 4

        d_qkv = (num_heads + 2 * num_kv_heads) * head_dim

        weights = {}
        weights["embedding"] = torch.randn(vocab_size, d_model)
        weights["final_norm"] = torch.ones(d_model)
        weights["lm_head"] = torch.randn(d_model, vocab_size)

        for i in range(num_layers):
            prefix = f"block_{i}_"
            weights[f"{prefix}attn_qkv"] = torch.randn(d_model, d_qkv) * 0.02
            weights[f"{prefix}attn_out"] = torch.randn(d_model, d_model) * 0.02
            weights[f"{prefix}ffn_w1"] = torch.randn(d_model, d_ff) * 0.02
            weights[f"{prefix}ffn_w2"] = torch.randn(d_model, d_ff) * 0.02
            weights[f"{prefix}ffn_w3"] = torch.randn(d_ff, d_model) * 0.02
            weights[f"{prefix}ln1_w"] = torch.ones(d_model)
            weights[f"{prefix}ln2_w"] = torch.ones(d_model)

        token_ids = torch.randint(0, vocab_size, (batch, seq_len))

        return token_ids, weights, {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
        }

    def test_output_shape(self, setup):
        """输出形状 = [batch, seq_len, vocab_size]。"""
        token_ids, weights, config = setup
        logits = transformer_lm(token_ids, weights, **config)
        assert logits.shape == (token_ids.shape[0], token_ids.shape[1], weights["embedding"].shape[0])

    def test_gradient(self, setup):
        """完整 LM 应支持反向传播。"""
        token_ids, weights, config = setup
        for k in weights:
            weights[k].requires_grad = True

        logits = transformer_lm(token_ids, weights, **config)
        loss = logits.sum()
        loss.backward()

        for k in weights:
            assert weights[k].grad is not None, f"{k} 应有梯度"

    def test_no_nan(self, setup):
        """合理的权重初始化不应产生 NaN。"""
        token_ids, weights, config = setup
        logits = transformer_lm(token_ids, weights, **config)
        assert not torch.isnan(logits).any()

    def test_with_causal_mask(self, setup):
        """带因果掩码的完整前向传播。"""
        token_ids, weights, config = setup
        seq_len = token_ids.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        logits = transformer_lm(token_ids, weights, mask=mask, **config)
        assert not torch.isnan(logits).any()
