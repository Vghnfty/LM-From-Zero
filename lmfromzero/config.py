"""
模型配置 —— 所有超参数集中放在这，改起来方便。

用 dataclass 的好处：IDE 能自动补全，能序列化，有默认值不用每次都传一堆参数。
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """LLaMA 风格语言模型的配置。

    有两套预设：
    - 默认（~150M）：12 层，d_model=768，完整训练用
    - small（~55M）：6 层，d_model=512，快速验证/Colab 用
    """

    # 词汇表
    vocab_size: int = 16384

    # 序列
    max_seq_len: int = 512

    # Transformer 结构
    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 12
    num_kv_heads: int = 4
    d_ff: int = 3072

    # 归一化
    rms_norm_eps: float = 1e-6

    # RoPE
    rope_theta: float = 10000.0

    # 训练超参数
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 1000
    max_steps: int = 50000
    batch_size: int = 32
    grad_clip: float = 1.0

    # 数据
    context_length: int = 512

    def __post_init__(self):
        """跑之前检查下参数有没有写错。"""
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) 必须能被 num_heads ({self.num_heads}) 整除"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) 必须能被 num_kv_heads ({self.num_kv_heads}) 整除"
        assert self.max_seq_len >= self.context_length, \
            f"max_seq_len ({self.max_seq_len}) 必须 >= context_length ({self.context_length})"

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度 = d_model / num_heads。"""
        return self.d_model // self.num_heads

    @classmethod
    def small(cls) -> "ModelConfig":
        """Colab/T4 快速训练用的缩小版配置（~55M 参数）。

        d_model=512、6 层、context=256，T4 上大约 30-40 分钟跑完 5000 步。
        """
        return cls(
            vocab_size=4096,
            max_seq_len=256,
            num_layers=6,
            d_model=512,
            num_heads=8,
            num_kv_heads=4,
            d_ff=1536,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            learning_rate=5e-4,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            warmup_steps=500,
            max_steps=5000,
            batch_size=32,
            grad_clip=1.0,
            context_length=256,
        )
