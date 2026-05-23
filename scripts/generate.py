"""
文本生成脚本 —— 加载训练好的模型，输入 prompt，让它接着往下写。

交互模式（可以反复输入）：
    python scripts/generate.py --checkpoint checkpoints/final.pt --tokenizer_prefix checkpoints/tokenizer --interactive

单次生成：
    python scripts/generate.py --checkpoint checkpoints/final.pt --tokenizer_prefix checkpoints/tokenizer --prompt "Once upon a time"

生成参数：
    --temperature  温度，默认 0.8。越低越保守（接近复读），越高越放飞
    --top_k        Top-K 采样，默认 50
    --top_p        Nucleus 采样，默认 0.9
    --max_tokens   最多生成多少个 token，默认 200
"""
import argparse
import os
import pickle
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lmfromzero.config import ModelConfig
from lmfromzero.tokenizer import Tokenizer
from lmfromzero.model import transformer_lm
from lmfromzero.generate import generate


def main():
    parser = argparse.ArgumentParser(description="用训练好的语言模型生成文本")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--tokenizer_prefix", type=str, required=True, help="tokenizer 文件前缀")
    parser.add_argument("--prompt", type=str, default="", help="开头提示词")
    parser.add_argument("--max_tokens", type=int, default=200, help="最多新生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度 (0=贪心)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus 采样")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # 加载 Tokenizer
    print("[1/3] 加载 Tokenizer...")
    with open(f"{args.tokenizer_prefix}_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(f"{args.tokenizer_prefix}_merges.pkl", "rb") as f:
        merges = pickle.load(f)

    special_map = {}
    for tid, tbytes in vocab.items():
        s = tbytes.decode("utf-8", errors="replace")
        if s in ("<|endoftext|>",):
            special_map[s] = tid

    tokenizer = Tokenizer(vocab, merges, special_tokens=special_map)
    print(f"       词表大小: {tokenizer.vocab_size}")

    # 加载模型权重
    print("[2/3] 加载模型...")
    config = ModelConfig()
    config.vocab_size = tokenizer.vocab_size

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    weights = checkpoint["model_state"]
    weights = {k: v.to(device) for k, v in weights.items()}
    train_step = checkpoint.get("step", "未知")
    print(f"       训练步数: {train_step}")
    print(f"       参数量: ~{sum(p.numel() for p in weights.values()) / 1e6:.1f}M")

    # 前向函数
    print("[3/3] 准备生成...")

    def model_fn(token_ids: torch.Tensor) -> torch.Tensor:
        return transformer_lm(
            token_ids,
            weights=weights,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
        )

    if args.interactive:
        print("\n" + "=" * 50)
        print("交互模式 (输入 'quit' 退出)")
        print("=" * 50)
        while True:
            prompt = input("\n[输入] > ").strip()
            if prompt.lower() == "quit":
                break
            if not prompt:
                continue

            token_ids = tokenizer.encode_special(prompt)
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

            generated_ids = generate(
                model_fn, input_tensor,
                max_new_tokens=args.max_tokens,
                vocab_size=tokenizer.vocab_size,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            )

            output_text = tokenizer.decode(generated_ids)
            print(f"[输出] {prompt}{output_text}")
    else:
        prompt = args.prompt or "The "
        token_ids = tokenizer.encode_special(prompt)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        print(f"\n[提示] {prompt}")
        print(f"[生成中...] (temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p})")

        generated_ids = generate(
            model_fn, input_tensor,
            max_new_tokens=args.max_tokens,
            vocab_size=tokenizer.vocab_size,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )

        output_text = tokenizer.decode(generated_ids)
        print(f"\n[输出]\n{prompt}{output_text}")


if __name__ == "__main__":
    main()
