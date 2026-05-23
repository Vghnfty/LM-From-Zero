"""
训练 BPE Tokenizer 的入口脚本。

跑完后会生成两个文件：
    {output}_vocab.pkl  —— 词表
    {output}_merges.pkl —— 合并列表

用法：python scripts/train_tokenizer.py --input data/train.txt --output checkpoints/tokenizer
"""
import argparse
import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lmfromzero.tokenizer import train_bpe, Tokenizer


def main():
    parser = argparse.ArgumentParser(description="训练 BPE Tokenizer")
    parser.add_argument("--input", type=str, required=True, help="训练语料文件路径")
    parser.add_argument("--output", type=str, default="checkpoints/tokenizer", help="输出文件前缀")
    parser.add_argument("--vocab_size", type=int, default=16384, help="目标词表大小")
    parser.add_argument("--special_tokens", type=str, nargs="*",
                        default=["<|endoftext|>"], help="特殊 token")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[错误] 找不到输入文件: {args.input}")
        sys.exit(1)

    # 读语料
    print(f"[1/3] 读取语料: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"      文本长度: {len(text):,} 字符, {len(text.encode('utf-8')):,} 字节")

    # 训练 BPE
    print(f"[2/3] 训练 BPE (目标词表大小: {args.vocab_size})...")
    allowed_special = set(args.special_tokens)
    vocab, merges = train_bpe(text, vocab_size=args.vocab_size, allowed_special=allowed_special)
    print(f"      实际词表: {len(vocab)} 个 token, 合并 {len(merges)} 次")

    # 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    vocab_path = f"{args.output}_vocab.pkl"
    merges_path = f"{args.output}_merges.pkl"

    print(f"[3/3] 保存:")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"      {vocab_path}")
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    print(f"      {merges_path}")

    # 快速验证：encode → decode 往返
    special_map = {}
    for token_str in allowed_special:
        for tid, tbytes in vocab.items():
            if tbytes.decode("utf-8", errors="replace") == token_str:
                special_map[token_str] = tid
                break

    tokenizer = Tokenizer(vocab, merges, special_tokens=special_map)
    test_text = "Hello, world!"
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    print(f"\n[验证] 往返测试:")
    print(f"  原文:     {test_text!r}")
    print(f"  Token 数: {len(ids)}")
    print(f"  解码:     {decoded!r}")
    assert test_text == decoded, "往返测试失败!"
    print(f"  结果:     通过!")


if __name__ == "__main__":
    main()
