"""
Byte-level BPE（字节对编码）Tokenizer。

BPE 是现代 LLM 的标配分词方式，GPT 系列、LLaMA 都在用。核心思路很简单：
1. 从最底层开始：256 个字节就是最初始的 token
2. 反复找文本中出现次数最多的相邻 token pair，把它俩合并成一个新 token
3. 一直合并到词表大小达到目标为止

这样做的好处：
- 常见词保持完整不拆分（压缩率高）
- 罕见词被拆成子词单元（不会 OOV）
- 从没见过的词也能用字节拼出来（彻底消灭 UNK）

参考：Sennrich et al. (2016), GPT-2 tokenizer
"""
from collections import Counter
from typing import Optional


# ── BPE 训练 ────────────────────────────────────────────────────

def train_bpe(
    text: str,
    vocab_size: int,
    allowed_special: Optional[set[str]] = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从文本训练一个 BPE tokenizer。

    整个过程：
    1. 文本 → UTF-8 字节 → 每个字节一个初始 token
    2. 初始词表 = 0~255，每个 ID 对应一个字节
    3. 统计所有相邻 pair 的频次
    4. 取出最高频的 pair，合并成新 token，分一个新 ID
    5. 扫描序列把刚才那个 pair 全部替换成新 ID
    6. 回到第 3 步，直到词表达标

    返回 vocab（{id: bytes}）和 merges（合并顺序列表）。
    merges 的顺序就是优先级 —— 排在前面的先合并。
    """
    if allowed_special is None:
        allowed_special = set()

    # 文本 → 字节 → 初始 token（每个都是单字节）
    text_bytes = text.encode("utf-8")
    tokens = [bytes([b]) for b in text_bytes]
    ids = list(range(len(tokens)))

    # 初始词表：0~255 = 单字节
    vocab = {b: bytes([b]) for b in range(256)}

    # 特殊 token 放在词表末尾，不参与合并
    special_start = vocab_size - len(allowed_special)
    special_tokens = {}
    for i, token_str in enumerate(sorted(allowed_special)):
        special_tokens[token_str] = special_start + i

    # 统计初始 pair 频率
    pair_counts = Counter()
    for i in range(len(ids) - 1):
        pair_counts[(ids[i], ids[i + 1])] += 1

    merges: list[tuple[bytes, bytes]] = []
    next_id = 256

    # 主循环：一直合并到词表满
    while next_id < special_start:
        if not pair_counts:
            break

        (id_a, id_b), _count = pair_counts.most_common(1)[0]
        new_id = next_id
        next_id += 1

        token_a = vocab[id_a]
        token_b = vocab[id_b]
        vocab[new_id] = token_a + token_b
        merges.append((token_a, token_b))

        # 扫描替换：把所有 (id_a, id_b) → new_id
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == id_a and ids[i + 1] == id_b:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids

        # 重建计数（简单但 O(n) 每步）
        pair_counts = Counter()
        for i in range(len(ids) - 1):
            pair_counts[(ids[i], ids[i + 1])] += 1

    # 把特殊 token 塞进词表
    for token_str, token_id in special_tokens.items():
        vocab[token_id] = token_str.encode("utf-8")

    return vocab, merges


# ── Tokenizer 类 ──────────────────────────────────────────────────

class Tokenizer:
    """BPE Tokenizer —— 把文本变成 token ID，也能变回来。

    用法很简单：
        tokenizer = Tokenizer(vocab, merges)
        ids = tokenizer.encode("hello world")
        text = tokenizer.decode(ids)
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[dict[str, int]] = None,
    ):
        self.vocab = vocab
        self.special_tokens = special_tokens or {}

        # 合并优先级：{(bytes_a, bytes_b): rank}，rank 越小越优先
        self.merges = {}
        for rank, (a, b) in enumerate(merges):
            self.merges[(a, b)] = rank

        self.special_token_to_id = special_tokens or {}
        self.special_id_to_token = {v: k for k, v in self.special_token_to_id.items()}

        # decode 用的反向映射
        self.id_to_byte = {i: b for i, b in vocab.items() if i not in self.special_id_to_token}
        for token_str, tid in self.special_token_to_id.items():
            self.id_to_byte[tid] = token_str.encode("utf-8")

    def encode(self, text: str) -> list[int]:
        """把文本编码成 token ID 序列。

        步骤：
        1. UTF-8 编码成字节 → 每个字节一个初始 token
        2. 贪心地合并：每次找 current tokens 里 rank 最小的相邻 pair → 合并
        3. 重复直到没得合并了
        4. 把最终的 bytes token 映射回 ID
        """
        if not text:
            return []

        # 字节序列化
        text_bytes = text.encode("utf-8")
        tokens: list[bytes] = [bytes([b]) for b in text_bytes]

        # 贪心合并 —— 每次合并优先级最高的 pair
        while len(tokens) >= 2:
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(tokens) - 1):
                rank = self.merges.get((tokens[i], tokens[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break

            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

        # bytes → ID
        byte_to_id: dict[bytes, int] = {}
        for tid, tbytes in self.id_to_byte.items():
            if tid not in self.special_id_to_token:
                byte_to_id[tbytes] = tid

        ids = []
        for token_bytes in tokens:
            tid = byte_to_id.get(token_bytes)
            if tid is not None:
                ids.append(tid)
            else:
                # 极少见：合并结果不在词表里，回退到逐字节
                for b in token_bytes:
                    ids.append(b)

        return ids

    def decode(self, ids: list[int]) -> str:
        """把 token ID 序列变回文本。"""
        token_bytes = bytearray()
        for tid in ids:
            tbytes = self.id_to_byte.get(tid)
            if tbytes is not None:
                token_bytes.extend(tbytes)
            else:
                token_bytes.extend(f"[UNK:{tid}]".encode("utf-8"))

        return token_bytes.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_byte)

    def encode_special(self, text: str) -> list[int]:
        """带特殊 token 识别的编码。

        比如文本里出现了 "<|endoftext|>"，它应该被当成一个 token，
        而不是逐字符拆开编码。这个方法会先按特殊 token 把文本切开，
        普通部分走正常 encode，特殊部分直接用预留的 ID。
        """
        if not self.special_tokens:
            return self.encode(text)

        special_patterns = list(self.special_tokens.keys())
        # 按长度降序排，长的优先匹配（防止 <|eos|> 抢在 <|eos_token|> 前面）
        special_patterns.sort(key=len, reverse=True)

        result_ids = []
        i = 0
        while i < len(text):
            matched = False
            for sp in special_patterns:
                if text.startswith(sp, i):
                    result_ids.append(self.special_tokens[sp])
                    i += len(sp)
                    matched = True
                    break
            if not matched:
                end = len(text)
                for sp in special_patterns:
                    pos = text.find(sp, i)
                    if pos != -1 and pos < end:
                        end = pos
                if end > i:
                    result_ids.extend(self.encode(text[i:end]))
                i = end
        return result_ids
