"""BPE Tokenizer 模块的单元测试。"""
import pytest
from lmfromzero.tokenizer import train_bpe, Tokenizer


class TestTrainBPE:
    """测试 BPE 训练过程。"""

    def test_basic_vocab_size(self):
        """训练后的词表大小不应超过设定值。"""
        text = "hello world " * 100 + "foo bar baz " * 50
        vocab, merges = train_bpe(text, vocab_size=300)
        assert len(vocab) <= 300
        # 至少有 256 个基础字节 token
        assert len(vocab) >= 256

    def test_merges_ordered(self):
        """合并列表应该是按合并顺序排列的。"""
        text = "abababab abababab " * 100
        vocab, merges = train_bpe(text, vocab_size=280)
        # 高频率的 pair 应该先被合并
        assert len(merges) > 0

    def test_byte_tokens_present(self):
        """前 256 个 token 必须是单字节 token。"""
        text = "sample text for training " * 50
        vocab, mergers = train_bpe(text, vocab_size=300)
        for i in range(256):
            assert i in vocab, f"字节 token {i} 应该在词表中"
            assert len(vocab[i]) == 1, f"token {i} 应该只有 1 个字节"

    def test_empty_text(self):
        """空文本应该能正常处理。"""
        vocab, merges = train_bpe("", vocab_size=300)
        assert len(vocab) == 256  # 只有基础字节
        assert len(merges) == 0

    def test_repeated_merges(self):
        """高度重复的文本应产生合理的合并。"""
        text = "aaaa" * 200
        vocab, merges = train_bpe(text, vocab_size=260)
        # "aa" 合并成新的 token
        assert len(merges) > 0

    def test_with_special_tokens(self):
        """特殊 token 应该出现在词表中。"""
        text = "hello world " * 100
        vocab, merges = train_bpe(
            text, vocab_size=300,
            allowed_special={"<|endoftext|>", "<|pad|>"}
        )
        # 找到特殊 token 的 ID
        special_ids = set()
        for tid, tbytes in vocab.items():
            if tbytes.decode("utf-8", errors="replace") in ("<|endoftext|>", "<|pad|>"):
                special_ids.add(tid)
        assert len(special_ids) == 2


class TestTokenizerEncodeDecode:
    """测试 Tokenizer 的编解码功能。"""

    @pytest.fixture
    def sample_tokenizer(self):
        """创建一个在简单文本上训练的小 tokenizer。"""
        text = "the cat sat on the mat . the dog sat on the log . " * 200
        vocab, merges = train_bpe(text, vocab_size=300)
        return Tokenizer(vocab, merges)

    def test_roundtrip(self, sample_tokenizer):
        """编码后解码应该与原文本一致（无损）。"""
        text = "the cat sat on the mat"
        ids = sample_tokenizer.encode(text)
        decoded = sample_tokenizer.decode(ids)
        assert decoded == text

    def test_empty_encode(self, sample_tokenizer):
        """空字符串编码应为空列表。"""
        assert sample_tokenizer.encode("") == []

    def test_single_char(self, sample_tokenizer):
        """单个字符应该能正常编解码。"""
        text = "a"
        ids = sample_tokenizer.encode(text)
        decoded = sample_tokenizer.decode(ids)
        assert decoded == text

    def test_unknown_chars(self, sample_tokenizer):
        """任何 Unicode 字符都应该能处理（字节级 BPE 的好处）。"""
        text = "你好 world 🎉 test"
        ids = sample_tokenizer.encode(text)
        decoded = sample_tokenizer.decode(ids)
        assert decoded == text  # 字节级 BPE 保证无损

    def test_long_text(self, sample_tokenizer):
        """较长文本的编解码。"""
        text = "the cat sat on the mat . " * 50
        ids = sample_tokenizer.encode(text)
        decoded = sample_tokenizer.decode(ids)
        assert decoded == text

    def test_vocab_size_property(self, sample_tokenizer):
        """vocab_size 属性应正确反映词表大小。"""
        assert sample_tokenizer.vocab_size <= 300

    def test_ids_in_range(self, sample_tokenizer):
        """所有 token ID 都应该在有效范围内。"""
        text = "the quick brown fox jumps over the lazy dog"
        ids = sample_tokenizer.encode(text)
        for tid in ids:
            assert 0 <= tid < sample_tokenizer.vocab_size

    def test_special_token_encode(self):
        """带特殊 token 的编码测试。"""
        text = "hello world " * 100
        vocab, merges = train_bpe(
            text, vocab_size=300,
            allowed_special={"<|endoftext|>"}
        )
        tokenizer = Tokenizer(
            vocab, merges,
            special_tokens={"<|endoftext|>": 299}
        )
        ids = tokenizer.encode_special("hello <|endoftext|> world")
        assert 299 in ids

    def test_reproducibility(self, sample_tokenizer):
        """同一文本重复编码应得到相同结果。"""
        text = "the cat sat on the mat"
        ids1 = sample_tokenizer.encode(text)
        ids2 = sample_tokenizer.encode(text)
        assert ids1 == ids2
