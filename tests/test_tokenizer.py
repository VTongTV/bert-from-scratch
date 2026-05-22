import pytest
from data.tokens import CLS, SEP, MASK, PAD, UNK, SPECIAL_TOKENS
from data.vocabulary import Vocabulary
from data.tokenizer import WordPieceTokenizer


def make_test_vocab():
    tokens = ["hello", "world", "##ing", "test", "##s"]
    return Vocabulary(tokens)


def test_special_tokens():
    assert len(SPECIAL_TOKENS) == 5
    assert CLS == "[CLS]"
    assert SEP == "[SEP]"
    assert MASK == "[MASK]"
    assert PAD == "[PAD]"
    assert UNK == "[UNK]"


def test_vocab_lookup():
    vocab = make_test_vocab()
    assert vocab.get_id(PAD) == 0
    assert vocab.get_id(UNK) == 1
    assert "hello" in vocab
    assert vocab.get_id("missing") == vocab.get_id(UNK)


def test_vocab_len():
    vocab = make_test_vocab()
    assert len(vocab) == 10


def test_tokenize_basic():
    vocab = make_test_vocab()
    tok = WordPieceTokenizer(vocab)
    tokens = tok.tokenize("hello world")
    assert tokens == ["hello", "world"]


def test_tokenize_unknown():
    vocab = make_test_vocab()
    tok = WordPieceTokenizer(vocab)
    tokens = tok.tokenize("xyz")
    assert tokens == [UNK]


def test_encode_single():
    vocab = make_test_vocab()
    tok = WordPieceTokenizer(vocab)
    ids = tok.encode("hello world")
    assert ids[0] == vocab.get_id(CLS)
    assert ids[-1] == vocab.get_id(SEP)


def test_encode_pair():
    vocab = make_test_vocab()
    tok = WordPieceTokenizer(vocab)
    ids, segments = tok.encode_pair("hello", "world")
    assert ids[0] == vocab.get_id(CLS)
    assert 0 in segments
    assert 1 in segments


def test_decode():
    vocab = make_test_vocab()
    tok = WordPieceTokenizer(vocab)
    ids = tok.encode("hello world")
    text = tok.decode(ids)
    assert "hello" in text
    assert "world" in text
