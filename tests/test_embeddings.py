import pytest
import torch
from model.embeddings import TokenEmbedding, SegmentEmbedding, PositionEmbedding, BertEmbeddings


def test_token_embedding_shape():
    V, H = 100, 64
    emb = TokenEmbedding(V, H)
    ids = torch.randint(0, V, (2, 10))
    out = emb(ids)
    assert out.shape == (2, 10, H)


def test_segment_embedding_shape():
    H = 64
    emb = SegmentEmbedding(H)
    ids = torch.randint(0, 2, (2, 10))
    out = emb(ids)
    assert out.shape == (2, 10, H)


def test_position_embedding_shape():
    max_len, H = 512, 64
    emb = PositionEmbedding(max_len, H)
    ids = torch.arange(10).unsqueeze(0)
    out = emb(ids)
    assert out.shape == (1, 10, H)


def test_bert_embeddings_shape():
    V, H, max_len = 100, 64, 512
    emb = BertEmbeddings(V, H, max_len)
    input_ids = torch.randint(0, V, (2, 10))
    out = emb(input_ids)
    assert out.shape == (2, 10, H)


def test_bert_embeddings_with_segment_ids():
    V, H, max_len = 100, 64, 512
    emb = BertEmbeddings(V, H, max_len)
    input_ids = torch.randint(0, V, (2, 10))
    segment_ids = torch.randint(0, 2, (2, 10))
    out = emb(input_ids, segment_ids)
    assert out.shape == (2, 10, H)


def test_bert_embeddings_no_segment():
    V, H, max_len = 100, 64, 512
    emb = BertEmbeddings(V, H, max_len)
    input_ids = torch.randint(0, V, (2, 10))
    out = emb(input_ids)
    assert out.shape == (2, 10, H)
