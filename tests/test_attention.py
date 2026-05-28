import pytest
import torch
from model.attention import (
    scaled_dot_product_attention,
    compute_attention_scores,
    apply_attention_mask,
    MultiHeadAttention,
    split_heads,
    merge_heads,
    create_causal_mask,
)


def test_scaled_dot_product_attention_shape():
    B, A, S, d_k = 2, 4, 8, 16
    Q = torch.randn(B, A, S, d_k)
    K = torch.randn(B, A, S, d_k)
    V = torch.randn(B, A, S, d_k)
    out = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (B, A, S, d_k)


def test_compute_attention_scores_shape():
    B, A, S, d_k = 2, 4, 8, 16
    Q = torch.randn(B, A, S, d_k)
    K = torch.randn(B, A, S, d_k)
    scores = compute_attention_scores(Q, K)
    assert scores.shape == (B, A, S, S)


def test_multi_head_attention_shape():
    H, A = 64, 4
    mha = MultiHeadAttention(H, A)
    x = torch.randn(2, 8, H)
    out = mha(x)
    assert out.shape == (2, 8, H)


def test_split_merge_heads():
    H, A = 64, 4
    x = torch.randn(2, 8, H)
    split = split_heads(x, A)
    assert split.shape == (2, A, 8, H // A)
    merged = merge_heads(split)
    assert merged.shape == x.shape


def test_causal_mask():
    mask = create_causal_mask(8)
    assert mask.shape == (1, 1, 8, 8)
    assert mask[0, 0, 0, 0] == 1
    assert mask[0, 0, 0, 1] == 0
    assert mask[0, 0, 1, 0] == 1
