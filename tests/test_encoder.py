import pytest
import torch
from model.encoder import TransformerEncoderLayer


def test_encoder_layer_shape():
    H, A, d_ff = 64, 4, 256
    layer = TransformerEncoderLayer(H, A, d_ff)
    x = torch.randn(2, 8, H)
    out = layer(x)
    assert out.shape == (2, 8, H)


def test_encoder_layer_with_mask():
    H, A, d_ff = 64, 4, 256
    layer = TransformerEncoderLayer(H, A, d_ff)
    x = torch.randn(2, 8, H)
    mask = torch.ones(1, 1, 8, 8)
    out = layer(x, mask)
    assert out.shape == (2, 8, H)


def test_encoder_layer_gradient():
    H, A, d_ff = 64, 4, 256
    layer = TransformerEncoderLayer(H, A, d_ff)
    x = torch.randn(2, 8, H, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_encoder_layer_attention_extraction():
    H, A, d_ff = 64, 4, 256
    layer = TransformerEncoderLayer(H, A, d_ff)
    x = torch.randn(2, 8, H)
    attn_out = layer.self_attention_sublayer(x)
    assert attn_out.shape == (2, 8, H)


def test_encoder_layer_hidden_state():
    H, A, d_ff = 64, 4, 256
    layer = TransformerEncoderLayer(H, A, d_ff)
    x = torch.randn(2, 8, H)
    hidden = layer(x)
    assert hidden.shape == (2, 8, H)


def test_encoder_layer_config():
    H, A, d_ff = 64, 4, 256
    layer = TransformerEncoderLayer(H, A, d_ff)
    assert layer.H == H
    assert layer.A == A
    assert layer.d_ff == d_ff
