import pytest
import torch
from model.ffn import GELU, PositionWiseFFN, LayerNorm, ResidualConnection, DropoutWrapper


def test_gelu_output():
    gelu = GELU()
    x = torch.randn(2, 4, 8)
    out = gelu(x)
    assert out.shape == x.shape


def test_ffn_shape():
    H, d_ff = 64, 256
    ffn = PositionWiseFFN(H, d_ff)
    x = torch.randn(2, 8, H)
    out = ffn(x)
    assert out.shape == (2, 8, H)


def test_layer_norm_shape():
    H = 64
    ln = LayerNorm(H)
    x = torch.randn(2, 8, H)
    out = ln(x)
    assert out.shape == x.shape


def test_residual_connection():
    H = 64
    res = ResidualConnection(H)
    x = torch.randn(2, 8, H)
    out = res(x, lambda x: x)
    assert out.shape == x.shape


def test_dropout_wrapper():
    dw = DropoutWrapper(0.5)
    x = torch.ones(10, 10)
    dw.train()
    out = dw(x)
    assert out.shape == x.shape


def test_layer_norm_stability():
    H = 64
    ln = LayerNorm(H)
    x = torch.randn(2, 8, H) * 100
    out = ln(x)
    assert not torch.isnan(out).any()
    assert out.std() < x.std()


def test_gelu_gradient():
    gelu = GELU()
    x = torch.randn(4, requires_grad=True)
    out = gelu(x)
    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
