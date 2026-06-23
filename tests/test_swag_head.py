import pytest
import torch
from model.swag_head import SWAGHead, swag_loss


def test_swag_head():
    H = 64
    head = SWAGHead(H)
    pooled_outputs = [torch.randn(2, H) for _ in range(4)]
    scores = head(pooled_outputs)
    assert scores.shape == (2, 4)


def test_swag_loss():
    scores = torch.randn(2, 4)
    labels = torch.tensor([0, 3])
    loss = swag_loss(scores, labels)
    assert loss.item() > 0


def test_swag_integration():
    H = 64
    head = SWAGHead(H)
    pooled_outputs = [torch.randn(2, H) for _ in range(4)]
    scores = head(pooled_outputs)
    labels = torch.tensor([1, 2])
    loss = swag_loss(scores, labels)
    loss.backward()
    for p in head.parameters():
        if p.requires_grad:
            assert p.grad is not None
