import pytest
import torch
from config.bert_config import BertConfig
from model.bert import BertModel
from model.nsp_head import NSPHead, nsp_loss


def test_nsp_head():
    H = 64
    head = NSPHead(H)
    x = torch.randn(2, H)
    out = head(x)
    assert out.shape == (2, 2)


def test_nsp_loss():
    logits = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 1, 0])
    loss = nsp_loss(logits, labels)
    assert loss.item() > 0
    assert loss.ndim == 0


def test_nsp_prediction():
    H = 64
    head = NSPHead(H)
    x = torch.randn(2, H)
    out = head(x)
    preds = out.argmax(dim=-1)
    assert preds.shape == (2,)
    assert (preds >= 0).all() and (preds <= 1).all()


def test_nsp_accuracy():
    logits = torch.tensor([[2.0, 0.1], [0.1, 2.0], [2.0, 0.1], [0.1, 2.0]])
    labels = torch.tensor([0, 1, 0, 1])
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean()
    assert acc.item() == 1.0


def test_nsp_integration_with_bert():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    nsp = NSPHead(config.H)
    input_ids = torch.randint(0, 100, (2, 8))
    _, pooled = model(input_ids)
    logits = nsp(pooled)
    assert logits.shape == (2, 2)
    labels = torch.tensor([0, 1])
    loss = nsp_loss(logits, labels)
    assert loss.item() > 0
