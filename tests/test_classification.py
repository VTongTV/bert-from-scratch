import pytest
import torch
from config.bert_config import BertConfig
from model.bert import BertModel
from model.classification import ClassificationHead, BinaryClassificationHead, RegressionHead, classification_loss


def test_classification_head():
    H, K = 64, 3
    head = ClassificationHead(H, K)
    x = torch.randn(2, H)
    out = head(x)
    assert out.shape == (2, K)


def test_binary_classification_head():
    H = 64
    head = BinaryClassificationHead(H)
    x = torch.randn(2, H)
    out = head(x)
    assert out.shape == (2, 2)


def test_regression_head():
    H = 64
    head = RegressionHead(H)
    x = torch.randn(2, H)
    out = head(x)
    assert out.shape == (2, 1)


def test_classification_loss():
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 0])
    loss = classification_loss(logits, labels)
    assert loss.item() > 0


def test_regression_loss():
    preds = torch.randn(4, 1)
    labels = torch.randn(4)
    loss = classification_loss(preds, labels, num_classes=1)
    assert loss.item() >= 0


def test_classification_integration():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    head = ClassificationHead(config.H, 3)
    input_ids = torch.randint(0, 100, (2, 8))
    _, pooled = model(input_ids)
    logits = head(pooled)
    assert logits.shape == (2, 3)
    labels = torch.tensor([0, 2])
    loss = classification_loss(logits, labels)
    assert loss.item() > 0
