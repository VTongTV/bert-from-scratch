import pytest
import torch
from eval.metrics import accuracy, f1_score, precision, recall, spearman_corr, matthews_corr


def test_accuracy():
    preds = torch.tensor([0, 1, 1, 0])
    labels = torch.tensor([0, 1, 0, 1])
    assert accuracy(preds, labels) == 0.5


def test_f1_score():
    preds = torch.tensor([1, 1, 0, 0])
    labels = torch.tensor([1, 0, 1, 0])
    f1 = f1_score(preds, labels)
    assert 0.0 <= f1 <= 1.0


def test_precision():
    preds = torch.tensor([1, 1, 0, 0])
    labels = torch.tensor([1, 0, 1, 0])
    p = precision(preds, labels)
    assert 0.0 <= p <= 1.0


def test_recall():
    preds = torch.tensor([1, 1, 0, 0])
    labels = torch.tensor([1, 0, 1, 0])
    r = recall(preds, labels)
    assert 0.0 <= r <= 1.0


def test_spearman():
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = torch.tensor([1.0, 2.0, 3.0, 4.0])
    s = spearman_corr(preds, labels)
    assert abs(s - 1.0) < 0.01


def test_matthews():
    preds = torch.tensor([1, 1, 0, 0])
    labels = torch.tensor([1, 0, 1, 0])
    m = matthews_corr(preds, labels)
    assert -1.0 <= m <= 1.0
