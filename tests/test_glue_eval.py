import pytest
import torch
from eval.glue_eval import GLUEEvaluator


def test_cola_evaluator():
    ev = GLUEEvaluator("cola")
    logits = torch.tensor([[2.0, 0.1], [0.1, 2.0], [2.0, 0.1], [0.1, 2.0]])
    labels = torch.tensor([0, 1, 0, 1])
    ev.add(logits, labels)
    result = ev.compute()
    assert "mcc" in result


def test_sst2_evaluator():
    ev = GLUEEvaluator("sst-2")
    logits = torch.tensor([[2.0, 0.1], [0.1, 2.0]])
    labels = torch.tensor([0, 1])
    ev.add(logits, labels)
    result = ev.compute()
    assert "acc" in result
    assert result["acc"] == 1.0


def test_mrpc_evaluator():
    ev = GLUEEvaluator("mrpc")
    logits = torch.tensor([[2.0, 0.1], [0.1, 2.0]])
    labels = torch.tensor([0, 1])
    ev.add(logits, labels)
    result = ev.compute()
    assert "acc" in result
    assert "f1" in result


def test_stsb_evaluator():
    ev = GLUEEvaluator("sts-b")
    logits = torch.tensor([[0.5], [0.8]])
    labels = torch.tensor([0.5, 0.8])
    ev.add(logits, labels)
    result = ev.compute()
    assert "spearman" in result


def test_glue_integration():
    ev = GLUEEvaluator("mnli")
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 0])
    ev.add(logits, labels)
    result = ev.compute()
    assert "acc" in result
