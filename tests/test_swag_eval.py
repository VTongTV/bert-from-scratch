import pytest
import torch
from eval.swag_eval import SWAGEvaluator


def test_swag_evaluator():
    ev = SWAGEvaluator()
    scores = torch.tensor([[2.0, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 2.0]])
    labels = torch.tensor([0, 3])
    ev.add(scores, labels)
    result = ev.compute()
    assert "acc" in result
    assert result["acc"] == 1.0


def test_swag_evaluator_partial():
    ev = SWAGEvaluator()
    scores = torch.tensor([[2.0, 0.1, 0.1, 0.1], [0.1, 2.0, 0.1, 0.1]])
    labels = torch.tensor([0, 3])
    ev.add(scores, labels)
    result = ev.compute()
    assert result["acc"] == 0.5
