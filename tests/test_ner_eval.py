import pytest
from eval.ner_eval import entity_f1


def test_entity_f1_perfect():
    preds = [0, 1, 1, 0, 2, 2]
    labels = [0, 1, 1, 0, 2, 2]
    f1 = entity_f1(preds, labels)
    assert abs(f1 - 1.0) < 1e-6


def test_entity_f1_partial():
    preds = [0, 1, 1, 0, 0, 0]
    labels = [0, 1, 1, 0, 2, 2]
    f1 = entity_f1(preds, labels)
    assert 0.0 <= f1 <= 1.0


def test_entity_f1_no_match():
    preds = [1, 1, 0, 0, 0, 0]
    labels = [0, 0, 0, 2, 2, 2]
    f1 = entity_f1(preds, labels)
    assert f1 == 0.0
