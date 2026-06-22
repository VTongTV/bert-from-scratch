import pytest
from eval.squad_eval import normalize_answer, exact_match, f1_score_qa, squad_evaluate


def test_normalize():
    assert normalize_answer("Hello, World!") == "hello world"
    assert normalize_answer("  foo   bar  ") == "foo bar"


def test_exact_match():
    assert exact_match("the cat", "The Cat") == 1
    assert exact_match("the cat", "the dog") == 0


def test_f1_score():
    f1 = f1_score_qa("the cat sat", "the cat sat on the mat")
    assert 0.0 <= f1 <= 1.0
    assert f1 > 0


def test_squad_evaluate():
    preds = ["the cat", "a dog"]
    golds = ["the cat", "the dog"]
    result = squad_evaluate(preds, golds)
    assert "em" in result
    assert "f1" in result
