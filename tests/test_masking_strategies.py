import pytest
import torch
from data.masking_strategies import MaskingStrategy, STRATEGIES


def test_default_strategy():
    s = STRATEGIES["default"]
    assert s.mask_prob == 0.15
    assert abs(s.replace_mask - 0.8) < 1e-6
    assert abs(s.replace_random - 0.1) < 1e-6


def test_masking_applied():
    s = MaskingStrategy(0.15, 0.8, 0.1, 0.1)
    input_ids = torch.randint(0, 1000, (4, 32))
    masked, labels, mask = s.apply(input_ids, 1000)
    assert masked.shape == input_ids.shape
    assert labels.shape == input_ids.shape
    assert (labels[~mask.bool()] == -100).all()


def test_all_mask_strategy():
    s = STRATEGIES["all_mask"]
    input_ids = torch.randint(0, 100, (4, 32))
    masked, labels, mask = s.apply(input_ids, 100)
    assert (masked[labels != -100] == 103).all()


def test_all_strategies_exist():
    for name in ["default", "all_mask", "no_random", "no_mask", "all_random", "all_keep"]:
        assert name in STRATEGIES
