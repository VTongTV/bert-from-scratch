import pytest
from config.ablation_configs import small_config, medium_config_3, medium_config_12, large_h_config


def test_small_config():
    c = small_config()
    assert c.L == 3 and c.H == 768 and c.A == 12


def test_medium_config_3():
    c = medium_config_3()
    assert c.L == 6 and c.H == 768 and c.A == 3


def test_medium_config_12():
    c = medium_config_12()
    assert c.L == 6 and c.H == 768 and c.A == 12


def test_large_h_config():
    c = large_h_config()
    assert c.L == 12 and c.H == 1024 and c.A == 16
