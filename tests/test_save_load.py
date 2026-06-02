import pytest
import torch
import tempfile
import os
from config.bert_config import BertConfig
from model.bert import BertModel, count_parameters
from model.save_load import save_model, load_model
from model.config_io import save_config, load_config


def test_parameter_counting():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    n = count_parameters(model)
    assert n > 0


def test_model_save_load():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    model.eval()
    input_ids = torch.randint(0, 100, (2, 8))
    out1 = model(input_ids)
    path = os.path.join(tempfile.gettempdir(), "test_bert.pt")
    try:
        save_model(model, path)
        model2 = BertModel(config)
        load_model(model2, path)
        model2.eval()
        out2 = model2(input_ids)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    assert torch.allclose(out1[0], out2[0], atol=1e-5)


def test_config_serialization():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    path = os.path.join(tempfile.gettempdir(), "test_config.json")
    try:
        save_config(config, path)
        loaded = load_config(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    assert loaded.L == config.L
    assert loaded.H == config.H
    assert loaded.A == config.A
