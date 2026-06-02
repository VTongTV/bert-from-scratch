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
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        save_model(model, f.name)
        model2 = BertModel(config)
        load_model(model2, f.name)
        model2.eval()
        out2 = model2(input_ids)
        os.unlink(f.name)
    assert torch.allclose(out1[0], out2[0], atol=1e-5)


def test_config_serialization():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        save_config(config, f.name)
        loaded = load_config(f.name)
        os.unlink(f.name)
    assert loaded.L == config.L
    assert loaded.H == config.H
    assert loaded.A == config.A
