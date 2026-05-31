import pytest
import torch
from config.bert_config import BertConfig, bert_base_config
from model.bert import BertModel, count_parameters, init_bert_weights


def test_bert_model_forward():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    input_ids = torch.randint(0, 100, (2, 16))
    encoder_out, pooled_out = model(input_ids)
    assert encoder_out.shape == (2, 16, 64)
    assert pooled_out.shape == (2, 64)


def test_bert_model_with_segment_ids():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    input_ids = torch.randint(0, 100, (2, 16))
    segment_ids = torch.randint(0, 2, (2, 16))
    encoder_out, pooled_out = model(input_ids, segment_ids)
    assert encoder_out.shape == (2, 16, 64)


def test_bert_model_with_attention_mask():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones(2, 16)
    encoder_out, pooled_out = model(input_ids, attention_mask=attention_mask)
    assert encoder_out.shape == (2, 16, 64)


def test_bert_hidden_states():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    input_ids = torch.randint(0, 100, (2, 16))
    hidden = model.get_hidden_states(input_ids)
    assert len(hidden) == 3
    assert hidden[0].shape == (2, 16, 64)


def test_count_parameters():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    n = count_parameters(model)
    assert n > 0


def test_bert_integration():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    model.apply(init_bert_weights)
    input_ids = torch.randint(0, 100, (2, 16))
    segment_ids = torch.randint(0, 2, (2, 16))
    attention_mask = torch.ones(2, 16)
    encoder_out, pooled_out = model(input_ids, segment_ids, attention_mask)
    assert encoder_out.shape == (2, 16, 64)
    assert pooled_out.shape == (2, 64)
    assert not torch.isnan(encoder_out).any()
    assert not torch.isnan(pooled_out).any()
    loss = pooled_out.sum()
    loss.backward()
