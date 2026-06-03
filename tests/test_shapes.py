import pytest
import torch
from config.bert_config import BertConfig, bert_base_config, bert_large_config
from model.bert import BertModel, count_parameters


def test_bert_base_config():
    config = bert_base_config()
    assert config.L == 12
    assert config.H == 768
    assert config.A == 12
    assert config.d_ff == 3072


def test_bert_large_config():
    config = bert_large_config()
    assert config.L == 24
    assert config.H == 1024
    assert config.A == 16
    assert config.d_ff == 4096


def test_attention_shape():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    input_ids = torch.randint(0, 100, (2, 16))
    encoder_out, _ = model(input_ids)
    assert encoder_out.shape == (2, 16, 64)


def test_embedding_shape():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    input_ids = torch.randint(0, 100, (2, 16))
    emb_out = model.embeddings(input_ids)
    assert emb_out.shape == (2, 16, 64)


def test_encoder_output_shape():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    input_ids = torch.randint(0, 100, (2, 16))
    encoder_out, _ = model(input_ids)
    assert encoder_out.shape == (2, 16, 64)


def test_pooler_output_shape():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    input_ids = torch.randint(0, 100, (2, 16))
    _, pooled_out = model(input_ids)
    assert pooled_out.shape == (2, 64)


def test_batch_processing():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    input_ids = torch.randint(0, 100, (4, 16))
    encoder_out, pooled_out = model(input_ids)
    assert encoder_out.shape == (4, 16, 64)
    assert pooled_out.shape == (4, 64)


def test_variable_sequence_length():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    for seq_len in [8, 16, 32]:
        input_ids = torch.randint(0, 100, (2, seq_len))
        encoder_out, pooled_out = model(input_ids)
        assert encoder_out.shape == (2, seq_len, 64)


def test_attention_mask_correctness():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    model.eval()
    input_ids = torch.randint(0, 100, (2, 16))
    mask = torch.ones(2, 16)
    mask[0, 8:] = 0
    out_masked = model(input_ids, attention_mask=mask)
    out_full = model(input_ids, attention_mask=torch.ones(2, 16))
    assert not torch.equal(out_masked[0], out_full[0])


def test_gradient_backpropagation():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    input_ids = torch.randint(0, 100, (2, 16))
    encoder_out, pooled_out = model(input_ids)
    loss = pooled_out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
