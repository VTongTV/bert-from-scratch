import pytest
import torch
from config.bert_config import BertConfig
from model.pretrain_model import BertForPreTraining


def test_pretrain_model_forward_no_labels():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertForPreTraining(config)
    input_ids = torch.randint(0, 100, (2, 8))
    mlm_logits, nsp_logits = model(input_ids)
    assert mlm_logits.shape == (2, 8, config.V)
    assert nsp_logits.shape == (2, 2)


def test_pretrain_model_forward_with_labels():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertForPreTraining(config)
    input_ids = torch.randint(0, 100, (2, 8))
    mlm_labels = torch.full((2, 8), -100)
    mlm_labels[0, 3] = 5
    mlm_labels[1, 4] = 10
    nsp_labels = torch.tensor([0, 1])
    loss = model(input_ids, mlm_labels=mlm_labels, nsp_labels=nsp_labels)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_pretrain_model_backward():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertForPreTraining(config)
    input_ids = torch.randint(0, 100, (2, 8))
    mlm_labels = torch.full((2, 8), -100)
    mlm_labels[0, 3] = 5
    nsp_labels = torch.tensor([1, 0])
    loss = model(input_ids, mlm_labels=mlm_labels, nsp_labels=nsp_labels)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_pretrain_weight_tying():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertForPreTraining(config)
    assert model.mlm.decoder.decoder.weight is model.bert.embeddings.token.embedding.weight
