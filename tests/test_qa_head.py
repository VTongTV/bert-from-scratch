import pytest
import torch
from config.bert_config import BertConfig
from model.bert import BertModel
from model.qa_head import QAHead, qa_loss, best_span, best_span_batch


def test_qa_head():
    H = 64
    head = QAHead(H)
    x = torch.randn(2, 8, H)
    start, end = head(x)
    assert start.shape == (2, 8)
    assert end.shape == (2, 8)


def test_qa_loss():
    start_logits = torch.randn(2, 8)
    end_logits = torch.randn(2, 8)
    start_pos = torch.tensor([3, 5])
    end_pos = torch.tensor([4, 6])
    loss = qa_loss(start_logits, end_logits, start_pos, end_pos)
    assert loss.item() > 0


def test_best_span():
    start_logits = torch.tensor([0.1, 0.5, 0.3, 0.8])
    end_logits = torch.tensor([0.1, 0.2, 0.9, 0.4])
    s, e = best_span(start_logits, end_logits)
    assert s == 1
    assert e == 2


def test_qa_integration():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    head = QAHead(config.H)
    input_ids = torch.randint(0, 100, (2, 8))
    encoder_out, _ = model(input_ids)
    start, end = head(encoder_out)
    assert start.shape == (2, 8)
    loss = qa_loss(start, end, torch.tensor([3, 5]), torch.tensor([4, 6]))
    assert loss.item() > 0
