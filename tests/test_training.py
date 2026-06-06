import pytest
import torch
from config.bert_config import BertConfig
from model.bert import BertModel
from training.optimizer import create_adam_optimizer
from training.scheduler import WarmupLinearScheduler, warmup_lambda, linear_decay_lambda
from training.grad_accum import gradient_accumulation


def test_adam_optimizer_creation():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    optimizer = create_adam_optimizer(model, lr=1e-4)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_warmup_lambda():
    assert warmup_lambda(0, 1000) == 0.0
    assert abs(warmup_lambda(500, 1000) - 0.5) < 1e-6
    assert warmup_lambda(2000, 1000) == 1.0


def test_linear_decay_lambda():
    assert abs(linear_decay_lambda(0, 1000) - 1.0) < 1e-6
    assert abs(linear_decay_lambda(500, 1000) - 0.5) < 1e-6
    assert linear_decay_lambda(2000, 1000) == 0.0


def test_warmup_linear_scheduler():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    optimizer = create_adam_optimizer(model, lr=1e-4)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]
    scheduler = WarmupLinearScheduler(optimizer, warmup_steps=100, total_steps=1000)
    lrs = []
    for _ in range(5):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])
    assert lrs[0] < lrs[1]


def test_gradient_clipping():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    input_ids = torch.randint(0, 100, (2, 8))
    _, pooled = model(input_ids)
    loss = pooled.sum()
    loss.backward()
    grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert grad_norm_before > 0
