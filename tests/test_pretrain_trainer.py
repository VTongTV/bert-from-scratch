import pytest
import torch
from config.bert_config import BertConfig
from training.pretrain_trainer import PreTrainer


def test_pretrain_trainer_creation():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    trainer = PreTrainer(config)
    assert trainer.global_step == 0


def test_pretrain_train_step():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    trainer = PreTrainer(config)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "mlm_labels": torch.full((2, 8), -100),
        "nsp_labels": torch.tensor([0, 1]),
    }
    batch["mlm_labels"][0, 3] = 5
    batch["mlm_labels"][1, 4] = 10
    loss = trainer.train_step(batch)
    assert loss > 0
    assert trainer.global_step == 1


def test_pretrain_eval_step():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    trainer = PreTrainer(config)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "mlm_labels": torch.full((2, 8), -100),
        "nsp_labels": torch.tensor([0, 1]),
    }
    batch["mlm_labels"][0, 3] = 5
    loss = trainer.eval_step(batch)
    assert loss > 0


def test_pretrain_multiple_steps():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    trainer = PreTrainer(config)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "mlm_labels": torch.full((2, 8), -100),
        "nsp_labels": torch.tensor([0, 1]),
    }
    batch["mlm_labels"][0, 3] = 5
    losses = []
    for _ in range(3):
        losses.append(trainer.train_step(batch))
    assert trainer.global_step == 3
    assert all(l > 0 for l in losses)
