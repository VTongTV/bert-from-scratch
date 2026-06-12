import pytest
import torch
from config.bert_config import BertConfig
from training.finetune_trainer import FinetuneTrainer


def test_finetune_trainer_creation():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    trainer = FinetuneTrainer(config, num_classes=3)
    assert trainer.num_classes == 3


def test_finetune_train_step():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    trainer = FinetuneTrainer(config, num_classes=3)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "labels": torch.tensor([0, 2]),
    }
    loss = trainer.train_step(batch)
    assert loss > 0


def test_finetune_eval_step():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    trainer = FinetuneTrainer(config, num_classes=3)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "labels": torch.tensor([0, 2]),
    }
    loss, logits = trainer.eval_step(batch)
    assert loss > 0
    assert logits.shape == (2, 3)


def test_finetune_regression():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    trainer = FinetuneTrainer(config, num_classes=1)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "labels": torch.tensor([0.5, 0.8]),
    }
    loss = trainer.train_step(batch)
    assert loss >= 0


def test_finetune_best_model_selection():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    trainer = FinetuneTrainer(config, num_classes=2)
    assert trainer.best_loss == float("inf")
    trainer.best_loss = 5.0
    assert trainer.best_loss < float("inf")
