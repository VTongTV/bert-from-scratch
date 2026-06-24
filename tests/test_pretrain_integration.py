import pytest
import torch
from config.bert_config import BertConfig
from training.pretrain_trainer import PreTrainer


def test_overfit_small_batch():
    config = BertConfig(L=1, H=32, A=2, V=50, max_len=16)
    trainer = PreTrainer(config)
    batch = {
        "input_ids": torch.randint(0, 50, (4, 16)),
        "segment_ids": torch.zeros(4, 16, dtype=torch.long),
        "mlm_labels": torch.full((4, 16), -100, dtype=torch.long),
        "nsp_labels": torch.tensor([0, 1, 0, 1]),
    }
    batch["mlm_labels"][:, 3] = torch.randint(0, 50, (4,))
    losses = []
    for _ in range(100):
        loss = trainer.train_step(batch)
        losses.append(loss)
    assert losses[-1] < losses[0]


def test_mlm_loss_convergence():
    config = BertConfig(L=1, H=32, A=2, V=50, max_len=16)
    trainer = PreTrainer(config)
    batch = {
        "input_ids": torch.randint(0, 50, (4, 16)),
        "segment_ids": torch.zeros(4, 16, dtype=torch.long),
        "mlm_labels": torch.full((4, 16), -100, dtype=torch.long),
        "nsp_labels": torch.tensor([0, 1, 0, 1]),
    }
    batch["mlm_labels"][:, 3] = torch.randint(0, 50, (4,))
    losses = []
    for _ in range(100):
        loss = trainer.train_step(batch)
        losses.append(loss)
    assert losses[-1] < losses[0]


def test_nsp_accuracy_convergence():
    config = BertConfig(L=1, H=32, A=2, V=50, max_len=16)
    from model.nsp_head import nsp_accuracy
    trainer = PreTrainer(config)
    batch = {
        "input_ids": torch.randint(0, 50, (4, 16)),
        "segment_ids": torch.zeros(4, 16, dtype=torch.long),
        "mlm_labels": torch.full((4, 16), -100, dtype=torch.long),
        "nsp_labels": torch.tensor([0, 1, 0, 1]),
    }
    batch["mlm_labels"][:, 3] = torch.randint(0, 50, (4,))
    for _ in range(10):
        trainer.train_step(batch)
    trainer.model.eval()
    with torch.no_grad():
        mlm_logits, nsp_logits = trainer.model(batch["input_ids"], batch["segment_ids"])
    acc = nsp_accuracy(nsp_logits, batch["nsp_labels"])
    assert acc.item() >= 0.25


def test_gradient_flow():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    trainer = PreTrainer(config)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "mlm_labels": torch.full((2, 8), -100),
        "nsp_labels": torch.tensor([0, 1]),
    }
    batch["mlm_labels"][0, 3] = 5
    trainer.model.train()
    trainer.model.zero_grad()
    loss = trainer.model(batch["input_ids"], mlm_labels=batch["mlm_labels"], nsp_labels=batch["nsp_labels"])
    loss.backward()
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_smoke_test():
    config = BertConfig(L=1, H=32, A=2, V=50, max_len=16)
    trainer = PreTrainer(config)
    batch = {
        "input_ids": torch.randint(0, 50, (2, 16)),
        "mlm_labels": torch.full((2, 16), -100),
        "nsp_labels": torch.tensor([0, 1]),
    }
    batch["mlm_labels"][:, 3] = torch.randint(0, 50, (2,))
    loss = trainer.train_step(batch)
    assert loss > 0
    assert trainer.global_step == 1


def test_mini_run():
    config = BertConfig(L=1, H=32, A=2, V=50, max_len=16)
    trainer = PreTrainer(config)
    batch = {
        "input_ids": torch.randint(0, 50, (2, 16)),
        "mlm_labels": torch.full((2, 16), -100),
        "nsp_labels": torch.tensor([0, 1]),
    }
    batch["mlm_labels"][:, 3] = torch.randint(0, 50, (2,))
    for _ in range(10):
        loss = trainer.train_step(batch)
    assert trainer.global_step == 10


def test_config_validation():
    config = BertConfig(L=12, H=768, A=12, V=30000, max_len=512)
    assert config.L == 12
    assert config.H == 768
    assert config.A == 12
    assert config.d_ff == 3072
