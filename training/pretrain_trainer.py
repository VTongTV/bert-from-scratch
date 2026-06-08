import torch
from model.pretrain_model import BertForPreTraining
from training.optimizer import create_adam_optimizer
from training.scheduler import WarmupLinearScheduler
from training.checkpoint import save_checkpoint, load_checkpoint


class PreTrainer:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = BertForPreTraining(config).to(self.device)
        self.optimizer = create_adam_optimizer(self.model, lr=1e-4)
        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]
        self.scheduler = WarmupLinearScheduler(self.optimizer, warmup_steps=10000, total_steps=1000000)
        self.global_step = 0

    def train_step(self, batch):
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        segment_ids = batch.get("segment_ids", None)
        if segment_ids is not None:
            segment_ids = segment_ids.to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        mlm_labels = batch["mlm_labels"].to(self.device)
        nsp_labels = batch["nsp_labels"].to(self.device)
        loss = self.model(input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1
        return loss.item()

    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.device)
            segment_ids = batch.get("segment_ids", None)
            if segment_ids is not None:
                segment_ids = segment_ids.to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            mlm_labels = batch["mlm_labels"].to(self.device)
            nsp_labels = batch["nsp_labels"].to(self.device)
            loss = self.model(input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels)
        return loss.item()

    def save(self, path):
        save_checkpoint(self.model, self.optimizer, self.scheduler, self.global_step, path)

    def load(self, path):
        self.global_step = load_checkpoint(self.model, self.optimizer, self.scheduler, path)
