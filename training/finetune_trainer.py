import torch
from model.bert import BertModel
from model.classification import ClassificationHead, RegressionHead, classification_loss
from training.optimizer import create_adam_optimizer
from training.scheduler import WarmupLinearScheduler


class FinetuneTrainer:
    def __init__(self, config, num_classes, device="cpu", lr=2e-5, epochs=3):
        self.device = torch.device(device)
        self.epochs = epochs
        self.lr = lr
        self.bert = BertModel(config).to(self.device)
        if num_classes == 1:
            self.head = RegressionHead(config.H).to(self.device)
        else:
            self.head = ClassificationHead(config.H, num_classes).to(self.device)
        self.num_classes = num_classes
        self.optimizer = create_adam_optimizer(torch.nn.ModuleList([self.bert, self.head]), lr=lr, weight_decay=0.01)
        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]
        self.scheduler = None
        self.best_loss = float("inf")

    def train_step(self, batch):
        self.bert.train()
        self.head.train()
        input_ids = batch["input_ids"].to(self.device)
        segment_ids = batch.get("segment_ids")
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"].to(self.device)
        if segment_ids is not None:
            segment_ids = segment_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        _, pooled = self.bert(input_ids, segment_ids, attention_mask)
        logits = self.head(pooled)
        loss = classification_loss(logits, labels, self.num_classes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.bert.parameters()) + list(self.head.parameters()), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def eval_step(self, batch):
        self.bert.eval()
        self.head.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.device)
            segment_ids = batch.get("segment_ids")
            attention_mask = batch.get("attention_mask")
            labels = batch["labels"].to(self.device)
            if segment_ids is not None:
                segment_ids = segment_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            _, pooled = self.bert(input_ids, segment_ids, attention_mask)
            logits = self.head(pooled)
            loss = classification_loss(logits, labels, self.num_classes)
        return loss.item(), logits

    def train_epoch(self, dataloader):
        total_loss = 0
        for batch in dataloader:
            total_loss += self.train_step(batch)
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        total_loss = 0
        all_logits = []
        for batch in dataloader:
            loss, logits = self.eval_step(batch)
            total_loss += loss
            all_logits.append(logits)
        return total_loss / len(dataloader), torch.cat(all_logits)
