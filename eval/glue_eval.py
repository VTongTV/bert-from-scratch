import torch

from eval.metrics import accuracy, f1_score, spearman_corr, matthews_corr


class GLUEEvaluator:
    def __init__(self, task_name):
        self.task_name = task_name
        self.predictions = []
        self.labels = []

    def add(self, logits, labels):
        if self.task_name == "sts-b":
            preds = logits.squeeze(-1)
        else:
            preds = logits.argmax(dim=-1)
        self.predictions.append(preds.cpu())
        self.labels.append(labels.cpu())

    def compute(self):
        preds = torch.cat(self.predictions)
        labels = torch.cat(self.labels)
        if self.task_name == "cola":
            return {"mcc": matthews_corr(preds, labels)}
        elif self.task_name == "sst-2":
            return {"acc": accuracy(preds, labels)}
        elif self.task_name == "mrpc":
            return {"acc": accuracy(preds, labels), "f1": f1_score(preds, labels)}
        elif self.task_name == "sts-b":
            return {"spearman": spearman_corr(preds.float(), labels.float())}
        elif self.task_name == "qqp":
            return {"acc": accuracy(preds, labels), "f1": f1_score(preds, labels)}
        else:
            return {"acc": accuracy(preds, labels)}

    def reset(self):
        self.predictions = []
        self.labels = []
