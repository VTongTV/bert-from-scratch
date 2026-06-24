import torch

from eval.metrics import accuracy


class SWAGEvaluator:
    def __init__(self):
        self.predictions = []
        self.labels = []

    def add(self, scores, labels):
        preds = scores.argmax(dim=-1)
        self.predictions.append(preds.cpu())
        self.labels.append(labels.cpu())

    def compute(self):
        preds = torch.cat(self.predictions)
        labels = torch.cat(self.labels)
        return {"acc": accuracy(preds, labels)}

    def reset(self):
        self.predictions = []
        self.labels = []
