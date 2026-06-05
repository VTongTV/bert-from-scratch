import torch
import torch.nn as nn
import torch.nn.functional as F


class NSPHead(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.classifier = nn.Linear(H, 2)

    def forward(self, pooled_output):
        return self.classifier(pooled_output)


def nsp_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def nsp_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean()
