import torch
import torch.nn as nn
import torch.nn.functional as F


class SWAGHead(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.classifier = nn.Linear(H, 1)

    def forward(self, pooled_outputs):
        scores = []
        for pooled in pooled_outputs:
            scores.append(self.classifier(pooled))
        return torch.cat(scores, dim=-1)


def swag_loss(scores, labels):
    return F.cross_entropy(scores, labels)
