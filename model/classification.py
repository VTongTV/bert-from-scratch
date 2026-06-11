import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, H, K, P_drop=0.1):
        super().__init__()
        self.dense = nn.Linear(H, H)
        self.dropout = nn.Dropout(P_drop)
        self.out_proj = nn.Linear(H, K)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BinaryClassificationHead(nn.Module):
    def __init__(self, H, P_drop=0.1):
        super().__init__()
        self.classifier = ClassificationHead(H, 2, P_drop)

    def forward(self, x):
        return self.classifier(x)


class RegressionHead(nn.Module):
    def __init__(self, H, P_drop=0.1):
        super().__init__()
        self.dense = nn.Linear(H, H)
        self.dropout = nn.Dropout(P_drop)
        self.out_proj = nn.Linear(H, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def classification_loss(logits, labels, num_classes=None):
    if num_classes is not None and num_classes == 1:
        return F.mse_loss(logits.squeeze(-1), labels.float())
    return F.cross_entropy(logits, labels)
