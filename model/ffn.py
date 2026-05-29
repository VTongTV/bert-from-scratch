import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class PositionWiseFFN(nn.Module):
    def __init__(self, H, d_ff, P_drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(H, d_ff)
        self.fc2 = nn.Linear(d_ff, H)
        self.gelu = GELU()
        self.dropout = nn.Dropout(P_drop)

    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class LayerNorm(nn.Module):
    def __init__(self, H, eps=1e-12):
        super().__init__()
        self.norm = nn.LayerNorm(H, eps=eps)

    def forward(self, x):
        return self.norm(x)


class ResidualConnection(nn.Module):
    def __init__(self, H, eps=1e-12):
        super().__init__()
        self.norm = LayerNorm(H, eps)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class DropoutWrapper(nn.Module):
    def __init__(self, P_drop=0.1):
        super().__init__()
        self.dropout = nn.Dropout(P_drop)

    def forward(self, x):
        return self.dropout(x)
