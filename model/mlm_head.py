import torch
import torch.nn as nn

from model.ffn import GELU, LayerNorm


class MLMHead(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.dense = nn.Linear(H, H)
        self.gelu = GELU()
        self.layer_norm = LayerNorm(H)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.gelu(x)
        x = self.layer_norm(x)
        return x
