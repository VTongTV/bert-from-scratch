import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MLMDecoder(nn.Module):
    def __init__(self, H, V):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(V))

    def forward(self, hidden_states, word_embeddings):
        x = F.linear(hidden_states, word_embeddings.weight, self.bias)
        return x
