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
        self.decoder = nn.Linear(H, V, bias=True)

    def forward(self, hidden_states):
        return self.decoder(hidden_states)

    def tie_weights(self, word_embeddings):
        self.decoder.weight = word_embeddings.weight


class MLMPrediction(nn.Module):
    def __init__(self, H, V):
        super().__init__()
        self.head = MLMHead(H)
        self.decoder = MLMDecoder(H, V)

    def forward(self, hidden_states):
        x = self.head(hidden_states)
        x = self.decoder(x)
        return x

    def tie_weights(self, word_embeddings):
        self.decoder.tie_weights(word_embeddings)


def mlm_loss(logits, labels, mask):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
    loss = loss.view(labels.size())
    return (loss * mask).sum() / mask.sum()
