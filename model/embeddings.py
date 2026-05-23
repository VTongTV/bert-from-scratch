import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.embedding = nn.Embedding(V, H)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class SegmentEmbedding(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.embedding = nn.Embedding(2, H)

    def forward(self, segment_ids):
        return self.embedding(segment_ids)
