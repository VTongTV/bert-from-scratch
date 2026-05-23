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


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, H):
        super().__init__()
        self.embedding = nn.Embedding(max_len, H)

    def forward(self, position_ids):
        return self.embedding(position_ids)


class BertEmbeddings(nn.Module):
    def __init__(self, V, H, max_len):
        super().__init__()
        self.token = TokenEmbedding(V, H)
        self.segment = SegmentEmbedding(H)
        self.position = PositionEmbedding(max_len, H)
        self.layer_norm = nn.LayerNorm(H, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids=None):
        seq_len = input_ids.size(1)
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token(input_ids) + self.segment(segment_ids) + self.position(position_ids)
        return self.dropout(self.layer_norm(x))
