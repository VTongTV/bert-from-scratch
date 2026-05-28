import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


def apply_attention_mask(scores, mask):
    return scores.masked_fill(mask == 0, -1e9)


class AttentionDropout(nn.Module):
    def __init__(self, P_drop=0.1):
        super().__init__()
        self.dropout = nn.Dropout(P_drop)

    def forward(self, attn_weights):
        return self.dropout(attn_weights)
