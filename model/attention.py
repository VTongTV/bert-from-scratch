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


def compute_attention_scores(Q, K):
    d_k = Q.size(-1)
    return torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)


class AttentionOutputProjection(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.linear = nn.Linear(H, H)

    def forward(self, x):
        return self.linear(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, H, A, P_drop=0.1):
        super().__init__()
        self.A = A
        self.d_k = H // A
        self.W_Q = nn.Linear(H, H)
        self.W_K = nn.Linear(H, H)
        self.W_V = nn.Linear(H, H)
        self.W_O = nn.Linear(H, H)
        self.attn_dropout = nn.Dropout(P_drop)
        self.out_dropout = nn.Dropout(P_drop)

    def forward(self, x, mask=None):
        B, S, H = x.size()
        Q = self.W_Q(x).view(B, S, self.A, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.A, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.A, self.d_k).transpose(1, 2)
        scores = compute_attention_scores(Q, K)
        if mask is not None:
            scores = apply_attention_mask(scores, mask)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, S, H)
        return self.out_dropout(self.W_O(context))
