import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.ffn import PositionWiseFFN, LayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(self, H, A, d_ff, P_drop=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(H, A, P_drop)
        self.attn_norm = LayerNorm(H)
        self.ffn = PositionWiseFFN(H, d_ff, P_drop)
        self.ffn_norm = LayerNorm(H)
        self.attn_dropout = nn.Dropout(P_drop)
        self.ffn_dropout = nn.Dropout(P_drop)

    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.attn_norm(x + self.attn_dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.ffn_dropout(ffn_out))
        return x
