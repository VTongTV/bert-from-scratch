import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.ffn import PositionWiseFFN, LayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(self, H, A, d_ff, P_drop=0.1):
        super().__init__()
        self.H = H
        self.A = A
        self.d_ff = d_ff
        self.attention = MultiHeadAttention(H, A, P_drop)
        self.attn_norm = LayerNorm(H)
        self.ffn = PositionWiseFFN(H, d_ff, P_drop)
        self.ffn_norm = LayerNorm(H)
        self.attn_dropout = nn.Dropout(P_drop)
        self.ffn_dropout = nn.Dropout(P_drop)

    def forward(self, x, mask=None):
        # section 3.1
        x = self.self_attention_sublayer(x, mask)
        x = self.feed_forward_sublayer(x)
        return x

    def self_attention_sublayer(self, x, mask=None):
        attn_out = self.attention(x, mask)
        return self.attn_norm(x + self.attn_dropout(attn_out))

    def feed_forward_sublayer(self, x):
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + self.ffn_dropout(ffn_out))


def post_layer_norm(x, sublayer_out, norm, dropout):
    return norm(x + dropout(sublayer_out))


class BertEncoder(nn.Module):
    def __init__(self, L, H, A, d_ff, P_drop=0.1):
        super().__init__()
        self.layer = nn.ModuleList([
            TransformerEncoderLayer(H, A, d_ff, P_drop) for _ in range(L)
        ])

    def forward(self, x, mask=None):
        for layer in self.layer:
            x = layer(x, mask)
        return x
