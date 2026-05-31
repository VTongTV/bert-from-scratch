import torch
import torch.nn as nn

from config.bert_config import BertConfig
from model.embeddings import BertEmbeddings
from model.encoder import BertEncoder
from model.pooler import BertPooler


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config.V, config.H, config.max_len)
        self.encoder = BertEncoder(config.L, config.H, config.A, config.d_ff, config.P_drop)
        self.pooler = BertPooler(config.H)

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        embedding_output = self.embeddings(input_ids, segment_ids)
        encoder_output = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output

    def get_hidden_states(self, input_ids, segment_ids=None, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        embedding_output = self.embeddings(input_ids, segment_ids)
        all_hidden = [embedding_output]
        x = embedding_output
        for layer in self.encoder.layer:
            x = layer(x, attention_mask)
            all_hidden.append(x)
        return all_hidden

    def get_attention_weights(self, input_ids, segment_ids=None, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        embedding_output = self.embeddings(input_ids, segment_ids)
        all_attn = []
        x = embedding_output
        for layer in self.encoder.layer:
            Q = layer.attention.W_Q(x)
            K = layer.attention.W_K(x)
            from model.attention import compute_attention_scores, split_heads
            B, S, H = Q.size()
            A = layer.attention.A
            Q = split_heads(Q, A)
            K = split_heads(K, A)
            scores = compute_attention_scores(Q, K)
            import torch.nn.functional as F
            attn = F.softmax(scores, dim=-1)
            all_attn.append(attn)
            x = layer(x, attention_mask)
        return all_attn


def init_bert_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
        nn.init.zeros_(module.bias)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
