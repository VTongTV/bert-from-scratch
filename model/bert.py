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
