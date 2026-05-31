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
