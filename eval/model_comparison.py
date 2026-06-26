import torch

from config.bert_config import BertConfig
from model.bert import BertModel, count_parameters


def perplexity(loss):
    return torch.exp(torch.tensor(loss))


def model_size_comparison():
    configs = {
        "small": BertConfig(L=3, H=768, A=12),
        "medium_3": BertConfig(L=6, H=768, A=3),
        "medium_12": BertConfig(L=6, H=768, A=12),
        "base": BertConfig(L=12, H=768, A=12),
        "large": BertConfig(L=24, H=1024, A=16),
    }
    results = {}
    for name, config in configs.items():
        model = BertModel(config)
        results[name] = {"params": count_parameters(model), "L": config.L, "H": config.H, "A": config.A}
    return results
