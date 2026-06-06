import torch
from torch.optim import AdamW


def create_adam_optimizer(model, lr=1e-4, b1=0.9, b2=0.999, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(params, lr=lr, betas=(b1, b2))
