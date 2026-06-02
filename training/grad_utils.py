import torch


def clip_grad_norm(parameters, max_norm=1.0):
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
