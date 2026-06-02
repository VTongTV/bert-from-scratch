import torch


def move_to_device(model, device):
    return model.to(device)
