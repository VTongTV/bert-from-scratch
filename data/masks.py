import torch


def create_padding_mask(input_ids, pad_id=0):
    return (input_ids != pad_id).float()
