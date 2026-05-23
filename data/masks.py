import torch


def create_padding_mask(input_ids, pad_id=0):
    return (input_ids != pad_id).float()


def create_attention_mask(input_ids, pad_id=0):
    padding_mask = create_padding_mask(input_ids, pad_id)
    return padding_mask.unsqueeze(1).unsqueeze(2)
