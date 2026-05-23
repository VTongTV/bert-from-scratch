import torch


def create_padding_mask(input_ids, pad_id=0):
    return (input_ids != pad_id).float()


def create_attention_mask(input_ids, pad_id=0):
    padding_mask = create_padding_mask(input_ids, pad_id)
    return padding_mask.unsqueeze(1).unsqueeze(2)


def create_token_type_ids(input_ids, sep_id):
    token_type_ids = torch.zeros_like(input_ids)
    for i in range(input_ids.size(0)):
        sep_positions = (input_ids[i] == sep_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            token_type_ids[i, sep_positions[0] + 1:] = 1
    return token_type_ids
