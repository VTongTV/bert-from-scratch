import torch


def pretrain_collate(batch):
    input_ids = torch.tensor([e.input_ids for e in batch], dtype=torch.long)
    segment_ids = torch.tensor([e.segment_ids for e in batch], dtype=torch.long)
    attention_mask = torch.tensor([e.attention_mask for e in batch], dtype=torch.long)
    masked_ids = torch.tensor([e.masked_ids for e in batch], dtype=torch.long)
    mlm_labels = torch.tensor([e.mlm_labels for e in batch], dtype=torch.long)
    nsp_labels = torch.tensor([e.nsp_label for e in batch], dtype=torch.long)
    return input_ids, segment_ids, attention_mask, masked_ids, mlm_labels, nsp_labels
