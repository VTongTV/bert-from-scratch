from torch.utils.data import DataLoader

from data.pretrain_data import PretrainDataset
from data.collator import pretrain_collate


def create_pretrain_dataloader(corpus, tokenizer, batch_size=256, max_len=128, mlm_prob=0.15, shuffle=True):
    dataset = PretrainDataset(corpus, tokenizer, max_len, mlm_prob)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pretrain_collate)


def dynamic_padding_collate(batch):
    max_len = max(len(e.input_ids) for e in batch)
    padded = []
    for e in batch:
        pad_len = max_len - len(e.input_ids)
        padded.append({
            "input_ids": e.input_ids + [0] * pad_len,
            "segment_ids": e.segment_ids + [0] * pad_len,
            "attention_mask": e.attention_mask + [0] * pad_len,
        })
    import torch
    return {
        "input_ids": torch.tensor([p["input_ids"] for p in padded], dtype=torch.long),
        "segment_ids": torch.tensor([p["segment_ids"] for p in padded], dtype=torch.long),
        "attention_mask": torch.tensor([p["attention_mask"] for p in padded], dtype=torch.long),
    }
