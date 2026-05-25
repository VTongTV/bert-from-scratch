from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from data.tokens import CLS, SEP, PAD, MASK
from data.mlm import apply_mlm_mask
from data.nsp import nsp_label
from data.nsp_sampler import NSPSampler


@dataclass
class PretrainExample:
    input_ids: list
    segment_ids: list
    attention_mask: list
    masked_ids: list
    mlm_labels: list
    nsp_label: int


class PretrainDataset(Dataset):
    def __init__(self, corpus, tokenizer, max_len=128, mlm_prob=0.15):
        self.sampler = NSPSampler(corpus)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.rng = __import__("random").Random()

    def __len__(self):
        return len(self.sampler.corpus)

    def __getitem__(self, idx):
        text_a, text_b, is_next = self.sampler.sample(self.rng)
        ids, segment_ids = self.tokenizer.encode_pair(text_a, text_b, self.max_len)
        attention_mask = [1] * len(ids)
        masked_ids, mlm_labels = apply_mlm_mask(ids, self.rng, len(self.tokenizer.vocab), self.mlm_prob)
        while len(ids) < self.max_len:
            ids.append(self.tokenizer.vocab.get_id(PAD))
            segment_ids.append(0)
            attention_mask.append(0)
            masked_ids.append(self.tokenizer.vocab.get_id(PAD))
            mlm_labels.append(-1)
        return PretrainExample(
            input_ids=ids[:self.max_len],
            segment_ids=segment_ids[:self.max_len],
            attention_mask=attention_mask[:self.max_len],
            masked_ids=masked_ids[:self.max_len],
            mlm_labels=mlm_labels[:self.max_len],
            nsp_label=nsp_label(is_next),
        )
