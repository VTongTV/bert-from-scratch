import csv

import torch
from torch.utils.data import Dataset


class SwagReader:
    def read(self, filepath):
        examples = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                examples.append({
                    "id": row["video-id"],
                    "sent1": row["sent1"],
                    "sent2": row["sent2"],
                    "ending0": row["ending0"],
                    "ending1": row["ending1"],
                    "ending2": row["ending2"],
                    "ending3": row["ending3"],
                    "label": int(row.get("label", 0)),
                })
        return examples


class SwagDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        choices = []
        for i in range(4):
            text_a = ex["sent1"] + " " + ex["sent2"]
            text_b = ex[f"ending{i}"]
            ids, segment_ids = self.tokenizer.encode_pair(text_a, text_b, self.max_len)
            choices.append({
                "input_ids": ids,
                "segment_ids": segment_ids,
            })
        return {
            "choices": choices,
            "label": ex["label"],
        }
