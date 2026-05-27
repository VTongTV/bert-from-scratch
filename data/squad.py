import json

import torch
from torch.utils.data import Dataset


class SquadReader:
    def read(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        examples = []
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    qas_id = qa["id"]
                    answers = []
                    for answer in qa["answers"]:
                        answers.append({
                            "text": answer["text"],
                            "start": answer["answer_start"],
                        })
                    examples.append({
                        "id": qas_id,
                        "question": question,
                        "context": context,
                        "answers": answers,
                    })
        return examples


def extract_spans(context, answer_text, answer_start, tokenizer):
    context_tokens = context.split()
    char_idx = 0
    token_start = None
    token_end = None
    for i, token in enumerate(context_tokens):
        if char_idx == answer_start:
            token_start = i
        char_idx += len(token) + 1
        if char_idx >= answer_start + len(answer_text) and token_end is None:
            token_end = i
    if token_start is None:
        token_start = 0
    if token_end is None:
        token_end = 0
    return token_start, token_end


def squad_to_features(examples, tokenizer, max_len=384, doc_stride=128):
    features = []
    for ex in examples:
        ids, segment_ids = tokenizer.encode_pair(ex["question"], ex["context"], max_len)
        start_position = 0
        end_position = 0
        if ex["answers"]:
            ans = ex["answers"][0]
            start_position, end_position = extract_spans(
                ex["context"], ans["text"], ans["start"], tokenizer
            )
            start_position += len(tokenizer.tokenize(ex["question"])) + 2
            end_position += len(tokenizer.tokenize(ex["question"])) + 2
        attention_mask = [1] * len(ids)
        features.append({
            "id": ex["id"],
            "input_ids": ids,
            "segment_ids": segment_ids,
            "attention_mask": attention_mask,
            "start_position": start_position,
            "end_position": end_position,
        })
    return features


class SquadDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        return {
            "input_ids": torch.tensor(f["input_ids"], dtype=torch.long),
            "segment_ids": torch.tensor(f["segment_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(f["attention_mask"], dtype=torch.long),
            "start_position": torch.tensor(f["start_position"], dtype=torch.long),
            "end_position": torch.tensor(f["end_position"], dtype=torch.long),
        }
