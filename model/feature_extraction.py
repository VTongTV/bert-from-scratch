import torch
import torch.nn as nn


class FeatureExtractor:
    def __init__(self, model):
        self.model = model

    def concat_last_4(self, input_ids, segment_ids=None, attention_mask=None):
        all_hidden = self.model.get_hidden_states(input_ids, segment_ids, attention_mask)
        last_4 = all_hidden[-4:]
        return torch.cat(last_4, dim=-1)

    def weighted_sum_last_4(self, input_ids, segment_ids=None, attention_mask=None):
        all_hidden = self.model.get_hidden_states(input_ids, segment_ids, attention_mask)
        last_4 = all_hidden[-4:]
        weights = torch.softmax(torch.randn(4), dim=0)
        result = sum(w * h for w, h in zip(weights, last_4))
        return result

    def weighted_sum_all(self, input_ids, segment_ids=None, attention_mask=None):
        all_hidden = self.model.get_hidden_states(input_ids, segment_ids, attention_mask)
        L = len(all_hidden)
        weights = torch.softmax(torch.randn(L), dim=0)
        result = sum(w * h for w, h in zip(weights, all_hidden))
        return result
