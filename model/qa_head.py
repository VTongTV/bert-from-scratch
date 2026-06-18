import torch
import torch.nn as nn
import torch.nn.functional as F


class QAHead(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.start_linear = nn.Linear(H, 1)
        self.end_linear = nn.Linear(H, 1)

    def forward(self, hidden_states):
        start_logits = self.start_linear(hidden_states).squeeze(-1)
        end_logits = self.end_linear(hidden_states).squeeze(-1)
        return start_logits, end_logits


def qa_loss(start_logits, end_logits, start_positions, end_positions):
    loss_start = F.cross_entropy(start_logits, start_positions)
    loss_end = F.cross_entropy(end_logits, end_positions)
    return (loss_start + loss_end) / 2


def span_score(start_logits, end_logits, i, j):
    return start_logits[i] + end_logits[j]


def best_span(start_logits, end_logits, max_len=None):
    S = start_logits.shape[-1]
    if max_len is not None:
        S = min(S, max_len)
    best_score = float("-inf")
    best_start = 0
    best_end = 0
    for i in range(S):
        for j in range(i, S):
            score = start_logits[i] + end_logits[j]
            if score > best_score:
                best_score = score
                best_start = i
                best_end = j
    return best_start, best_end


def best_span_batch(start_logits, end_logits, max_len=None):
    starts = []
    ends = []
    for s, e in zip(start_logits, end_logits):
        si, ei = best_span(s, e, max_len)
        starts.append(si)
        ends.append(ei)
    return starts, ends
