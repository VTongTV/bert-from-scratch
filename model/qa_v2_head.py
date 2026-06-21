import torch
import torch.nn as nn
import torch.nn.functional as F

from model.qa_head import QAHead, qa_loss, best_span


class QAv2Head(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.qa_head = QAHead(H)
        self.has_answer = nn.Linear(H, 1)

    def forward(self, hidden_states, cls_hidden=None):
        start_logits, end_logits = self.qa_head(hidden_states)
        if cls_hidden is None:
            cls_hidden = hidden_states[:, 0]
        null_score = self.has_answer(cls_hidden).squeeze(-1)
        return start_logits, end_logits, null_score


def null_score_s(start_logits, end_logits):
    return start_logits[0] + end_logits[0]


def predict_span_v2(start_logits, end_logits, null_score, tau=0.0):
    s, e = best_span(start_logits, end_logits)
    span_s = start_logits[s] + end_logits[e]
    if span_s > null_score + tau:
        return s, e
    return 0, 0
