import torch
import torch.nn as nn


class NERClassifier(nn.Module):
    def __init__(self, H, num_labels, P_drop=0.1):
        super().__init__()
        self.classifier = nn.Linear(H, num_labels)
        self.dropout = nn.Dropout(P_drop)

    def forward(self, hidden_states):
        return self.classifier(self.dropout(hidden_states))
