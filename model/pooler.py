import torch
import torch.nn as nn


class BertPooler(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.dense = nn.Linear(H, H)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token = hidden_states[:, 0]
        return self.activation(self.dense(first_token))
