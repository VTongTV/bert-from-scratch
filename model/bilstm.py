import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, H, hidden_size, num_classes, num_layers=1, P_drop=0.1):
        super().__init__()
        self.lstm = nn.LSTM(H, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(P_drop)

    def forward(self, hidden_states):
        lstm_out, _ = self.lstm(hidden_states)
        pooled = lstm_out[:, 0]
        return self.classifier(self.dropout(pooled))
