import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=256, num_classes=4):
        super().__init__()

        # motion extractor
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # sequence model
        self.lstm = nn.LSTM(
            128,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )

        # attention over frames
        self.attn = nn.Linear(hidden_dim * 2, 1)

        # classifier
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, mask):
        # x: (B, T, 63)
        # mask: (B, T)

        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)

        out, _ = self.lstm(x)

        scores = self.attn(out).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)

        context = torch.sum(out * weights.unsqueeze(-1), dim=1)
        return self.fc(context)

    def predict_proba(self, x, mask):
        """
        Returns probabilities for ALL classes.
        """
        logits = self.forward(x, mask)
        return torch.softmax(logits, dim=1)
