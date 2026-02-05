import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=252, hidden_dim=192, num_classes=4):
        super().__init__()

        # -------------------------
        # Temporal motion extractor
        # -------------------------
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Drop random frames to prevent memorization
        self.frame_dropout = nn.Dropout(p=0.2)

        # -------------------------
        # Sequence model
        # -------------------------
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )

        # -------------------------
        # Soft attention over time
        # -------------------------
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.attn_temp = nn.Parameter(torch.tensor(1.5))  # soften attention

        # -------------------------
        # Classifier
        # -------------------------
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, mask):
        """
        x:    (B, T, 252)  positions + velocities  (two hands × 21 landmarks × 3 × 2)
        mask: (B, T)      valid-frame mask
        """

        # -------------------------
        # Normalize motion per sequence
        # -------------------------
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x = (x - mean) / std

        # -------------------------
        # Temporal convolution
        # -------------------------
        x = x.transpose(1, 2)          # (B, 63, T)
        x = self.temporal_conv(x)      # (B, 128, T)
        x = x.transpose(1, 2)          # (B, T, 128)

        # -------------------------
        # Frame-level dropout
        # -------------------------
        x = self.frame_dropout(x)

        # -------------------------
        # LSTM
        # -------------------------
        out, _ = self.lstm(x)           # (B, T, 2*hidden_dim)

        # -------------------------
        # Attention pooling
        # -------------------------
        scores = self.attn(out).squeeze(-1)   # (B, T)
        scores = scores / self.attn_temp
        scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)    # (B, T)
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)

        # -------------------------
        # Classification
        # -------------------------
        return self.fc(context)

    def predict_proba(self, x, mask):
        """
        Returns probabilities for ALL classes.
        """
        logits = self.forward(x, mask)
        return torch.softmax(logits, dim=1)
