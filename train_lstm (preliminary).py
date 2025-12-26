import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from asl_dataset import ASLDataset

# -----------------
# Config
# -----------------
DATA_DIR = "data/landmarks"
SEQ_LEN = 64
BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# Dataset
# -----------------
dataset = ASLDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(dataset.labels)
print("Classes:", dataset.labels)

# -----------------
# Model
# -----------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, 63)
        out, _ = self.lstm(x)
        last = out[:, -1, :]      # last timestep
        return self.fc(last)

model = LSTMClassifier(num_classes=num_classes).to(device)

# -----------------
# Training setup
# -----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------
# Training loop
# -----------------
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1:02d} | loss={total_loss:.4f} | acc={acc:.2f}")

# -----------------
# Save model
# -----------------
torch.save({
    "model_state": model.state_dict(),
    "labels": dataset.labels
}, "asl_lstm_test.pt")

print("Saved model to asl_lstm_test.pt")
