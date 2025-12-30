import torch
from torch.utils.data import DataLoader
from asl_dataset import ASLDataset
from train_lstm import LSTMClassifier

DATA_DIR = "data/landmarks"
BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ASLDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = LSTMClassifier(num_classes=len(dataset.labels)).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1:02d} | loss={total_loss:.4f} | acc={acc:.3f}")

torch.save(
    {"model_state": model.state_dict(), "labels": dataset.labels},
    "asl_lstm_test.pt"
)

print("Saved model to asl_lstm_test.pt")
