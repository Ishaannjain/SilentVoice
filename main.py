import os
from collections import Counter

import torch
from torch.utils.data import DataLoader
from asl_dataset import ASLDataset, label_from_filename
from train_lstm import LSTMClassifier

DATA_DIR   = "data/landmarks"
BATCH_SIZE = 4
EPOCHS     = 50    # more epochs — augmentation slows convergence
LR         = 5e-4
TOP_N      = 300   # train on the N most-sampled classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- auto-select top-N classes ----------
all_files      = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
label_counts   = Counter(label_from_filename(f) for f in all_files)
MIN_SAMPLES    = 10    # need enough per-class samples to generalise; <10 just memorises
ALLOWED_LABELS = {label for label, count in label_counts.most_common(TOP_N) if count >= MIN_SAMPLES}

print(f"Training on top {TOP_N} classes ({sum(label_counts[l] for l in ALLOWED_LABELS)} samples):")
for label in sorted(ALLOWED_LABELS):
    print(f"  {label:>12s}: {label_counts[label]}")

# training=True enables augmentation; val set has no augmentation
train_dataset = ASLDataset(DATA_DIR, split="train", training=True,  allowed_labels=ALLOWED_LABELS)
val_dataset   = ASLDataset(DATA_DIR, split="val",   training=False, allowed_labels=ALLOWED_LABELS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

model     = LSTMClassifier(num_classes=len(train_dataset.labels)).to(device)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # ---------- training ----------
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, mask, y in train_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x, mask)
        loss   = criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)

    train_acc = correct / total

    # ---------- validation ----------
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, mask, y in val_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            val_correct += (logits.argmax(1) == y).sum().item()
            val_total   += y.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1:02d} | loss={total_loss:.4f} | train={train_acc:.3f} | val={val_acc:.3f}")

    # Save only when val accuracy improves — this keeps the best-generalising
    # checkpoint rather than just the last epoch
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {"model_state": model.state_dict(), "labels": train_dataset.labels},
            "asl_lstm_test.pt"
        )
        print(f"  -> saved (best val={val_acc:.3f})")

print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
