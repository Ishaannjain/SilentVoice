"""
Train the ASL recognition model.

Usage:
    python train.py                  # Train on merged dataset (default)
    python train.py --dataset merged # Train on MS-ASL + WLASL100 merged
    python train.py --dataset wlasl  # Train on WLASL100 only
"""
import os
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader
from asl_dataset import ASLDataset, label_from_filename
from model import LSTMClassifier

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

DATASETS = {
    "merged": {
        "path": "data/landmarks_merged",
        "output": "asl_merged.pt",
        "name": "MS-ASL + WLASL100 Merged"
    },
    "wlasl": {
        "path": "data/landmarks_wlasl100",
        "output": "asl_wlasl100.pt",
        "name": "WLASL100"
    }
}

BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 3e-4
MIN_SAMPLES  = 10  # Filter words with fewer samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Train ASL recognition model")
    parser.add_argument(
        "--dataset",
        choices=["merged", "wlasl"],
        default="merged",
        help="Dataset to train on (default: merged)"
    )
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    DATA_DIR = config["path"]
    OUTPUT_MODEL = config["output"]

    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print(f"Dataset: {config['name']}")
    print(f"{'='*60}")

    # ─────────────────────────────────────────────────────────────
    # Load and filter classes
    # ─────────────────────────────────────────────────────────────

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found: {DATA_DIR}")
        return

    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
    label_counts = Counter(label_from_filename(f) for f in all_files)
    allowed_labels = {label for label, count in label_counts.items() if count >= MIN_SAMPLES}

    print(f"\nTotal files: {len(all_files)}")
    print(f"Total words: {len(label_counts)}")
    print(f"Words with >= {MIN_SAMPLES} samples: {len(allowed_labels)}")
    print(f"Training samples: {sum(label_counts[l] for l in allowed_labels)}")

    # ─────────────────────────────────────────────────────────────
    # Create datasets
    # ─────────────────────────────────────────────────────────────

    train_dataset = ASLDataset(DATA_DIR, split="train", training=True, allowed_labels=allowed_labels)
    val_dataset = ASLDataset(DATA_DIR, split="val", training=False, allowed_labels=allowed_labels)

    print(f"\nTrain: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Classes: {len(train_dataset.labels)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    # ─────────────────────────────────────────────────────────────
    # Model, optimizer, scheduler
    # ─────────────────────────────────────────────────────────────

    model = LSTMClassifier(num_classes=len(train_dataset.labels)).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ─────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────

    best_val_acc = 0.0
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x, mask, y in train_loader:
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

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, mask, y in val_loader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                logits = model(x, mask)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1:02d} | loss={total_loss:.4f} | train={train_acc:.3f} | val={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "labels": train_dataset.labels},
                OUTPUT_MODEL
            )
            print(f"  -> saved (best val={val_acc:.3f})")

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
