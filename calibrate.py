"""
Fine-tune the ASL model on your calibration recordings.

This script:
1. Loads the pre-trained model (asl_merged.pt)
2. Freezes the feature extraction layers (Conv, LSTM)
3. Fine-tunes only the classifier head on your recordings
4. Saves the personalized model (asl_calibrated.pt)

Usage:
    python finetune_calibration.py
"""

import os
import torch
import numpy as np
from pathlib import Path
from collections import Counter

from asl_dataset import ASLDataset, label_from_filename
from model import LSTMClassifier


CALIBRATION_DIR = Path("data/calibration")
PRETRAINED_MODEL = "asl_merged.pt"
OUTPUT_MODEL = "asl_calibrated.pt"


EPOCHS = 50
LR = 5e-4  
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_calibration_data(calibration_dir, label_to_idx):
    """Load calibration .npy files and prepare for training."""
    from asl_dataset import SEQ_LEN

    dataset = ASLDataset("data/landmarks_merged")

    files = list(calibration_dir.glob("*.npy"))
    if not files:
        return None, None, None

    sequences = []
    masks = []
    labels = []

    for f in files:
        word = label_from_filename(f.name)

        if word not in label_to_idx:
            print(f"  Warning: '{word}' not in model vocabulary, skipping")
            continue

    
        raw = np.load(f)  # (T, 126)

    
        if raw.shape[0] > SEQ_LEN:
            raw = raw[:SEQ_LEN]

    
        seq, mask = dataset._process_sequence(raw)

        sequences.append(seq.numpy())
        masks.append(mask.numpy())
        labels.append(label_to_idx[word])

    if not sequences:
        return None, None, None

    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    M = torch.tensor(np.array(masks), dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.long)

    return X, M, Y


def main():
    print("="*60)
    print("FINE-TUNING ON CALIBRATION DATA")
    print("="*60)

    # Check calibration data exists
    if not CALIBRATION_DIR.exists():
        print(f"Error: Calibration directory not found: {CALIBRATION_DIR}")
        print("Run 'python record_calibration.py' first")
        return

    cal_files = list(CALIBRATION_DIR.glob("*.npy"))
    if not cal_files:
        print(f"Error: No calibration files found in {CALIBRATION_DIR}")
        print("Run 'python record_calibration.py' first")
        return

    cal_counts = Counter(label_from_filename(f.name) for f in cal_files)
    print(f"\nCalibration data:")
    for word, count in sorted(cal_counts.items()):
        print(f"  {word}: {count} recordings")
    print(f"Total: {len(cal_files)} recordings")

    print(f"\nLoading pre-trained model: {PRETRAINED_MODEL}")
    if not os.path.exists(PRETRAINED_MODEL):
        print(f"Error: Model not found: {PRETRAINED_MODEL}")
        return

    checkpoint = torch.load(PRETRAINED_MODEL, map_location=device, weights_only=False)
    model_labels = checkpoint["labels"]
    label_to_idx = {l: i for i, l in enumerate(model_labels)}

    print(f"Model vocabulary: {len(model_labels)} words")

    missing = [w for w in cal_counts.keys() if w not in label_to_idx]
    if missing:
        print(f"\nWarning: These words are not in model vocabulary: {missing}")
        print("They will be skipped during fine-tuning")

    # Load model
    model = LSTMClassifier(num_classes=len(model_labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    
    print("\nUnfreezing LSTM, attention, and FC layers...")
    for name, param in model.named_parameters():
    
        if "conv" in name or "bn" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"  Training: {name}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    
    X, M, Y = load_calibration_data(CALIBRATION_DIR, label_to_idx)
    if X is None:
        print("Error: No valid calibration data")
        return

    print(f"\nLoaded {len(X)} calibration samples")

    
    dataset = torch.utils.data.TensorDataset(X, M, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    print(f"\nFine-tuning for {EPOCHS} epochs...")
    print("-"*40)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1:02d} | loss={total_loss:.4f} | acc={acc:.3f}")

  
    torch.save(
        {"model_state": model.state_dict(), "labels": model_labels},
        OUTPUT_MODEL
    )
    print(f"\nSaved fine-tuned model to: {OUTPUT_MODEL}")

    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE")
    print("="*60)
    print(f"Original model: {PRETRAINED_MODEL}")
    print(f"Calibrated model: {OUTPUT_MODEL}")
    print()
    print("To test: Update webcam.py to use 'asl_calibrated.pt'")
    print("         then run 'python webcam.py'")


if __name__ == "__main__":
    main()
