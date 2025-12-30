import torch
from collections import Counter
from torch.utils.data import DataLoader

from asl_dataset import ASLDataset
from train_lstm import LSTMClassifier

# -----------------
# Config
# -----------------
DATA_DIR = "data/landmarks"
MODEL_PATH = "asl_lstm_test.pt"
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# Load dataset
# -----------------
dataset = ASLDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------
# Load model
# -----------------
ckpt = torch.load(MODEL_PATH, map_location=device)
labels = ckpt["labels"]

model = LSTMClassifier(num_classes=len(labels)).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# -----------------
# Count predictions
# -----------------
prediction_counts = Counter()

with torch.no_grad():
    for x, mask, y in loader:
        x = x.to(device)
        mask = mask.to(device)

        logits = model(x, mask)
        preds = torch.argmax(logits, dim=1)

        for p in preds.cpu().tolist():
            prediction_counts[labels[p]] += 1

# -----------------
# Print results
# -----------------
print("\n=== PREDICTIONS ON TRAINING SET ===")
for label in labels:
    print(f"{label:12s}: {prediction_counts[label]}")

print("\nTotal samples:", len(dataset))
