import torch
import numpy as np
from asl_dataset import ASLDataset
from train_lstm import LSTMClassifier

# -----------------
# Config
# -----------------
MODEL_PATH = "asl_lstm_test.pt"
LANDMARK_FILE = "data/landmarks/afternoon_000.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# Load model + labels
# -----------------
ckpt = torch.load(MODEL_PATH, map_location=device)
labels = ckpt["labels"]

model = LSTMClassifier(num_classes=len(labels)).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# -----------------
# Load + process landmarks
# -----------------
seq = np.load(LANDMARK_FILE).astype(np.float32)

# reuse dataset logic for normalization + padding
dataset = ASLDataset("data/landmarks")
seq, mask = dataset._process_sequence(seq)

x = seq.unsqueeze(0).to(device)
mask = mask.unsqueeze(0).to(device)

# -----------------
# Inference
# -----------------
with torch.no_grad():
    logits = model(x, mask)
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()

print("Predicted label:", labels[pred_idx])
print("Probabilities:")
for lbl, p in zip(labels, probs[0].cpu().numpy()):
    print(f"  {lbl:>6}: {p:.3f}")
