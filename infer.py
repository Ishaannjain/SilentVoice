import torch
import numpy as np
from asl_dataset import ASLDataset, SEQ_LEN

# -----------------
# Config
# -----------------
MODEL_PATH = "asl_lstm_test.pt"
LANDMARK_FILE = "data/landmarks/drink_000.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# Load model + labels
# -----------------
ckpt = torch.load(MODEL_PATH, map_location=device)
labels = ckpt["labels"]
num_classes = len(labels)

class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_classes=4):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

model = LSTMClassifier(num_classes=num_classes).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# -----------------
# Load + process landmarks
# -----------------
seq = np.load(LANDMARK_FILE).astype(np.float32)

# reuse dataset logic for padding
dataset = ASLDataset("data/landmarks")
seq = dataset._process_sequence(seq)

x = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, 64, 63)

# -----------------
# Inference
# -----------------
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()

print("Predicted label:", labels[pred_idx])
print("Probabilities:")
for lbl, p in zip(labels, probs[0].cpu().numpy()):
    print(f"  {lbl:>6}: {p:.3f}")
