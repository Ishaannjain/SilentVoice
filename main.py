import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from asl_dataset import ASLDataset, SEQ_LEN

# -----------------
# Config
# -----------------
MODEL_PATH = "asl_lstm_test.pt"
PRED_EVERY = 10          # predict every N frames
CONF_THRESH = 0.45       # minimum confidence to show label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_classes=4):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# -----------------
# Load model + labels
# -----------------
ckpt = torch.load(MODEL_PATH, map_location=device)
labels = ckpt["labels"]
num_classes = len(labels)

model = LSTMClassifier(num_classes=num_classes).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# reuse dataset logic
dataset = ASLDataset("data/landmarks")

# -----------------
# MediaPipe setup
# -----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -----------------
# Webcam loop
# -----------------
cap = cv2.VideoCapture(0)
buffer = deque(maxlen=SEQ_LEN)
frame_count = 0
current_label = "..."

print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        buffer.append(landmarks)
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    frame_count += 1

    # -----------------
    # Run prediction
    # -----------------
    if len(buffer) >= SEQ_LEN and frame_count % PRED_EVERY == 0:
        seq = np.array(buffer, dtype=np.float32)
        seq = dataset._process_sequence(seq)

        x = torch.from_numpy(seq).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            idx = torch.argmax(probs).item()
            conf = probs[idx].item()

        if conf >= CONF_THRESH:
            current_label = f"{labels[idx]} ({conf:.2f})"
        else:
            current_label = "..."

    # -----------------
    # Overlay text
    # -----------------
    cv2.putText(
        frame,
        current_label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3
    )

    cv2.imshow("SilentVoice - ASL Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
