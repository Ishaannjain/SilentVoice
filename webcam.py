import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from asl_dataset import ASLDataset, SEQ_LEN
from train_lstm import LSTMClassifier

MODEL_PATH = "asl_lstm_test.pt"
PRED_EVERY = 10
CONF_THRESH = 0.45

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(MODEL_PATH, map_location=device)
labels = ckpt["labels"]

model = LSTMClassifier(num_classes=len(labels)).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

dataset = ASLDataset("data/landmarks")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)
buffer = deque(maxlen=SEQ_LEN)
frame_count = 0
current_label = "..."

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

    if len(buffer) == SEQ_LEN and frame_count % PRED_EVERY == 0:
        seq, mask = dataset._process_sequence(np.array(buffer))
        x = seq.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(x, mask), dim=1)[0]
            idx = probs.argmax().item()
            conf = probs[idx].item()

        current_label = f"{labels[idx]} ({conf:.2f})" if conf >= CONF_THRESH else "..."

    cv2.putText(frame, current_label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

    cv2.imshow("SilentVoice - ASL", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
