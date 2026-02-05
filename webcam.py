import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque

from asl_dataset import ASLDataset, SEQ_LEN, LANDMARKS_PER_HAND
from train_lstm import LSTMClassifier

MODEL_PATH  = "asl_lstm_test.pt"
PRED_EVERY  = 3
CONF_THRESH = 0.45          # higher = fewer false triggers
SMOOTH_WIN  = 10            # average over more predictions before deciding
LOCK_FRAMES = 5             # new word must win this many consecutive rounds before replacing current

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load model
# -------------------------
ckpt = torch.load(MODEL_PATH, map_location=device)
labels = ckpt["labels"]

model = LSTMClassifier(num_classes=len(labels)).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

dataset = ASLDataset("data/landmarks")

# -------------------------
# MediaPipe  (two hands)
# -------------------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

# Buffers
# SEQ_LEN is imported as 64 — matches the training sequence length
EMPTY_HAND      = [[0.0, 0.0, 0.0]] * LANDMARKS_PER_HAND
landmark_buffer = deque(maxlen=SEQ_LEN)
prob_buffer     = deque(maxlen=SMOOTH_WIN)

frame_count   = 0
current_label = "..."
locked_idx    = -1          # index of the currently displayed word (-1 = none)
candidate_idx = -1          # index of the word trying to replace the current one
candidate_streak = 0        # how many consecutive rounds the candidate has been on top

print("Keep hands in view of the camera")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process in ORIGINAL orientation so handedness labels match
    # extract_coordinates.py (which does not flip).
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw landmarks on the original frame, then flip everything together
    # so the on-screen overlay still looks correct in mirror view.
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    frame = cv2.flip(frame, 1)   # mirror for display

    # -------------------------
    # Extract landmarks  [right(63) | left(63)]  =  126 features
    # -------------------------
    if result.multi_hand_landmarks:
        right_hand = None
        left_hand  = None

        for i, hand in enumerate(result.multi_hand_landmarks):
            handedness = result.multi_handedness[i].classification[0].display_name
            # z * 0.3 must match extract_coordinates.py
            lms = [[lm.x, lm.y, lm.z * 0.3] for lm in hand.landmark]
            if handedness == "Right":
                right_hand = lms
            else:
                left_hand = lms

        frame_landmarks = (right_hand or EMPTY_HAND) + (left_hand or EMPTY_HAND)
        landmark_buffer.append(frame_landmarks)

    frame_count += 1

    # -------------------------
    # Run inference every PRED_EVERY frames once buffer is full
    # -------------------------
    if len(landmark_buffer) == SEQ_LEN and frame_count % PRED_EVERY == 0:
        # (T, 42, 3) flattened to (T, 126)
        seq = np.array(landmark_buffer, dtype=np.float32).reshape(SEQ_LEN, -1)

        # _process_sequence handles normalisation + padding — don't touch seq before this
        seq, mask = dataset._process_sequence(seq)

        x    = seq.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(x, mask), dim=1)[0]

        prob_buffer.append(probs)

        # Temporal smoothing over the last SMOOTH_WIN predictions
        avg_probs = torch.stack(list(prob_buffer)).mean(dim=0)
        idx  = avg_probs.argmax().item()
        conf = avg_probs[idx].item()

        if conf < CONF_THRESH:
            # confidence too low — reset everything, show nothing
            locked_idx       = -1
            candidate_idx    = -1
            candidate_streak = 0
            current_label    = "..."
        elif idx == locked_idx:
            # same word as what's already displayed — keep it, reset candidate
            candidate_idx    = -1
            candidate_streak = 0
        else:
            # different word trying to take over
            if idx == candidate_idx:
                candidate_streak += 1
            else:
                candidate_idx    = idx
                candidate_streak = 1

            if candidate_streak >= LOCK_FRAMES:
                # new word has been consistent long enough — lock it in
                locked_idx       = idx
                candidate_idx    = -1
                candidate_streak = 0
                current_label    = f"{labels[idx]} ({conf:.2f})"
            elif locked_idx == -1:
                # nothing displayed yet — show immediately on first confident hit
                locked_idx       = idx
                candidate_idx    = -1
                candidate_streak = 0
                current_label    = f"{labels[idx]} ({conf:.2f})"

    cv2.putText(
        frame,
        current_label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3
    )

    cv2.imshow("SilentVoice - ASL", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
