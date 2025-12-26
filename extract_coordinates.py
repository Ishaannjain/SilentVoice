import os
import cv2
import numpy as np
import mediapipe as mp

VIDEO_DIR = "data/videos"
OUT_DIR = "data/landmarks"
LOG_FILE = "failed_landmark_extraction.txt"

os.makedirs(OUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    if not cap.isOpened():
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            frame_landmarks = []
            for lm in hand.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
            sequence.append(frame_landmarks)

    cap.release()

    if len(sequence) == 0:
        return None

    return np.array(sequence, dtype=np.float32)


# =========================
# Main loop
# =========================

videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
print(f"Found {len(videos)} videos")

for vid in videos:
    video_path = os.path.join(VIDEO_DIR, vid)
    out_path = os.path.join(OUT_DIR, vid.replace(".mp4", ".npy"))

    if os.path.exists(out_path):
        continue

    print(f"Processing {vid}")

    try:
        landmarks = extract_landmarks(video_path)
        if landmarks is None:
            raise ValueError("No landmarks extracted")

        np.save(out_path, landmarks)

       

    except Exception as e:
        print(f"Failed on {vid}: {e}")
        with open(LOG_FILE, "a") as log:
            log.write(f"{vid}\n")
