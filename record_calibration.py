"""
Record calibration samples for personalized ASL recognition.

Usage:
    python record_calibration.py

Controls:
    SPACE  - Start/stop recording
    N      - Skip to next word
    Q      - Quit

Records landmarks to data/calibration/ directory.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import time

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# Words to calibrate (start with just 5 common ones)
CALIBRATION_WORDS = ["book", "drink", "hello", "yes", "no"]

# How many recordings per word
RECORDINGS_PER_WORD = 5

# Recording duration in seconds
RECORD_DURATION = 2.5

# Output directory
OUTPUT_DIR = Path("data/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MediaPipe setup (same as extract_coordinates.py)
# ─────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_frame_landmarks(frame_rgb):
    """
    Extract hand landmarks from a single frame.
    Returns: (126,) array [right_hand(63), left_hand(63)]
    """
    result = hands.process(frame_rgb)

    landmarks = np.zeros(126, dtype=np.float32)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks, result.multi_handedness
        ):
            # Determine if right or left hand
            label = handedness.classification[0].label
            offset = 0 if label == "Right" else 63

            for i, lm in enumerate(hand_landmarks.landmark):
                landmarks[offset + i*3]     = lm.x
                landmarks[offset + i*3 + 1] = lm.y
                landmarks[offset + i*3 + 2] = 0.0  # z=0 to match WLASL100 training data

    return landmarks, result


def count_existing_recordings(word):
    """Count how many recordings already exist for this word."""
    existing = list(OUTPUT_DIR.glob(f"{word}_*.npy"))
    return len(existing)


def get_next_filename(word):
    """Get the next available filename for this word."""
    idx = count_existing_recordings(word)
    return OUTPUT_DIR / f"{word}_cal{idx:03d}.npy"


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("="*60)
    print("CALIBRATION RECORDING")
    print("="*60)
    print(f"Words to record: {CALIBRATION_WORDS}")
    print(f"Recordings per word: {RECORDINGS_PER_WORD}")
    print()
    print("Controls:")
    print("  SPACE - Start recording (hold the sign for 2.5 seconds)")
    print("  N     - Skip to next word")
    print("  Q     - Quit")
    print("="*60)

    word_idx = 0
    recording_idx = 0

    is_recording = False
    recorded_frames = []
    record_start_time = 0

    while word_idx < len(CALIBRATION_WORDS):
        word = CALIBRATION_WORDS[word_idx]
        existing = count_existing_recordings(word)
        needed = RECORDINGS_PER_WORD - existing

        if needed <= 0:
            print(f"Skipping '{word}' - already have {existing} recordings")
            word_idx += 1
            recording_idx = 0
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, result = extract_frame_landmarks(frame_rgb)

        # Draw hand landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Mirror for display
        frame = cv2.flip(frame, 1)

        # Recording logic
        if is_recording:
            recorded_frames.append(landmarks)
            elapsed = time.time() - record_start_time

            # Progress bar
            progress = min(elapsed / RECORD_DURATION, 1.0)
            bar_width = 300
            bar_x = 170
            bar_y = 420
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 20), (0, 255, 0), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (255, 255, 255), 2)

            # Recording indicator
            cv2.circle(frame, (50, 50), 15, (0, 0, 255), -1)
            cv2.putText(frame, "RECORDING", (75, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Auto-stop after duration
            if elapsed >= RECORD_DURATION:
                is_recording = False

                # Save recording
                if len(recorded_frames) > 10:  # Minimum frames
                    arr = np.array(recorded_frames, dtype=np.float32)
                    filename = get_next_filename(word)
                    np.save(filename, arr)
                    print(f"  Saved: {filename.name} ({len(recorded_frames)} frames)")
                    recording_idx += 1
                else:
                    print(f"  Discarded: too few frames ({len(recorded_frames)})")

                recorded_frames = []

                # Check if done with this word
                if recording_idx >= needed:
                    word_idx += 1
                    recording_idx = 0

        # UI overlay
        cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0), -1)

        # Word prompt
        cv2.putText(frame, f"Sign: {word.upper()}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Progress
        total_done = existing + recording_idx
        cv2.putText(frame, f"Recording {total_done + 1}/{RECORDINGS_PER_WORD}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Word progress
        cv2.putText(frame, f"Word {word_idx + 1}/{len(CALIBRATION_WORDS)}", (480, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Instructions
        if not is_recording:
            cv2.putText(frame, "Press SPACE to start recording", (150, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Calibration Recording", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and not is_recording:
            # Start recording
            is_recording = True
            recorded_frames = []
            record_start_time = time.time()
            print(f"\nRecording '{word}' ({recording_idx + 1}/{needed})...")

        elif key == ord('n'):
            # Skip to next word
            is_recording = False
            recorded_frames = []
            word_idx += 1
            recording_idx = 0
            print(f"\nSkipped to next word")

        elif key == ord('q'):
            print("\nQuitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Summary
    print()
    print("="*60)
    print("RECORDING COMPLETE")
    print("="*60)
    total = len(list(OUTPUT_DIR.glob("*.npy")))
    print(f"Total calibration samples: {total}")
    for word in CALIBRATION_WORDS:
        count = count_existing_recordings(word)
        print(f"  {word}: {count}")
    print()
    print("Next: Run 'python finetune_calibration.py' to adapt the model")


if __name__ == "__main__":
    main()
