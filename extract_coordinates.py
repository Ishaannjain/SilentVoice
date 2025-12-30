import os
import cv2
import time
import numpy as np
import mediapipe as mp


class VideoLandmarkExtractor:
    def __init__(
        self,
        video_dir,
        out_dir,
        log_file="failed_landmark_extraction.txt",
        max_frames=250,
        max_seconds=4,
        min_frames=5,
        no_hand_abort=40,
        min_fps=5,
        max_fps=120,
    ):
        self.video_dir = video_dir
        self.out_dir = out_dir
        self.log_file = log_file

        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self.min_frames = min_frames
        self.no_hand_abort = no_hand_abort
        self.min_fps = min_fps
        self.max_fps = max_fps

        os.makedirs(self.out_dir, exist_ok=True)

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    # -------------------------
    # HARD VIDEO VALIDATION
    # -------------------------
    def _is_valid_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps < self.min_fps or fps > self.max_fps:
            return False
        if frames <= 0 or frames > 10000:
            return False

        return True

    # -------------------------
    # LANDMARK EXTRACTION
    # -------------------------
    def _extract_landmarks(self, path):
        if not self._is_valid_video(path):
            return None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return None

        sequence = []
        start_time = time.time()
        frame_count = 0
        no_hand_counter = 0

        while True:
            if time.time() - start_time > self.max_seconds:
                cap.release()
                return None

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count > self.max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            if result.multi_hand_landmarks:
                no_hand_counter = 0
                hand = result.multi_hand_landmarks[0]
                frame_landmarks = []
                for lm in hand.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
                sequence.append(frame_landmarks)
            else:
                no_hand_counter += 1
                if no_hand_counter > self.no_hand_abort:
                    cap.release()
                    return None

        cap.release()

        if len(sequence) < self.min_frames:
            return None

        return np.array(sequence, dtype=np.float32)

    # -------------------------
    # MAIN PIPELINE
    # -------------------------
    def run(self):
        videos = [f for f in os.listdir(self.video_dir) if f.lower().endswith(".mp4")]
        print(f"Found {len(videos)} .mp4 files")

        for vid in videos:
            video_path = os.path.join(self.video_dir, vid)
            out_path = os.path.join(self.out_dir, vid.replace(".mp4", ".npy"))

            if os.path.exists(out_path):
                continue

            print(f"Processing {vid}")

            try:
                landmarks = self._extract_landmarks(video_path)
                if landmarks is None:
                    raise RuntimeError("Skipped")

                np.save(out_path, landmarks)

            except Exception:
                print(f"Failed on {vid}")
                with open(self.log_file, "a") as log:
                    log.write(f"{vid}\n")


# =====================================================
# AUTO-RUN WHEN FILE IS EXECUTED
# =====================================================
if __name__ == "__main__":
    extractor = VideoLandmarkExtractor(
        video_dir="data/videos",
        out_dir="data/landmarks"
    )

    extractor.run()
