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
        failed_cache_file="failed_landmark_videos_cache.txt",
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
        self.failed_cache_file = failed_cache_file

        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self.min_frames = min_frames
        self.no_hand_abort = no_hand_abort
        self.min_fps = min_fps
        self.max_fps = max_fps

        os.makedirs(self.out_dir, exist_ok=True)

        # Load failed cache (so we don't keep retrying bad vids)
        self.failed_videos = set()
        if os.path.exists(self.failed_cache_file):
            with open(self.failed_cache_file, "r", encoding="utf-8") as f:
                self.failed_videos = set(line.strip() for line in f if line.strip())

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def _mark_failed(self, vid_name: str):
        if vid_name in self.failed_videos:
            return
        self.failed_videos.add(vid_name)
        with open(self.failed_cache_file, "a", encoding="utf-8") as f:
            f.write(vid_name + "\n")
        with open(self.log_file, "a", encoding="utf-8") as log:
            log.write(vid_name + "\n")

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

        # Placeholder for a missing hand: 21 landmarks × 3 coords of zeros
        empty_hand = [0.0] * (21 * 3)

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

                # Sort detected hands into right / left using MediaPipe's label
                right_hand = None
                left_hand  = None
                for i, hand in enumerate(result.multi_hand_landmarks):
                    handedness = result.multi_handedness[i].classification[0].display_name
                    lms = []
                    for lm in hand.landmark:
                        lms.extend([lm.x, lm.y, lm.z * 0.3])
                    if handedness == "Right":
                        right_hand = lms
                    else:
                        left_hand = lms

                # Always store as [right(63), left(63)] = 126 features
                frame_landmarks = (right_hand or empty_hand) + (left_hand or empty_hand)
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
        videos = sorted([f for f in os.listdir(self.video_dir) if f.lower().endswith(".mp4")])
        print(f"Found {len(videos)} .mp4 files")

        done = 0
        skipped_existing = 0
        skipped_failed_cache = 0
        failed = 0

        for vid in videos:
            if vid in self.failed_videos:
                skipped_failed_cache += 1
                continue

            video_path = os.path.join(self.video_dir, vid)
            out_path = os.path.join(self.out_dir, vid.replace(".mp4", ".npy"))

            # Resume: skip if already extracted
            if os.path.exists(out_path):
                skipped_existing += 1
                continue

            print(f"Processing {vid}")

            landmarks = self._extract_landmarks(video_path)
            if landmarks is None:
                print(f"Failed on {vid}")
                self._mark_failed(vid)
                failed += 1
                continue

            np.save(out_path, landmarks)
            done += 1

        print("\n=== EXTRACTION SUMMARY ===")
        print("Extracted new:", done)
        print("Skipped (already exists):", skipped_existing)
        print("Skipped (failed cache):", skipped_failed_cache)
        print("Failed newly:", failed)
        print("Failed videos log:", self.log_file)
        print("Failed cache:", self.failed_cache_file)


if __name__ == "__main__":
    extractor = VideoLandmarkExtractor(
        video_dir="data/videos",
        out_dir="data/landmarks"
    )
    extractor.run()
