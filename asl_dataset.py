import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

SEQ_LEN = 64
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURE_DIM = NUM_HANDS * LANDMARKS_PER_HAND * 3  # 126  (positions)
MODEL_INPUT_DIM = FEATURE_DIM * 2                  # 252  (positions + velocities)


def label_from_filename(fname: str) -> str:
    """word_NNN.npy -> word   |   every_morning_013.npy -> every_morning
    The last _-segment is always the numeric index; everything before it is the label."""
    return "_".join(fname.replace(".npy", "").split("_")[:-1])


class ASLDataset(Dataset):
    """
    Loads hand-landmark sequences from .npy files.

    split:    "all" | "train" | "val"
    training: if True, applies augmentation in __getitem__
    """

    def __init__(self, landmark_dir, split="all", val_ratio=0.15, seed=42, training=False, allowed_labels=None):
        self.dir = landmark_dir
        self.training = training

        all_files = sorted([f for f in os.listdir(landmark_dir) if f.endswith(".npy")])

        # Optionally restrict to a subset of classes (e.g. top-N by count)
        if allowed_labels is not None:
            all_files = [f for f in all_files if label_from_filename(f) in allowed_labels]

        # Labels come from filenames: word_NNN.npy  ->  word
        self.labels = sorted(list(set(label_from_filename(f) for f in all_files)))
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

        if split == "all":
            self.files = all_files
        else:
            self.files = self._stratified_split(all_files, split, val_ratio, seed)

    # -------------------------
    # Stratified split
    # -------------------------
    def _stratified_split(self, all_files, split, val_ratio, seed):
        """Same ratio of val samples taken from every class. Deterministic via seed."""
        rng = random.Random(seed)
        by_label = {}
        for f in all_files:
            label = label_from_filename(f)
            by_label.setdefault(label, []).append(f)

        train_files, val_files = [], []
        for label in sorted(by_label):
            files = by_label[label][:]
            rng.shuffle(files)
            n_val = max(1, int(len(files) * val_ratio))
            val_files.extend(files[:n_val])
            train_files.extend(files[n_val:])

        return train_files if split == "train" else val_files

    def __len__(self):
        return len(self.files)

    # -------------------------
    # Normalization (per-hand)
    # -------------------------
    def _normalize_hand(self, hand):
        """
        hand: (T, 21, 3)
        Centers on wrist, scales by mean palm length.
        All-zero hands (not detected) are left untouched.
        """
        if hand.abs().sum() < 1e-6:
            return hand

        # Translation: center on wrist (landmark 0)
        wrist = hand[:, 0:1, :]
        hand = hand - wrist

        # Scale: mean distance to middle-finger tip (landmark 9),
        # ignoring frames where the hand wasn't detected
        palm_norms = torch.norm(hand[:, 9, :], dim=1)  # (T,)
        nonzero = palm_norms > 1e-6
        scale = palm_norms[nonzero].mean() if nonzero.any() else torch.tensor(1.0)
        hand = hand / (scale + 1e-6)

        return hand

    def _normalize(self, seq):
        """
        seq: numpy array or tensor of shape (T, FEATURE_DIM).
        Normalizes each hand independently, returns (T, FEATURE_DIM) tensor.
        """
        seq = torch.as_tensor(seq, dtype=torch.float32)
        seq = seq.view(seq.shape[0], NUM_HANDS * LANDMARKS_PER_HAND, 3)  # (T, 42, 3)

        right = self._normalize_hand(seq[:, :LANDMARKS_PER_HAND, :])
        left  = self._normalize_hand(seq[:, LANDMARKS_PER_HAND:, :])

        seq = torch.cat([right, left], dim=1)
        return seq.view(seq.shape[0], -1)  # (T, FEATURE_DIM)

    # -------------------------
    # Augmentation (training only)
    # -------------------------
    def _rotate_2d(self, seq, angle_deg):
        """Rotate all landmark x,y coordinates around the origin by angle_deg.
        Works on (T, FEATURE_DIM) where every 3 values are (x, y, z)."""
        angle  = angle_deg * math.pi / 180.0
        cos_a  = math.cos(angle)
        sin_a  = math.sin(angle)
        seq    = seq.clone().view(seq.shape[0], -1, 3)  # (T, 42, 3)
        x      = seq[:, :, 0].clone()
        y      = seq[:, :, 1].clone()
        seq[:, :, 0] = cos_a * x - sin_a * y
        seq[:, :, 1] = sin_a * x + cos_a * y
        return seq.view(seq.shape[0], -1)

    def _augment(self, seq):
        """
        seq: (T, FEATURE_DIM) tensor.  Returns tensor with a possibly different T.
        """
        T = seq.shape[0]

        # Joint noise — small random jitter so the model doesn't memorise
        # exact landmark positions from the training videos
        if torch.rand(1).item() < 0.5:
            seq = seq + torch.randn_like(seq) * 0.02

        # 2D rotation — simulates slight camera-angle differences between signers.
        # After wrist-centering the landmarks live around the origin, so rotating
        # around (0,0) is the right thing to do.
        if torch.rand(1).item() < 0.5:
            angle = (torch.rand(1).item() - 0.5) * 30.0  # uniform [-15, +15] degrees
            seq = self._rotate_2d(seq, angle)

        # Speed perturbation — resample the sequence at 0.8x–1.2x speed.
        # This teaches the model that the same sign at different speeds is
        # still the same sign.
        if torch.rand(1).item() < 0.5:
            speed  = 0.8 + torch.rand(1).item() * 0.4  # uniform [0.8, 1.2]
            new_T  = max(1, int(T * speed))
            indices = torch.linspace(0, T - 1, new_T).long().clamp(max=T - 1)
            seq = seq[indices]

        # Random frame drop — remove a few frames so the model can't rely on
        # every single frame being present (mimics real-world dropped detections)
        if torch.rand(1).item() < 0.3:
            T_now  = seq.shape[0]
            n_drop = torch.randint(1, max(2, T_now // 8), (1,)).item()
            keep   = torch.ones(T_now, dtype=torch.bool)
            keep[torch.randperm(T_now)[:n_drop]] = False
            seq = seq[keep]

        return seq

    # -------------------------
    # Velocity features
    # -------------------------
    def _add_velocity(self, seq):
        """Append frame-to-frame deltas.  (T, 126) -> (T, 252).
        Motion patterns generalize across signers much better than raw positions."""
        velocity = torch.zeros_like(seq)
        velocity[1:] = seq[1:] - seq[:-1]   # first frame has zero velocity
        return torch.cat([seq, velocity], dim=1)

    # -------------------------
    # Padding / truncation  ->  fixed SEQ_LEN
    # -------------------------
    def _pad_or_truncate(self, seq):
        """Returns (seq, mask) both of length SEQ_LEN."""
        T = seq.shape[0]

        if T > SEQ_LEN:
            # Random crop (temporal jitter during training)
            start = torch.randint(0, T - SEQ_LEN + 1, (1,)).item()
            seq   = seq[start:start + SEQ_LEN]
            mask  = torch.ones(SEQ_LEN)
        else:
            pad  = SEQ_LEN - T
            seq  = torch.cat([seq, torch.zeros(pad, seq.shape[1])])
            mask = torch.cat([torch.ones(T), torch.zeros(pad)])

        return seq, mask

    # -------------------------
    # Public helper for inference (no augmentation)
    # -------------------------
    def _process_sequence(self, seq):
        """Normalize + velocity + pad.  Used by infer.py / webcam.py at inference time."""
        seq = self._normalize(seq)
        seq = self._add_velocity(seq)
        return self._pad_or_truncate(seq)

    # -------------------------
    # Dataset access
    # -------------------------
    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.files[idx])
        raw  = np.load(path)

        seq = self._normalize(raw)

        if self.training:
            seq = self._augment(seq)

        seq  = self._add_velocity(seq)
        seq, mask = self._pad_or_truncate(seq)
        label = self.label_to_idx[label_from_filename(self.files[idx])]

        return seq, mask, label


def smooth_predictions(prob_buffer):
    """prob_buffer: list of probability tensors"""
    avg_probs = torch.stack(list(prob_buffer)).mean(dim=0)
    conf, pred = torch.max(avg_probs, dim=0)
    return pred.item(), conf.item()
