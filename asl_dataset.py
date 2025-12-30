import os
import numpy as np
import torch
from torch.utils.data import Dataset

SEQ_LEN = 64


class ASLDataset(Dataset):
    def __init__(self, landmark_dir):
        self.dir = landmark_dir
        self.files = sorted([f for f in os.listdir(landmark_dir) if f.endswith(".npy")])

        self.labels = sorted(list(set(f.split("_")[0] for f in self.files)))
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.files)

    def _normalize(self, seq):
        seq = torch.tensor(seq, dtype=torch.float32)
        seq = seq.view(seq.shape[0], 21, 3)

        # translation invariance
        wrist = seq[:, 0:1, :]
        seq = seq - wrist

        # scale invariance
        scale = torch.norm(seq[:, 9, :], dim=1).mean()
        seq = seq / (scale + 1e-6)

        return seq.view(seq.shape[0], -1)

    def _process_sequence(self, seq):
        seq = self._normalize(seq)
        T = seq.shape[0]

        # temporal jitter
        if T > SEQ_LEN:
            start = torch.randint(0, T - SEQ_LEN + 1, (1,)).item()
            seq = seq[start:start + SEQ_LEN]
            mask = torch.ones(SEQ_LEN)
        else:
            pad = SEQ_LEN - T
            seq = torch.cat([seq, torch.zeros(pad, seq.shape[1])])
            mask = torch.cat([torch.ones(T), torch.zeros(pad)])

        return seq, mask

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.files[idx])
        raw = np.load(path)

        seq, mask = self._process_sequence(raw)
        label = self.label_to_idx[self.files[idx].split("_")[0]]

        return seq, mask, label
