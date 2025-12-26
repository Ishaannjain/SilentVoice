import os
import numpy as np
import torch
from torch.utils.data import Dataset

SEQ_LEN = 64

class ASLDataset(Dataset):
    def __init__(self, landmark_dir):
        self.landmark_dir = landmark_dir
        self.files = [f for f in os.listdir(landmark_dir) if f.endswith(".npy")]

        # label = prefix before _
        self.labels = sorted(list(set(f.split("_")[0] for f in self.files)))
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.files)

    def _process_sequence(self, seq):
        T = seq.shape[0]

        if T >= SEQ_LEN:
            # center crop
            start = (T - SEQ_LEN) // 2
            return seq[start:start + SEQ_LEN]

        # pad if too short
        pad_len = SEQ_LEN - T
        pad = np.zeros((pad_len, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])

    def __getitem__(self, idx):
        fname = self.files[idx]
        label_str = fname.split("_")[0]
        label = self.label_to_idx[label_str]

        path = os.path.join(self.landmark_dir, fname)
        seq = np.load(path).astype(np.float32)

        seq = self._process_sequence(seq)

        return (
            torch.from_numpy(seq),              
            torch.tensor(label, dtype=torch.long)
        )
