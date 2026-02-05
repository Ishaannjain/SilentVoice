# quick check — paste in python
import os
from collections import Counter
from asl_dataset import label_from_filename
files = [f for f in os.listdir("data/landmarks") if f.endswith(".npy")]
counts = Counter(label_from_filename(f) for f in files)
print(f"Words with >= 3 landmarks: {sum(1 for c in counts.values() if c >= 3)}")
print(f"Words with >= 5 landmarks: {sum(1 for c in counts.values() if c >= 5)}")
print(f"Total landmarks: {len(files)}")
