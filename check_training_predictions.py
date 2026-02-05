from collections import Counter
from asl_dataset import ASLDataset

dataset = ASLDataset("data/landmarks")
true_counts = Counter()

for _, _, y in dataset:
    true_counts[dataset.labels[y]] += 1

print("\n=== TRUE LABEL COUNTS IN DATASET ===")
for lab in dataset.labels:
    print(f"{lab:12s}: {true_counts[lab]}")
print("Total:", len(dataset))
