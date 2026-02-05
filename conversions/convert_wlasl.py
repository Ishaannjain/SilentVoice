"""
Convert WLASL100 pre-extracted landmarks (CSV) to our .npy format.

CSV format:
- RHx0,RHy0,...,RHx20,RHy20 (right hand 21 landmarks x 2 coords)
- LHx0,LHy0,...,LHx20,LHy20 (left hand 21 landmarks x 2 coords)
- labels: array of class indices (0-99) per frame
- Each cell is a string array "[val1, val2, ...]" for all frames

Our format:
- .npy file per video, shape (num_frames, 126)
- 126 = right_hand(63) + left_hand(63)
- 63 = 21 landmarks x 3 coords (x, y, z)
- z is set to 0 since WLASL100 only has x,y
"""

import os
import json
import numpy as np
import pandas as pd
import ast
from pathlib import Path
from collections import Counter

# Paths
WLASL_DIR = Path("data/wlasl100_raw/inputs/WLASL100/MediaPipe_data/x-y")
OUTPUT_DIR = Path("data/landmarks_wlasl100")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load class mapping
MAPPING_FILE = Path("data/wlasl100_class_mapping.json")
with open(MAPPING_FILE) as f:
    CLASS_TO_WORD = {int(k): v for k, v in json.load(f).items()}

# Column names
RH_COLS = [f"RH{c}{i}" for i in range(21) for c in ["x", "y"]]  # 42 columns
LH_COLS = [f"LH{c}{i}" for i in range(21) for c in ["x", "y"]]  # 42 columns


def parse_array_string(s):
    """Parse a string like '[0., 0.5, ...]' into numpy array."""
    if pd.isna(s) or s == "" or s == "nan":
        return None
    try:
        s = s.replace("\n", "").replace(" ", "").strip()
        arr = np.array(ast.literal_eval(s))
        return arr
    except:
        return None


def convert_row_to_npy(row, rh_cols, lh_cols):
    """
    Convert one CSV row to our (num_frames, 126) format.

    WLASL100 has x,y only. We add z=0.
    Our format: [right_hand(63), left_hand(63)]
    Where each hand is [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
    """
    # Parse right hand columns
    rh_data = {}
    for col in rh_cols:
        arr = parse_array_string(row[col])
        if arr is not None:
            rh_data[col] = arr

    # Parse left hand columns
    lh_data = {}
    for col in lh_cols:
        arr = parse_array_string(row[col])
        if arr is not None:
            lh_data[col] = arr

    if not rh_data and not lh_data:
        return None

    # Determine number of frames from first valid column
    num_frames = None
    for col in rh_cols + lh_cols:
        if col in rh_data:
            num_frames = len(rh_data[col])
            break
        if col in lh_data:
            num_frames = len(lh_data[col])
            break

    if num_frames is None or num_frames == 0:
        return None

    # Build output array: (num_frames, 126)
    output = np.zeros((num_frames, 126), dtype=np.float32)

    # Fill right hand (first 63 features: 21 landmarks x 3 coords)
    for i in range(21):
        x_col = f"RHx{i}"
        y_col = f"RHy{i}"

        if x_col in rh_data:
            output[:, i*3] = rh_data[x_col][:num_frames]  # x
        if y_col in rh_data:
            output[:, i*3 + 1] = rh_data[y_col][:num_frames]  # y
        # z stays 0

    # Fill left hand (next 63 features, starting at index 63)
    for i in range(21):
        x_col = f"LHx{i}"
        y_col = f"LHy{i}"

        if x_col in lh_data:
            output[:, 63 + i*3] = lh_data[x_col][:num_frames]  # x
        if y_col in lh_data:
            output[:, 63 + i*3 + 1] = lh_data[y_col][:num_frames]  # y
        # z stays 0

    return output


def get_label_from_row(row):
    """Extract the class label (word) from the labels column."""
    labels_str = row["labels"]
    label_arr = parse_array_string(labels_str)
    if label_arr is None or len(label_arr) == 0:
        return None

    # Get the class index (should be same for all frames)
    class_idx = int(label_arr[0])

    # Map to word
    if class_idx in CLASS_TO_WORD:
        return CLASS_TO_WORD[class_idx]
    return None


def process_csv(csv_path, split_name):
    """Process a single CSV file and save .npy files."""
    print(f"\nProcessing {csv_path}...")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples")

    # Track label counts for this split
    label_counts = Counter()
    success = 0
    failed = 0

    for idx, row in df.iterrows():
        # Get word label
        label = get_label_from_row(row)
        if label is None:
            failed += 1
            continue

        label_counts[label] += 1

        # Convert to .npy format
        landmarks = convert_row_to_npy(row, RH_COLS, LH_COLS)

        if landmarks is None:
            failed += 1
            continue

        # Generate filename: label_splitXXX.npy
        sample_idx = label_counts[label] - 1
        filename = f"{label}_{split_name}{sample_idx:03d}.npy"
        output_path = OUTPUT_DIR / filename

        np.save(output_path, landmarks)
        success += 1

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")

    print(f"  Done: {success} saved, {failed} failed")
    return label_counts


def main():
    print("=" * 60)
    print("WLASL100 -> SilentVoice Format Conversion")
    print("=" * 60)

    # Process all splits
    all_labels = Counter()

    for split in ["train", "val", "test"]:
        csv_path = WLASL_DIR / f"WLASL100_{split}.csv"
        if csv_path.exists():
            counts = process_csv(csv_path, split)
            all_labels.update(counts)

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)

    total_files = len(list(OUTPUT_DIR.glob("*.npy")))
    print(f"Total .npy files: {total_files}")
    print(f"Unique labels: {len(all_labels)}")

    # Show sample counts per label
    print("\nSamples per word (top 20):")
    for label, count in sorted(all_labels.items(), key=lambda x: -x[1])[:20]:
        print(f"  {label}: {count}")

    if len(all_labels) > 20:
        print(f"  ... and {len(all_labels) - 20} more words")

    # Save word list
    wordlist_path = Path("data/wlasl100_words.txt")
    with open(wordlist_path, "w") as f:
        for label in sorted(all_labels.keys()):
            f.write(f"{label}\n")
    print(f"\nWord list saved to {wordlist_path}")


if __name__ == "__main__":
    main()
