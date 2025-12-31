import json
import os
import subprocess
from collections import defaultdict

INPUT_JSON = os.path.join("data", "msasl_small.json")
OUTPUT_DIR = os.path.join("data", "videos")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Count already-downloaded videos per word
# --------------------------------------------------
counter = defaultdict(int)

for fname in os.listdir(OUTPUT_DIR):
    if fname.endswith(".mp4") and "_" in fname:
        word = "_".join(fname.split("_")[:-1])
        counter[word] += 1

# --------------------------------------------------
# Load MS-ASL metadata
# --------------------------------------------------
with open(INPUT_JSON, "r") as f:
    samples = json.load(f)

# --------------------------------------------------
# Download loop
# --------------------------------------------------
for sample in samples:
    word = sample["text"].replace(" ", "_")
    url = sample["url"]
    start = sample["start_time"]
    end = sample["end_time"]
    duration = end - start

    idx = counter[word]
    out_file = os.path.join(OUTPUT_DIR, f"{word}_{idx:03d}.mp4")

    #  Skip if this clip already exists
    if os.path.exists(out_file):
        continue

    print(f"Downloading {out_file}")

    cmd = [
        "yt-dlp",
        "--legacy-server-connect",
        "-f", "mp4",
        "--no-check-certificates",
        "--external-downloader", "ffmpeg",
        "--external-downloader-args",
        f"ffmpeg_i:-ss {start} -t {duration}",
        "-o", out_file,
        url
    ]

    try:
        subprocess.run(cmd, check=True)
        counter[word] += 1  # ✅ increment only after success
    except subprocess.CalledProcessError:
        print(f"Failed: {url}")
        with open("failed_videos.txt", "a") as log:
            log.write(f"{url}\n")
