import json
import os
import subprocess

INPUT_JSON = os.path.join("data", "msasl_small.json")
OUTPUT_DIR = os.path.join("data", "videos")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_JSON, "r") as f:
    samples = json.load(f)

counter = {}

for sample in samples:
    word = sample["text"].replace(" ", "_")
    url = sample["url"]
    start = sample["start_time"]
    end = sample["end_time"]
    duration = end - start

    counter[word] = counter.get(word, 0) + 1
    idx = counter[word] - 1

    out_file = os.path.join(OUTPUT_DIR, f"{word}_{idx:03d}.mp4")

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
    except subprocess.CalledProcessError:
        print(f"Failed: {url}")
        with open("failed_videos.txt", "a") as log:
            log.write(f"{url}\n")