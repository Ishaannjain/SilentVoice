import json
import os
import subprocess
from collections import Counter, defaultdict

# All three MS-ASL splits — together they cover 25 513 clips across 1 000 words.
# Using only train.json leaves val+test (9 459 extra clips) on the table.
INPUT_JSONS = [
    os.path.join("data",   "MSASL_train.json"),   # 16 054 entries
    os.path.join("MS-ASL", "MSASL_val.json"),     #  5 287 entries
    os.path.join("MS-ASL", "MSASL_test.json"),    #  4 172 entries
]
OUTPUT_DIR = os.path.join("data", "videos")

MAX_PER_WORD = 50         # 50 clips/word gives solid training signal
MAX_WORDS    = 300        # target: 300 words (matches top-300 by frequency)

FAILED_LOG = os.path.join("data", "failed_video_urls.txt")
WORDLIST_FILE = os.path.join("data", "selected_words.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAILED_LOG), exist_ok=True)

# -------------------------
# Load metadata (all splits)
# -------------------------
samples = []
for json_path in INPUT_JSONS:
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            samples.extend(json.load(f))
        print(f"Loaded {json_path}")
    else:
        print(f"WARNING: {json_path} not found, skipping")
print(f"Total entries across all splits: {len(samples)}")

# -------------------------
# Load failed URL cache
# -------------------------
failed_urls = set()
if os.path.exists(FAILED_LOG):
    with open(FAILED_LOG, "r", encoding="utf-8") as f:
        failed_urls = set(line.strip() for line in f if line.strip())

def log_failed(url: str):
    if url in failed_urls:
        return
    failed_urls.add(url)
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(url + "\n")

# -------------------------
# Resume: count existing mp4s
# -------------------------
counter = defaultdict(int)
selected_words = set()

for fname in os.listdir(OUTPUT_DIR):
    if fname.endswith(".mp4") and "_" in fname:
        word = "_".join(fname.split("_")[:-1])
        counter[word] += 1
        selected_words.add(word)

def save_wordlist():
    with open(WORDLIST_FILE, "w", encoding="utf-8") as f:
        for w in sorted(selected_words):
            f.write(w + "\n")

# -------------------------
# Lock word list across runs (prevents drift)
# -------------------------
if os.path.exists(WORDLIST_FILE):
    with open(WORDLIST_FILE, "r", encoding="utf-8") as f:
        selected_words = set(line.strip() for line in f if line.strip())
else:
    # First run: pick the MAX_WORDS most-frequent words across all splits.
    # This ensures we target words with the best chance of yielding enough clips.
    freq = Counter(s["text"].replace(" ", "_") for s in samples)
    selected_words = {w for w, _ in freq.most_common(MAX_WORDS)}
    save_wordlist()
    print(f"Pre-selected top {len(selected_words)} words by frequency")

# -------------------------
# Helper: should we add new words?
# -------------------------
def can_add_new_word(word: str) -> bool:
    return (word not in selected_words) and (len(selected_words) < MAX_WORDS)

# -------------------------
# Download loop (fill words to MAX_PER_WORD)
# -------------------------
for sample in samples:
    word = sample["text"].replace(" ", "_")

    # If we already locked a word list, ignore words not in it
    if os.path.exists(WORDLIST_FILE):
        if word not in selected_words:
            continue
    else:
        # First run: grow word list up to MAX_WORDS
        if can_add_new_word(word):
            selected_words.add(word)
            save_wordlist()
        if word not in selected_words:
            continue

    # Stop once word is full
    if counter[word] >= MAX_PER_WORD:
        continue

    url = sample["url"]
    if url in failed_urls:
        continue

    start = sample["start_time"]
    end = sample["end_time"]
    duration = end - start
    if duration <= 0:
        log_failed(url)
        continue

    idx = counter[word]
    out_file = os.path.join(OUTPUT_DIR, f"{word}_{idx:03d}.mp4")

    # Skip already-downloaded
    if os.path.exists(out_file):
        counter[word] += 1
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
        counter[word] += 1
    except subprocess.CalledProcessError:
        print(f"Failed (private/unlisted/blocked): {url}")
        log_failed(url)
        # do NOT increment counter; we will keep trying other clips for this word

# -------------------------
# Print summary
# -------------------------
print("\n=== DOWNLOAD SUMMARY ===")
for w in sorted(selected_words):
    print(f"{w:20s} {counter[w]}/{MAX_PER_WORD}")
print(f"\nSaved wordlist: {WORDLIST_FILE}")
print(f"Failed URLs logged to: {FAILED_LOG}")
