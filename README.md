# SilentVoice

Real-time American Sign Language (ASL) word recognition using hand landmarks and deep learning.

**Vocabulary:** 110 words (see [vocabulary.txt](vocabulary.txt))

## Quick Start

### 1. Setup
```bash
# Create virtual environment (Python 3.11 recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run
```bash
python webcam.py
```
Press `Q` to quit.

## Personalization 

The base model is trained on public datasets. Due to the signer gap problem, the Model requires the user's calibration for 5 counts up to 5 words. 

```bash
# 1. Record calibration samples (5 words, 5 recordings each)
python record_calibration.py

# 2. Fine-tune the model on your recordings
python calibrate.py

# 3. Run webcam (automatically uses calibrated model)
python webcam.py
```

## Project Structure

```
SilentVoice/
├── webcam.py              # Real-time inference
├── record_calibration.py  # Record personal samples
├── calibrate.py           # Fine-tune on calibration data
├── train.py               # Train model from scratch
├── model.py               # LSTM architecture
├── asl_dataset.py         # Data loading & processing
├── download_vids.py       # Download MS-ASL videos
├── vocabulary.txt         # 110 supported words
│
├── asl_merged.pt          # Pre-trained model (110 words)
├── asl_calibrated.pt      # Your personalized model (gitignored)
│
└── data/
    ├── landmarks_merged/  # Training landmarks (MS-ASL + WLASL100)
    ├── calibration/       # Your calibration recordings
    └── videos/            # Downloaded videos (gitignored)
```

## Training From Scratch

If you want to retrain the model:

```bash
# Train on merged dataset (default, ~3000 samples, 110 words)
python train.py

# Or train on WLASL100 only (~2000 samples, 100 words)
python train.py --dataset wlasl
```

### Expanding the Dataset

```bash
# 1. Install ffmpeg
winget install ffmpeg       # Windows
brew install ffmpeg         # Mac

# 2. Download videos from MS-ASL
python download_vids.py

# 3. Extract landmarks (in conversions/ folder)
python conversions/extract_coordinates.py

# 4. Train
python train.py
```

## How It Works

### Pipeline Overview

```
Webcam Frame → MediaPipe Hands → Landmarks → Preprocessing → Model → Prediction
```

### 1. Hand Landmark Extraction

**MediaPipe Hands** detects and tracks 21 landmarks per hand in real-time:

```
        8   12  16  20
        |   |   |   |
    4   7   11  15  19
    |   |   |   |   |
    3   6   10  14  18
    |   |   |   |   |
    2   5   9   13  17
     \   \  |  /   /
       \  \ | /  /
         \  0  /  ← Wrist
```

Each landmark has (x, y) coordinates normalized to [0, 1]. We track both hands, giving us **42 landmarks × 2 coordinates = 84 features per frame**.

### 2. Feature Engineering

Raw landmarks aren't enough — the model needs to understand *motion*. We compute:

- **Position features**: Normalized (x, y) for each landmark (z is set to 0)
- **Velocity features**: Change between consecutive frames (Δx, Δy)

This doubles our features: **84 positions + 84 velocities = 168 features** (stored as 252 with padding for both hands).

Velocity features help the model generalize across signers because they capture *how* you move, not just *where* your hands are.

### 3. Model Architecture

We use a **BiLSTM with Attention** — designed for sequential data where context matters:

```
Input (64 frames × 252 features)
          ↓
┌─────────────────────────┐
│  Temporal Convolution   │  Extract local motion patterns
│  (Conv1d: 252 → 128)    │  Kernel size 5 captures ~150ms
└─────────────────────────┘
          ↓
┌─────────────────────────┐
│  Bidirectional LSTM     │  Process sequence forward & backward
│  (2 layers, 192 hidden) │  Captures long-range dependencies
└─────────────────────────┘
          ↓
┌─────────────────────────┐
│  Soft Attention         │  Focus on important frames
│  (learned weights)      │  Ignores idle/transition frames
└─────────────────────────┘
          ↓
┌─────────────────────────┐
│  Classifier (FC)        │  Output: 110 word probabilities
└─────────────────────────┘
```

**Why this architecture?**
- **Conv1d** captures short-term motion patterns (finger wiggling, hand shapes)
- **BiLSTM** understands the sequence both forwards and backwards (signs have beginnings and endings)
- **Attention** lets the model focus on the "important" frames and ignore transitions

### 4. Real-time Inference

During webcam inference, we use several techniques for stability:

| Technique | Purpose |
|-----------|---------|
| **64-frame buffer** | Accumulates ~2 seconds of landmarks before predicting |
| **Temporal smoothing** | Averages last 10 predictions to reduce flickering |
| **Confidence threshold** | Only shows predictions above 30% confidence |
| **Lock-in mechanism** | New word must win 5 consecutive rounds to replace current |

### 5. Calibration (Fine-tuning)

When you calibrate, we:
1. **Freeze** the Conv1d layer (low-level motion patterns are universal)
2. **Unfreeze** LSTM + Attention + FC layers (~90% of model)
3. **Fine-tune** on your recordings for 50 epochs

This teaches the model *your* hand proportions and signing style while preserving the general knowledge of ASL signs.

## Requirements

- Python 3.11
- opencv-python
- mediapipe
- torch
- numpy
- yt-dlp (for downloading videos)

## Citations

### Datasets
- **MS-ASL**: Joze, H. R. V., & Koller, O. (2019). MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language.
- **WLASL**: Li, D., et al. (2020). Word-level Deep Sign Language Recognition from Video.
- **ASL Citizen**: Desai, A., Berger, L., Minakov, F., Milano, N., Singh, C., Pumphrey, K., ... & Bragg, D. (2023). ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition. *Advances in Neural Information Processing Systems, 36*, 76893-76907.
- **ASL Alphabet**: Akash Nagaraj. (2018). ASL Alphabet [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/29550

### Papers & References
- Gangal, A., Kuppahally, A., & Ravindran, M. (n.d.). Sign Language Recognition with Convolutional Neural Networks. https://cs231n.stanford.edu/2024/papers/sign-language-recognition-with-convolutional-neural-networks.pdf
- Chavan, S., Yu, X., & Saniie, J. (2021). Convolutional Neural Network Hand Gesture Recognition for American Sign Language. https://doi.org/10.1109/EIT51626.2021.9491897
- http://www.lrec-conf.org/proceedings/lrec2022/workshops/sltat/pdf/2022.sltat-1.7.pdf
- https://github.com/cristinalunaj/InterpretableTransformer_SignLanguage