## To run the project

### Step 1 — Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

### Python 3.11 is recommended
python3.11 -m venv venv
source venv/bin/activate

### install dependencies 
pip install requirements.txt

### Project Structure

SilentVoice/
│
├── asl_dataset.py          # Dataset + padding logic
├── extract_landmarks.py    # Converts videos → landmark arrays
├── train_lstm.py           # Trains word-level LSTM
├── main.py                # runs camera with the trained .npt files
├── webcam-test.py         # Real-time webcam inference test
├── asl_lstm_test.pt        # Trained model (small demo model for now)
│
└── data/
    ├── videos/             # raw ASL videos (add to .gitignore)
    └── landmarks/          # Extracted hand landmarks (.npy)



# For expanding dataset and training model further 

## Install Required Tools

### install ffmpeg for downloading training videos
winget install ffmpeg

### choose the videos to download in (data/msasl_small.json)

### download the videos
python download_videos.py

### run the AI model on the updated dataset
python train_lstm (preliminary).py

