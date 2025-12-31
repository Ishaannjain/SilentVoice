## To run the project

### Step 1 — Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

### Python 3.11 is recommended
python3.11 -m venv venv
source venv/bin/activate

### install dependencies 
pip install requirements.txt

### to run the model
python main.py (to train the model)
python check_training_predictions.py (for analysis)
python webcam.py (to run the model)


### Project Structure

SilentVoice/
│
├── asl_dataset.py                # Dataset loader (normalization, padding, masking)
├── train_lstm.py                 # Trains the word-level ASL BiLSTM model
├── check_training_predictions.py # Verifies which words the model has learned
├── webcam.py                     # Real-time webcam inference (where it runs)
├── webcam-test.py                # Experimental / debugging webcam inference
├── infer.py                      # Offline inference on saved landmark files
├── main.py                       # Entry-point wrapper
│
├── video_landmark_extractor.py   # Robust video → landmark extraction pipeline
├── extract_coordinates.py        # Landmark extraction runner
├── download_vids.py              # Downloads ASL videos from MS-ASL metadata
│
├── asl_lstm_test.pt              # Trained demo model checkpoint
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore
│
├── data/
│   ├── videos/                   # Raw ASL videos (.mp4)  (in gitignore)
│   ├── landmarks/                # Extracted hand landmarks (.npy)
│   ├── fails/
│   │   ├── failed_videos.txt     # Videos that could not be decoded
│   │   └── failed_landmark_extraction.txt
│   └── msasl_small.json          # Subset of MS-ASL used for downloading videos
│
├── MS-ASL/                       # Full MS-ASL metadata/resources
├── venv/                         # Virtual environment (ignored)



# For expanding dataset and training model further 

## Install Required Tools

### install ffmpeg for downloading training videos
winget install ffmpeg

### choose the videos to download in (data/msasl_small.json)

### download the videos
python download_videos.py

### write the model in train_lstm.py and train it in main.py
python main.py

### run the model in webcam.py
python webcam.py


# Citations
Akash Nagaraj. (2018). ASL Alphabet [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/29550\

Desai, A., Berger, L., Minakov, F., Milano, N., Singh, C., Pumphrey, K., ... & Bragg, D. (2023). ASL citizen: a community-sourced dataset for advancing isolated sign language recognition. Advances in Neural Information Processing Systems, 36, 76893-76907.

Gangal, A., Kuppahally, A., & Ravindran, M. (n.d.). Sign Language Recognition with Convolutional Neural Networks. Retrieved from https://cs231n.stanford.edu/2024/papers/sign-language-recognition-with-convolutional-neural-networks.pdf

Chavan, S., Yu, X., & Saniie, J. (2021, May 1). Convolutional Neural Network Hand Gesture Recognition for American Sign Language. https://doi.org/10.1109/EIT51626.2021.9491897

