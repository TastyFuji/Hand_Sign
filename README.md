# ASL Hand Sign Recognition (Sign Language MNIST)

Real-time American Sign Language (ASL) hand sign recognition using a CNN model + MediaPipe hand detection + OpenCV webcam.

Recognizes 24 ASL letters (A-Y, excluding J and Z which require motion).

## Quick Start (Laptop)

### 1. Setup

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Webcam Prediction

Model is already included in `artifacts/` -- ready to use immediately:

```bash
python predict_webcam.py
```

- Press **Q** to quit
- If using a phone as webcam (DroidCam/Iriun), change `CAMERA_INDEX` in the script

### 3. (Optional) Retrain the Model

If you want to retrain from scratch, download the Sign Language MNIST dataset and place CSVs in `data/`:

```
data/
  sign_mnist_train.csv
  sign_mnist_test.csv
```

Then run:

```bash
python train.py
```

## Project Structure

```
Hand_Sign/
  data/                  # Sign MNIST CSV files (not in git)
  artifacts/
    asl_cnn.h5           # Trained model (included in git)
  train.py               # Training script
  predict_webcam.py      # Real-time webcam prediction
  requirements.txt       # Python dependencies
```

## Requirements

- Python 3.11
- TensorFlow 2.15
- MediaPipe 0.10.9
- OpenCV
