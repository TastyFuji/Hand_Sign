# ASL Hand Sign Recognition

Real-time American Sign Language (ASL) hand sign recognition using **CNN** + **MediaPipe** + **OpenCV**.

Recognizes **24 static ASL letters** (A–Y, excluding J and Z which require motion).

---

## Techniques & Technologies

### 1. Convolutional Neural Network (CNN)

โมเดลหลักที่ใช้จำแนกตัวอักษร ASL เป็น CNN ที่ออกแบบเองด้วย TensorFlow / Keras

**Architecture:**

| Layer | Detail |
|-------|--------|
| Input | 28 × 28 × 1 (grayscale) |
| Conv Block 1 | Conv2D(32, 3×3, ReLU, padding=same) × 2 → MaxPool2D → Dropout(0.25) |
| Conv Block 2 | Conv2D(64, 3×3, ReLU, padding=same) × 2 → MaxPool2D → Dropout(0.25) |
| Dense | Flatten → Dense(128, ReLU) → Dropout(0.4) → Dense(24, softmax) |

**ทำไมถึงเลือก CNN?**
- CNN เหมาะกับงาน image classification เพราะ convolutional layer สามารถเรียนรู้ spatial features (ขอบ, มุม, รูปทรง) ของรูปมือได้โดยอัตโนมัติ
- ใช้ **2 Conv Blocks** เพื่อเรียนรู้ feature จาก low-level (ขอบ, เส้น) ไปจนถึง high-level (รูปทรงนิ้ว, ท่ามือ)
- **MaxPooling** ลด spatial dimension ทำให้โมเดลเล็กลงและลด overfitting
- **Dropout** (0.25 ใน conv, 0.4 ใน dense) เป็น regularization ป้องกัน overfitting โดยการ random ปิด neuron ระหว่าง training

### 2. MediaPipe Hands

ใช้ **Google MediaPipe** สำหรับ detect และ track มือแบบ real-time

- ตรวจจับ **21 hand landmarks** (ข้อนิ้ว, ปลายนิ้ว, ฝ่ามือ)
- ใช้ bounding box จาก landmarks เพื่อ crop เฉพาะบริเวณมือ (Region of Interest)
- ตั้ง `min_detection_confidence = 0.6` และ `min_tracking_confidence = 0.6`

**ทำไมถึงใช้ MediaPipe?**
- เป็น pre-trained model ที่แม่นยำและเร็วมาก สามารถรันบน CPU ได้ real-time
- ไม่ต้อง train hand detector เอง ประหยัดเวลาและ data

### 3. OpenCV

ใช้ **OpenCV** สำหรับ:
- จับภาพจาก webcam (`VideoCapture`)
- Image preprocessing (grayscale conversion, histogram equalization, resize)
- วาด bounding box และแสดงผลลัพธ์บนหน้าจอ

### 4. Dataset — Sign Language MNIST

- **Training set**: 27,455 ภาพ
- **Test set**: 7,172 ภาพ
- แต่ละภาพเป็น grayscale 28×28 pixel
- มี 24 classes (A–Y ไม่รวม J, Z)
- อยู่ในรูป CSV (แต่ละ row = 1 label + 784 pixel values)

---

## How It Works

### Training Pipeline (`train.py`)

```
CSV Data → Load & Parse → Normalize (÷255) → Reshape (28×28×1) → CNN Training → Save Model (.h5)
```

1. **โหลดข้อมูล** จาก CSV แยก label กับ pixel values
2. **Normalize** หาร 255 ให้ค่า pixel อยู่ในช่วง [0, 1]
3. **Reshape** จาก flat array (784,) เป็น (28, 28, 1)
4. **Train** ด้วย Adam optimizer, learning rate 1e-3, batch size 128, สูงสุด 30 epochs
5. **Callbacks:**
   - `ModelCheckpoint` — บันทึกโมเดลที่ val_accuracy ดีที่สุด
   - `EarlyStopping` (patience=5) — หยุด train ถ้า val_accuracy ไม่ดีขึ้น 5 epochs ติดกัน ป้องกัน overfitting

### Real-time Prediction Pipeline (`predict_webcam.py`)

```
Webcam Frame → MediaPipe Hand Detection → Crop ROI → Preprocess → CNN Predict → Smoothing → Display
```

1. **Capture** ภาพจาก webcam แล้ว flip แนวนอน (mirror)
2. **MediaPipe** ตรวจจับมือ คำนวณ bounding box จาก hand landmarks + padding 20px
3. **Preprocess** ภาพ ROI:
   - BGR → Grayscale
   - Histogram Equalization (ปรับ contrast ให้สม่ำเสมอ)
   - Resize เป็น 28×28 (ใช้ `INTER_AREA` interpolation เหมาะกับการย่อภาพ)
   - Normalize ÷ 255
4. **Predict** ด้วย CNN → ได้ probability ของแต่ละ class
5. **Majority Voting** (sliding window ขนาด 7 frames) — ลดการกระพริบของผลลัพธ์
6. **Confidence Threshold** — ถ้า confidence < 50% แสดง "Unknown"
7. **แสดงผล** — bounding box, hand landmarks, ตัวอักษรที่ทำนาย, confidence %, preview ROI 28×28

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~94% |
| Loss Function | Sparse Categorical Crossentropy |
| Optimizer | Adam (lr=1e-3) |
| Input Size | 28 × 28 grayscale |
| Output Classes | 24 (A–Y, no J/Z) |

---

## Project Structure

```
Hand_Sign/
├── data/                          # Sign MNIST CSV (not in git)
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
├── artifacts/
│   └── asl_cnn.h5                 # Trained CNN model
├── train.py                       # Training script
├── predict_webcam.py              # Real-time webcam prediction
├── ASL_Sign_MNIST_Train_and_Webcam.ipynb  # Jupyter notebook version
├── requirements.txt               # Dependencies
└── README.md
```

---

## Quick Start

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

Model is already included in `artifacts/` — ready to use immediately:

```bash
python predict_webcam.py
```

- Press **Q** to quit
- If using a phone as webcam (DroidCam/Iriun), change `CAMERA_INDEX` in the script

### 3. (Optional) Retrain the Model

Download Sign Language MNIST dataset, place CSVs in `data/`, then:

```bash
python train.py
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.15.x | Deep learning framework สำหรับ build และ train CNN |
| MediaPipe | 0.10.9 | Hand detection และ landmark tracking |
| OpenCV | latest | Image processing และ webcam capture |
| NumPy | <2 | Array operations |
| Pandas | latest | โหลด CSV dataset |
| scikit-learn | latest | Classification report และ evaluation metrics |
| Matplotlib | latest | Plot training history |

---

## FAQ — คำถามที่อาจารย์อาจถาม

### Q: ทำไมเลือกใช้ CNN ไม่ใช้โมเดลอื่น?
CNN เหมาะกับ image classification เพราะ convolutional layer สามารถเรียนรู้ spatial hierarchy ของ features ได้ (ขอบ → รูปทรง → ท่ามือ) โดยไม่ต้อง hand-craft features เอง ซึ่งต่างจาก traditional ML เช่น SVM หรือ Random Forest ที่ต้อง extract features ก่อน

### Q: ทำไมไม่ใช้ Transfer Learning (เช่น ResNet, VGG)?
เพราะ input เป็นภาพ grayscale 28×28 ซึ่งเล็กมาก pre-trained model อย่าง ResNet ออกแบบมาสำหรับภาพ 224×224 RGB การ resize ขึ้นไปจะทำให้ภาพเบลอ และ model ใหญ่เกินไปสำหรับ task นี้ CNN เล็กๆ ที่ออกแบบเองทำได้ ~94% accuracy แล้ว

### Q: Dropout ทำงานยังไง ทำไมถึงใช้?
Dropout จะ random ปิด neuron บางตัว (เช่น 25% หรือ 40%) ในแต่ละ training step ทำให้โมเดลไม่พึ่ง neuron ใด neuron หนึ่งมากเกินไป เป็นวิธี regularization ที่ช่วยลด overfitting ทำให้โมเดล generalize ได้ดีขึ้นกับข้อมูลที่ไม่เคยเห็น

### Q: EarlyStopping คืออะไร ช่วยอะไร?
EarlyStopping จะหยุด training อัตโนมัติเมื่อ validation accuracy ไม่ดีขึ้นเป็นเวลา 5 epochs ติดต่อกัน (patience=5) แล้ว restore weights กลับไปที่จุดที่ดีที่สุด ช่วยป้องกัน overfitting และประหยัดเวลา train

### Q: MediaPipe ทำหน้าที่อะไร ทำไมไม่ detect มือเอง?
MediaPipe เป็น pre-trained hand detection model ของ Google ที่ตรวจจับมือและ 21 จุด landmark ได้แม่นยำมากแบบ real-time บน CPU ถ้า train hand detector เองจะต้องใช้ dataset ขนาดใหญ่และเวลามาก MediaPipe ช่วยให้โฟกัสที่ classification task ได้เลย

### Q: Histogram Equalization ทำอะไร?
ปรับการกระจายตัวของ pixel intensity ให้สม่ำเสมอ ทำให้ภาพมี contrast ดีขึ้น ช่วยให้โมเดลทำงานได้ดีในสภาพแสงที่แตกต่างกัน (ห้องสว่าง/มืด)

### Q: Majority Voting ทำงานยังไง?
เก็บผลทำนาย 7 frames ล่าสุดไว้ใน sliding window แล้วเลือก class ที่ถูกทำนายบ่อยที่สุด (mode) ช่วยลดการกระพริบ (flickering) ของผลลัพธ์ให้ output นิ่งขึ้น

### Q: ทำไม J กับ Z ไม่มี?
J และ Z เป็นตัวอักษรที่ต้องขยับมือ (motion gestures) ไม่สามารถจำแนกจากภาพนิ่งเพียง frame เดียวได้ ต้องใช้เทคนิคอื่น เช่น sequence model (LSTM, Transformer) หรือ video classification

### Q: ถ้าจะเพิ่ม accuracy ทำยังไง?
- เพิ่ม **Data Augmentation** (rotation, shift, zoom, brightness) เพื่อให้โมเดลเห็นข้อมูลหลากหลายขึ้น
- ใช้ **Batch Normalization** เพื่อ stabilize training
- เพิ่ม Conv Block เป็น 3 ชั้น
- ใช้ **Learning Rate Scheduler** ลด lr ระหว่าง training
- ใช้ข้อมูลจริงจาก webcam มา fine-tune เพิ่ม

### Q: โมเดลนี้ deploy ได้ไหม?
ได้ สามารถ convert เป็น TensorFlow Lite สำหรับมือถือ หรือ TensorFlow.js สำหรับ web browser หรือ wrap เป็น REST API ด้วย Flask/FastAPI
