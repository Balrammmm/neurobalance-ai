# NeuroBalance AI — Stress Detection Pipeline

A **beginner-friendly** machine learning pipeline that detects stress levels (**Low / Medium / High**) using two simple inputs:

| Input | Source |
|-------|--------|
| **Blink rate** (blinks/min) | Webcam (via OpenCV + MediaPipe) |
| **Heart rate** (BPM) | Pulse sensor / smartwatch / manual entry |

---

## Project Structure

```
neurobalance-ai/
├── data.csv            # Generated dataset
├── generate_data.py    # Synthetic data generator
├── train_model.py      # Model training script
├── predict.py          # Prediction script
├── model.pkl           # Trained model (created after training)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python generate_data.py        # 150 rows (default)
python generate_data.py 300    # or specify a custom count
```

This creates `data.csv` with columns: `blink_rate`, `bpm`, `stress`.

### 3. Train the model

```bash
python train_model.py
```

Prints accuracy, classification report, and saves `model.pkl`.

### 4. Predict stress level

```bash
# Command-line mode
python predict.py 25 95

# Interactive mode
python predict.py
```

---

## Realistic Ranges Used

| Stress Level | Blink Rate (/min) | Heart Rate (BPM) |
|:------------:|:------------------:|:-----------------:|
| **Low**      | 10 – 18            | 60 – 80           |
| **Medium**   | 18 – 28            | 80 – 100          |
| **High**     | 28 – 40            | 100 – 130         |

---

## How to Collect Real Blink Data (Webcam)

You can measure blink rate from a webcam using **OpenCV + MediaPipe**:

```python
# Pseudocode for blink detection
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

blink_count = 0
start_time = time.time()

with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Calculate Eye Aspect Ratio (EAR) from landmarks
        # If EAR drops below threshold → blink detected
        # Track blinks over 60 seconds → blink_rate

elapsed = time.time() - start_time
blink_rate = blink_count / (elapsed / 60)
```

### Combining real data with synthetic BPM

Once you have a real `blink_rate`, you can:
1. Pair it with a BPM from a pulse sensor or smartwatch
2. Append it to `data.csv`:

```python
import csv

new_row = {"blink_rate": 22.5, "bpm": 88, "stress": "Medium"}
with open("data.csv", "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["blink_rate", "bpm", "stress"])
    writer.writerow(new_row)
```

---

## How to Improve the Model

1. **Collect real data** — Replace synthetic rows with actual blink + BPM readings
2. **Add more features** — e.g., blink duration, HRV (heart rate variability)
3. **Try other models** — `LogisticRegression`, `SVM`, `GradientBoosting`
4. **Tune hyperparameters** — Adjust `n_estimators`, `max_depth` in `train_model.py`
5. **Cross-validate** — Use `cross_val_score` for more robust evaluation

---

## Tech Stack

- **Python 3.8+**
- **pandas** — data handling
- **scikit-learn** — machine learning
- **numpy** — numerical operations
- **joblib** — model serialization

---

## License

This project is for educational purposes. Feel free to modify and extend it.
