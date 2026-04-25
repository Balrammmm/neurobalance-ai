"""
train_model.py — Train a Stress Detection Model
=================================================
Loads data.csv, trains a Random Forest classifier, prints accuracy,
and saves the model to model.pkl.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = "data.csv"
MODEL_FILE = "model.pkl"
TEST_SIZE = 0.2       # 20% for testing
RANDOM_STATE = 42     # for reproducible results

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("[LOAD] Loading data from", DATA_FILE)
if not os.path.exists(DATA_FILE):
    print(f"[ERROR] {DATA_FILE} not found. Run 'python generate_data.py' first.")
    exit(1)

df = pd.read_csv(DATA_FILE)
print(f"   -> {len(df)} rows loaded")
print(f"   -> Columns: {list(df.columns)}")
print(f"   -> Stress distribution:\n{df['stress'].value_counts().to_string()}\n")

# ---------------------------------------------------------------------------
# 2. Prepare features and labels
# ---------------------------------------------------------------------------
X = df[["blink_rate", "bpm"]]       # features
y = df["stress"]                     # labels (Low / Medium / High)

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[SPLIT] Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

# ---------------------------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------------------------
print("[TRAIN] Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=100,      # 100 trees (good default)
    max_depth=5,           # prevent overfitting on small data
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)
print("   [OK] Training complete\n")

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"[RESULT] Accuracy: {accuracy * 100:.1f}%\n")
print("[REPORT] Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importances
importances = model.feature_importances_
print("[INFO] Feature Importances:")
for name, imp in zip(X.columns, importances):
    print(f"   {name}: {imp:.3f}")

# ---------------------------------------------------------------------------
# 6. Save model
# ---------------------------------------------------------------------------
joblib.dump(model, MODEL_FILE)
print(f"\n[SAVED] Model saved to {MODEL_FILE}")
