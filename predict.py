"""
predict.py — Predict Stress Level from Blink Rate & Heart Rate
================================================================
Loads the trained model.pkl and predicts stress for given inputs.

Usage (interactive):
    python predict.py

Usage (command-line):
    python predict.py 25 95
    → blink_rate=25, bpm=95 → predicted stress level
"""

import joblib
import numpy as np
import sys
import os

MODEL_FILE = "model.pkl"


def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] {MODEL_FILE} not found. Run 'python train_model.py' first.")
        exit(1)
    return joblib.load(MODEL_FILE)


def predict_stress(model, blink_rate: float, bpm: float) -> str:
    """Predict stress level for given blink_rate and bpm."""
    features = np.array([[blink_rate, bpm]])
    prediction = model.predict(features)[0]
    return prediction


def main():
    model = load_model()

    # If command-line arguments are provided, use them
    if len(sys.argv) == 3:
        blink_rate = float(sys.argv[1])
        bpm = float(sys.argv[2])
        result = predict_stress(model, blink_rate, bpm)
        print(f"\n[INPUT]  blink_rate = {blink_rate}, bpm = {bpm}")
        print(f"[PREDICT] Stress Level = {result}")
        return

    # Interactive mode
    print("=" * 50)
    print("  Stress Level Predictor")
    print("=" * 50)
    print("Enter 'q' to quit.\n")

    while True:
        try:
            blink_input = input("Enter blink rate (per minute): ").strip()
            if blink_input.lower() == "q":
                print("Goodbye!")
                break

            bpm_input = input("Enter heart rate (BPM): ").strip()
            if bpm_input.lower() == "q":
                print("Goodbye!")
                break

            blink_rate = float(blink_input)
            bpm = float(bpm_input)

            result = predict_stress(model, blink_rate, bpm)

            print(f"\n[PREDICT] Stress Level: {result}")
            print(f"   (blink_rate={blink_rate}, bpm={bpm})")
            print("-" * 40 + "\n")

        except ValueError:
            print("[WARN] Please enter valid numbers.\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
