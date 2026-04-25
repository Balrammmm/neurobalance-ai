"""
generate_data.py — Synthetic Dataset Generator for Stress Detection
=====================================================================
Generates a CSV dataset with blink_rate, bpm, and stress columns.

Realistic ranges used (based on human physiology):
┌────────────┬──────────────────────┬───────────────────┐
│ Stress     │ Blink Rate (/min)    │ Heart Rate (BPM)  │
├────────────┼──────────────────────┼───────────────────┤
│ Low        │ 10 – 18              │ 60 – 80           │
│ Medium     │ 18 – 28              │ 80 – 100          │
│ High       │ 28 – 40              │ 100 – 130         │
└────────────┴──────────────────────┴───────────────────┘

Usage:
    python generate_data.py          # generates data.csv with 150 rows
    python generate_data.py 300      # generates data.csv with 300 rows
"""

import csv
import random
import sys
import os

# ---------------------------------------------------------------------------
# Configuration — feel free to change these ranges
# ---------------------------------------------------------------------------
STRESS_PROFILES = {
    "Low": {
        "blink_rate": (10, 18),   # blinks per minute (relaxed)
        "bpm": (60, 80),          # resting heart rate
    },
    "Medium": {
        "blink_rate": (18, 28),   # slightly elevated blinking
        "bpm": (80, 100),         # moderate heart rate
    },
    "High": {
        "blink_rate": (28, 40),   # rapid blinking under stress
        "bpm": (100, 130),        # elevated heart rate
    },
}

OUTPUT_FILE = "data.csv"


def generate_row(stress_level: str) -> dict:
    """Generate one data row for the given stress level."""
    profile = STRESS_PROFILES[stress_level]

    # Use Gaussian distribution centred in the range for natural variation
    blink_lo, blink_hi = profile["blink_rate"]
    blink_mean = (blink_lo + blink_hi) / 2
    blink_std = (blink_hi - blink_lo) / 4  # ~95% within range
    blink_rate = round(random.gauss(blink_mean, blink_std), 1)
    blink_rate = max(blink_lo - 2, min(blink_hi + 2, blink_rate))  # small overflow OK

    bpm_lo, bpm_hi = profile["bpm"]
    bpm_mean = (bpm_lo + bpm_hi) / 2
    bpm_std = (bpm_hi - bpm_lo) / 4
    bpm = round(random.gauss(bpm_mean, bpm_std), 1)
    bpm = max(bpm_lo - 3, min(bpm_hi + 3, bpm))

    return {"blink_rate": blink_rate, "bpm": bpm, "stress": stress_level}


def generate_dataset(total_rows: int = 150) -> list:
    """Generate a balanced dataset across all stress levels."""
    rows_per_class = total_rows // 3
    remainder = total_rows % 3
    data = []

    for i, level in enumerate(STRESS_PROFILES):
        count = rows_per_class + (1 if i < remainder else 0)
        for _ in range(count):
            data.append(generate_row(level))

    random.shuffle(data)
    return data


def save_csv(data: list, filepath: str):
    """Save a list of dicts to a CSV file."""
    fieldnames = ["blink_rate", "bpm", "stress"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"[OK] Saved {len(data)} rows to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    num_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    dataset = generate_dataset(num_rows)
    save_csv(dataset, OUTPUT_FILE)
    print(f"\nSample rows:")
    for row in dataset[:5]:
        print(f"  blink_rate={row['blink_rate']}, bpm={row['bpm']}, stress={row['stress']}")
