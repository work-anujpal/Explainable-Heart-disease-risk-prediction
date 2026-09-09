"""
Download the UCI Heart Disease dataset and convert it to the 13-feature schema
used by this project.

This helper is optional. You may instead place a compatible data/heart.csv
manually.
"""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

OUT = Path(__file__).resolve().parents[1] / "data" / "heart.csv"


def main():
    ds = fetch_ucirepo(id=45)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise RuntimeError(
            "The fetched UCI dataset schema did not contain expected columns: "
            + ", ".join(missing)
        )

    frame = X[FEATURES].copy()
    target_col = y.columns[0]
    frame["target"] = pd.to_numeric(y[target_col], errors="coerce").fillna(0)
    frame["target"] = (frame["target"] > 0).astype(int)

    # Convert '?' and non-numeric values to NaN.
    for col in FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Drop rows where target is unavailable; feature missing values are handled
    # during training by median imputation.
    frame = frame.dropna(subset=["target"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"Saved {len(frame)} rows to {OUT}")
    print(frame.head())


if __name__ == "__main__":
    main()
