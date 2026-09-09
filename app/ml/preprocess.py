from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

FEATURES: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Broad educational validation ranges. They prevent obviously invalid inputs;
# they are not intended to define clinical diagnostic thresholds.
RANGES = {
    "age": (18, 100),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (70, 250),
    "chol": (80, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (50, 230),
    "exang": (0, 1),
    "oldpeak": (-3.0, 10.0),
    "slope": (0, 2),
    "ca": (0, 4),
    "thal": (0, 3),
}

@dataclass
class ArtifactPaths:
    model: Path
    scaler: Path
    metadata: Path
    background: Path

def artifact_paths() -> ArtifactPaths:
    base = Path(__file__).resolve().parents[1] / "models"
    return ArtifactPaths(
        model=base / "model.pkl",
        scaler=base / "scaler.pkl",
        metadata=base / "metadata.json",
        background=base / "background.joblib",
    )

def validate_payload(payload: Dict[str, float]) -> None:
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    for feature in FEATURES:
        value = payload[feature]
        low, high = RANGES[feature]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError(f"{feature} cannot be empty.")
        if not (low <= float(value) <= high):
            raise ValueError(
                f"{feature} must be between {low} and {high}; received {value}."
            )

def to_frame(payload: Dict[str, float]) -> pd.DataFrame:
    validate_payload(payload)
    return pd.DataFrame([{feature: payload[feature] for feature in FEATURES}])

def load_artifacts():
    paths = artifact_paths()
    missing = [str(p) for p in (paths.model, paths.scaler, paths.metadata, paths.background) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model artifacts are missing. Run `python scripts/train_model.py` first. "
            f"Missing: {', '.join(missing)}"
        )
    model = joblib.load(paths.model)
    scaler = joblib.load(paths.scaler)
    background = joblib.load(paths.background)
    import json
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    return model, scaler, metadata, background
