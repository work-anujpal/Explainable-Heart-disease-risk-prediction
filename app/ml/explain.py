from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from app.ml.preprocess import FEATURES


def _positive_class_shap_values(model, scaled_row: np.ndarray) -> np.ndarray:
    """Return one SHAP contribution per original feature for the positive class."""
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(scaled_row)

    if isinstance(raw, list):
        values = np.asarray(raw[-1])[0]
    else:
        arr = np.asarray(raw)
        # Common shapes:
        # (n_samples, n_features)
        # (n_samples, n_features, n_classes)
        if arr.ndim == 3:
            values = arr[0, :, -1]
        elif arr.ndim == 2:
            values = arr[0]
        else:
            values = arr.reshape(-1)

    return np.asarray(values, dtype=float)


def shap_explanation(model, scaler, input_df: pd.DataFrame) -> List[Dict]:
    scaled = scaler.transform(input_df[FEATURES])
    values = _positive_class_shap_values(model, scaled)

    result = []
    for feature, raw_value, shap_value in zip(FEATURES, input_df.iloc[0][FEATURES], values):
        result.append(
            {
                "feature": feature,
                "value": float(raw_value),
                "contribution": float(shap_value),
                "direction": "increases risk" if shap_value > 0 else "decreases risk",
            }
        )

    result.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return result


def lime_explanation(model, scaler, input_df: pd.DataFrame, background_df: pd.DataFrame) -> List[Dict]:
    background = background_df[FEATURES].astype(float).to_numpy()

    def predict_fn(raw_rows: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(raw_rows, columns=FEATURES)
        scaled = scaler.transform(frame)
        return model.predict_proba(scaled)

    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=FEATURES,
        class_names=["Low Risk", "High Risk"],
        mode="classification",
        discretize_continuous=False,
        random_state=42,
    )

    explanation = explainer.explain_instance(
        input_df.iloc[0][FEATURES].astype(float).to_numpy(),
        predict_fn,
        num_features=len(FEATURES),
        labels=(1,),
    )

    return [
        {"rule": rule, "weight": float(weight)}
        for rule, weight in explanation.as_list(label=1)
    ]
