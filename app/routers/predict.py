from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db.database import save_prediction
from app.ml.explain import lime_explanation, shap_explanation
from app.ml.llm import generate_plain_language
from app.ml.preprocess import FEATURES, load_artifacts, to_frame

router = APIRouter(tags=["prediction"])


class PatientInput(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: int = Field(ge=70, le=250)
    chol: int = Field(ge=80, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=50, le=230)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-3.0, le=10.0)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


@lru_cache(maxsize=1)
def artifacts():
    return load_artifacts()


def _predict(payload: PatientInput):
    model, scaler, metadata, background = artifacts()
    data: Dict = payload.model_dump()
    frame = to_frame(data)
    scaled = scaler.transform(frame[FEATURES])
    probability = float(model.predict_proba(scaled)[0, 1])
    prediction = "High Risk" if probability >= 0.5 else "Low Risk"
    shap_items = shap_explanation(model, scaler, frame)
    explanation = generate_plain_language(prediction, probability, shap_items)
    return data, frame, background, prediction, probability, shap_items, explanation, metadata


@router.post("/predict")
def predict(payload: PatientInput):
    try:
        data, _, _, prediction, probability, shap_items, explanation, metadata = _predict(payload)
        prediction_id = save_prediction(
            inputs=data,
            prediction=prediction,
            probability=probability,
            explanation=explanation,
            shap_items=shap_items,
        )
        return {
            "id": prediction_id,
            "prediction": prediction,
            "probability": probability,
            "confidence_percent": round(probability * 100, 2),
            "explanation": explanation,
            "shap": shap_items,
            "deployed_model": metadata.get("deployed_model"),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@router.post("/explain")
def explain(payload: PatientInput):
    try:
        _, frame, background, prediction, probability, shap_items, explanation, metadata = _predict(payload)
        lime_items = lime_explanation(
            artifacts()[0], artifacts()[1], frame, background
        )
        return {
            "prediction": prediction,
            "probability": probability,
            "shap": shap_items,
            "lime": lime_items,
            "plain_language": explanation,
            "deployed_model": metadata.get("deployed_model"),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc
