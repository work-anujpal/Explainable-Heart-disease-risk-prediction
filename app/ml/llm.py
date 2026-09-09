from __future__ import annotations

import os
from typing import Dict, List

import requests


def template_explanation(prediction: str, probability: float, shap_items: List[Dict]) -> str:
    top = shap_items[:5]
    increasing = [x["feature"] for x in top if x["contribution"] > 0]
    decreasing = [x["feature"] for x in top if x["contribution"] < 0]

    parts = [
        f"The model classified this input as {prediction} with a predicted probability of {probability:.1%}."
    ]
    if increasing:
        parts.append(
            "Among the strongest model factors increasing the prediction were "
            + ", ".join(increasing) + "."
        )
    if decreasing:
        parts.append(
            "Factors that pushed the model toward lower predicted risk included "
            + ", ".join(decreasing) + "."
        )
    parts.append(
        "These are model explanations, not medical conclusions. A healthcare professional should interpret real clinical concerns."
    )
    return " ".join(parts)


def generate_plain_language(prediction: str, probability: float, shap_items: List[Dict]) -> str:
    fallback = template_explanation(prediction, probability, shap_items)

    if os.getenv("ENABLE_LLM", "false").lower() != "true":
        return fallback

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "phi4-mini")

    top_lines = "\n".join(
        f"- {x['feature']}: value={x['value']}, contribution={x['contribution']:+.4f}"
        for x in shap_items[:6]
    )
    prompt = f"""
You are explaining a machine-learning output for an educational heart-disease risk demo.
Do not diagnose disease, prescribe treatment, or invent clinical facts.
Prediction: {prediction}
Probability: {probability:.4f}
Top SHAP contributions:
{top_lines}

Write one short, plain-language paragraph explaining which model inputs pushed the prediction up or down.
End with a sentence saying this is an educational model output, not a medical diagnosis.
""".strip()

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=20,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        return text or fallback
    except Exception:
        return fallback
