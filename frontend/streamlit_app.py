from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="Explainable Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.title("❤️ Explainable Heart Disease Risk Prediction System")
st.warning(
    "Educational decision-support demo only. This application is not a substitute "
    "for professional medical diagnosis or treatment."
)

with st.sidebar:
    st.header("Patient Parameters")
    age = st.number_input("Age", min_value=18, max_value=100, value=55)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=70, max_value=250, value=140)
    chol = st.number_input("Serum Cholesterol", min_value=80, max_value=700, value=250)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate", min_value=50, max_value=230, value=150)
    exang = st.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=-3.0, max_value=10.0, value=1.2, step=0.1)
    slope = st.selectbox("ST Segment Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

payload = {
    "age": int(age),
    "sex": int(sex),
    "cp": int(cp),
    "trestbps": int(trestbps),
    "chol": int(chol),
    "fbs": int(fbs),
    "restecg": int(restecg),
    "thalach": int(thalach),
    "exang": int(exang),
    "oldpeak": float(oldpeak),
    "slope": int(slope),
    "ca": int(ca),
    "thal": int(thal),
}

tab_predict, tab_compare, tab_history, tab_about = st.tabs(
    ["Prediction", "Model Comparison", "History", "About"]
)

with tab_predict:
    st.subheader("Risk Prediction")
    st.write("Enter the 13 clinical parameters in the sidebar, then run the model.")

    if st.button("Predict Risk", type="primary"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict", json=payload, timeout=30
            )
            response.raise_for_status()
            result = response.json()

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", result["prediction"])
            c2.metric("Probability", f"{result['probability']:.1%}")
            c3.metric("Model", result.get("deployed_model", "Unknown"))

            st.subheader("Plain-Language Explanation")
            st.write(result["explanation"])

            shap_df = pd.DataFrame(result["shap"])
            shap_df = shap_df.sort_values("contribution")
            fig = px.bar(
                shap_df,
                x="contribution",
                y="feature",
                orientation="h",
                title="SHAP Feature Contributions",
                labels={"contribution": "SHAP contribution", "feature": "Feature"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top Feature Contributions")
            st.dataframe(
                shap_df[["feature", "value", "contribution", "direction"]],
                use_container_width=True,
                hide_index=True,
            )

            report_text = (
                "Explainable Heart Disease Risk Prediction Report\n"
                "================================================\n"
                f"Prediction: {result['prediction']}\n"
                f"Probability: {result['probability']:.4f}\n"
                f"Deployed model: {result.get('deployed_model')}\n\n"
                f"Explanation:\n{result['explanation']}\n\n"
                "Inputs:\n"
                + json.dumps(payload, indent=2)
                + "\n"
            )
            st.download_button(
                "Download prediction report (.txt)",
                data=report_text,
                file_name="heart_risk_prediction_report.txt",
                mime="text/plain",
            )

        except requests.RequestException as exc:
            st.error(
                "Could not reach the FastAPI backend. Start it with: "
                "`uvicorn app.main:app --reload`"
            )
            st.caption(str(exc))

    if st.button("Run SHAP + LIME Cross-Check"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/explain", json=payload, timeout=60
            )
            response.raise_for_status()
            result = response.json()
            st.subheader("LIME Cross-Check")
            st.dataframe(pd.DataFrame(result["lime"]), use_container_width=True, hide_index=True)
        except requests.RequestException as exc:
            st.error(f"Explanation request failed: {exc}")

with tab_compare:
    st.subheader("Model Comparison")
    metadata_path = Path(__file__).resolve().parents[1] / "app" / "models" / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = []
        for model_name, metrics in metadata["comparison"].items():
            rows.append(
                {
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1": metrics["f1"],
                    "ROC-AUC": metrics["roc_auc"],
                    "CV F1": metrics["cv_f1_mean"],
                    "CV ROC-AUC": metrics["cv_roc_auc_mean"],
                }
            )
        comp = pd.DataFrame(rows)
        st.dataframe(comp, use_container_width=True, hide_index=True)
        fig = px.bar(
            comp.melt(id_vars="Model", value_vars=["Accuracy", "F1", "ROC-AUC"]),
            x="Model",
            y="value",
            color="variable",
            barmode="group",
            title="Model Performance Comparison",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Selected deployment model: {metadata['deployed_model']}")
    else:
        st.info("Train the models first with `python scripts/train_model.py`.")

with tab_history:
    st.subheader("Prediction History")
    try:
        response = requests.get(f"{API_BASE_URL}/history?limit=100", timeout=10)
        response.raise_for_status()
        items = response.json()["items"]
        if items:
            history = pd.DataFrame(
                [
                    {
                        "ID": x["id"],
                        "Timestamp": x["created_at"],
                        "Prediction": x["prediction"],
                        "Probability": x["probability"],
                    }
                    for x in items
                ]
            )
            st.dataframe(history, use_container_width=True, hide_index=True)
            st.download_button(
                "Export history as CSV",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
            )
        else:
            st.info("No predictions have been saved yet.")
    except requests.RequestException:
        st.info("Start the FastAPI backend to view prediction history.")

with tab_about:
    st.subheader("About the Project")
    st.write(
        """
        This project demonstrates an end-to-end explainable machine-learning workflow:
        data preprocessing, model training and comparison, risk prediction, SHAP local
        explainability, LIME cross-checking, optional local-LLM summarization, FastAPI,
        Streamlit, and SQLite history.
        """
    )
    st.markdown(
        "**Important:** the output is a model estimate for educational demonstration, "
        "not a medical diagnosis."
    )
