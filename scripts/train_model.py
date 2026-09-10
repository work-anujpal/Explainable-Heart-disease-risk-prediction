from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_DIR = ROOT / "app" / "models"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    expected = FEATURES + ["target"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target is binary in the application. If supplied as 0..4, convert >0 to 1.
    df = df.dropna(subset=["target"])
    df["target"] = (df["target"] > 0).astype(int)
    return df

def metric_dict(y_true, pred, proba):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} does not exist. Run `python scripts/download_data.py` "
            "or place a compatible heart.csv in data/."
        )

    df = clean_frame(pd.read_csv(DATA_PATH))
    X = df[FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Median imputation first, then StandardScaler. We save both steps in a single
    # preprocessing Pipeline while keeping the historical artifact name scaler.pkl.
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison = {}

    for name, model in models.items():

        # Cross-validation pipeline prevents preprocessing data leakage.
        # The imputer and scaler are fitted independently inside each CV fold.
        cv_pipeline = Pipeline([
            ("preprocessor", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])),
            ("model", model),
        ])

        cv_scores = cross_validate(
            cv_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={"f1": "f1", "roc_auc": "roc_auc"},
            n_jobs=-1,
        )

        # Holdout evaluation uses preprocessing fitted only on the training set.
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = metric_dict(y_test, pred, proba)
        metrics["cv_f1_mean"] = float(np.mean(cv_scores["test_f1"]))
        metrics["cv_roc_auc_mean"] = float(np.mean(cv_scores["test_roc_auc"]))

        comparison[name] = metrics

    # The project documents recommend a tree model for deployment so SHAP can use
    # TreeExplainer. Compare Random Forest and XGBoost using CV F1 first, then ROC-AUC.
    deploy_candidates = ["Random Forest", "XGBoost"]
    deployed_name = max(
        deploy_candidates,
        key=lambda name: (
            comparison[name]["cv_f1_mean"],
            comparison[name]["cv_roc_auc_mean"],
        ),
    )

    deployed_model = models[deployed_name]
    deployed_model.fit(X_train_scaled, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployed_model, MODEL_DIR / "model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "scaler.pkl")

    # Keep original-space background rows so LIME can generate readable rules.
    background = X_train.sample(min(200, len(X_train)), random_state=42).reset_index(drop=True)
    joblib.dump(background, MODEL_DIR / "background.joblib")

    metadata = {
        "deployed_model": deployed_name,
        "feature_names": FEATURES,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "comparison": comparison,
        "selection_rule": "Best of Random Forest/XGBoost by CV F1, tie-break by CV ROC-AUC",
        "random_state": 42,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nMODEL COMPARISON")
    print("=" * 78)
    for name, m in comparison.items():
        print(
            f"{name:20s} "
            f"Acc={m['accuracy']:.3f} "
            f"Prec={m['precision']:.3f} "
            f"Recall={m['recall']:.3f} "
            f"F1={m['f1']:.3f} "
            f"ROC-AUC={m['roc_auc']:.3f} "
            f"CV-F1={m['cv_f1_mean']:.3f} "
            f"CV-AUC={m['cv_roc_auc_mean']:.3f}"
        )
    print(f"\nDeployed model: {deployed_name}")
    print(f"Saved artifacts to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
