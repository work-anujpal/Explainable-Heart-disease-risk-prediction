# Explainable Heart Disease Risk Prediction System

A Third Year Data Science project that predicts **High / Low heart-disease risk** from 13 clinical parameters and explains each prediction using **SHAP** and **LIME**.

> **Educational disclaimer:** This application is for learning and decision-support demonstrations only. It is **not** a substitute for professional medical diagnosis or treatment.

## Core features

- 13-field patient input form
- Input validation
- Data preprocessing with `StandardScaler`
- Model comparison: Logistic Regression, Random Forest, XGBoost
- Deployment model selected between Random Forest and XGBoost using cross-validated F1 and ROC-AUC, with Random Forest chosen for the explainability pipeline
- High/Low risk prediction with probability
- SHAP local feature contributions
- LIME cross-check
- Optional plain-language explanation with Ollama + Phi-4-mini
- FastAPI backend
- Streamlit dashboard
- SQLite prediction history
- Model metrics and comparison view

## Project structure

```text
.
├── app/
│   ├── main.py
│   ├── routers/
│   │   ├── predict.py
│   │   └── history.py
│   ├── ml/
│   │   ├── preprocess.py
│   │   ├── explain.py
│   │   └── llm.py
│   ├── db/
│   │   └── database.py
│   └── models/
├── frontend/
│   └── streamlit_app.py
├── notebooks/
│   └── training.ipynb
├── scripts/
│   ├── download_data.py
│   └── train_model.py
├── data/
├── tests/
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## 1. Setup on Windows

Open the project folder in VS Code, then open **Terminal → New Terminal**.

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Get the dataset

### Option A — automatic UCI download

```powershell
python scripts/download_data.py
```

This creates:

```text
data/heart.csv
```

### Option B — your own UCI/Kaggle CSV

Place a CSV at `data/heart.csv` with these columns:

```text
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
```

## 3. Train and compare the models

```powershell
python scripts/train_model.py
```

This creates generated files in `app/models/`:

```text
model.pkl
scaler.pkl
metadata.json
background.joblib
```

For deployment, the project compares the tree-based models (Random Forest and XGBoost) using cross-validation metrics. Random Forest is selected because it achieved a slightly higher cross-validated F1-score than XGBoost and the strongest holdout ROC-AUC among the tree-based candidates. This also keeps the deployed model compatible with the Tree SHAP explainability pipeline. Logistic Regression remains part of the model comparison but is not selected for deployment in this implementation.
## 4. Start the FastAPI backend

Terminal 1:

```powershell
uvicorn app.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## 5. Start the Streamlit dashboard

Open a **second terminal**, activate the same environment, then run:

```powershell
streamlit run frontend/streamlit_app.py
```

Streamlit normally opens:

```text
http://localhost:8501
```

## Optional: local LLM explanation

Install Ollama separately, then:

```powershell
ollama pull phi4-mini
```

Copy `.env.example` to `.env` and set:

```text
ENABLE_LLM=true
```

The application still works without Ollama; a deterministic template explanation is used as a fallback.

## API endpoints

- `GET /health`
- `POST /predict`
- `POST /explain`
- `GET /history`
- `DELETE /history`

## Example prediction body

```json
{
  "age": 55,
  "sex": 1,
  "cp": 0,
  "trestbps": 140,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
```
