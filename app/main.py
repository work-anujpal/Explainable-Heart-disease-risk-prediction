from fastapi import FastAPI

from app.db.database import init_db
from app.routers.history import router as history_router
from app.routers.predict import router as predict_router

app = FastAPI(
    title="Explainable Heart Disease Risk Prediction API",
    description=(
        "Educational machine-learning API for heart-disease risk prediction "
        "with SHAP/LIME explanations. Not a medical diagnosis."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup() -> None:
    init_db()


app.include_router(history_router)
app.include_router(predict_router)


@app.get("/")
def root():
    return {
        "name": "Explainable Heart Disease Risk Prediction System",
        "docs": "/docs",
        "disclaimer": "Educational decision-support demo only; not a medical diagnosis.",
    }
