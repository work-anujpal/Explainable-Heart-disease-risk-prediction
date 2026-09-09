from fastapi import APIRouter, Query

from app.db.database import clear_history, get_history

router = APIRouter(tags=["history"])


@router.get("/history")
def history(limit: int = Query(default=50, ge=1, le=500)):
    return {"items": get_history(limit=limit)}


@router.delete("/history")
def delete_history():
    clear_history()
    return {"message": "Prediction history cleared."}


@router.get("/health")
def health():
    return {"status": "ok"}
