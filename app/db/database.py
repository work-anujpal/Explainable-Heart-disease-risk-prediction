from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path(__file__).resolve().parents[2] / "prediction_history.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    prediction TEXT NOT NULL,
    probability REAL NOT NULL,
    inputs_json TEXT NOT NULL,
    explanation TEXT NOT NULL,
    shap_json TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def save_prediction(
    inputs: Dict[str, Any],
    prediction: str,
    probability: float,
    explanation: str,
    shap_items: List[Dict],
) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
            (prediction, probability, inputs_json, explanation, shap_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                prediction,
                probability,
                json.dumps(inputs),
                explanation,
                json.dumps(shap_items),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_history(limit: int = 50) -> List[Dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, prediction, probability, inputs_json, explanation, shap_json
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "prediction": row["prediction"],
                "probability": row["probability"],
                "inputs": json.loads(row["inputs_json"]),
                "explanation": row["explanation"],
                "shap": json.loads(row["shap_json"]),
            }
        )
    return result


def clear_history() -> None:
    init_db()
    with connect() as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()
