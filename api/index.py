"""Customer Churn Prediction API (FastAPI), deployed as a Vercel Python function."""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
PUBLIC_DIR = ROOT / "public"
MAX_BATCH_ROWS = 10_000
MAX_ERRORS_SHOWN = 5

pipeline = joblib.load(MODEL_PATH)
preprocessor = pipeline.named_steps["prep"]
forest = pipeline.named_steps["model"]

# Single source of truth: the allowed values come from the trained encoder.
CATEGORY_VALUES: dict[str, list[str]] = dict(
    zip(preprocessor.transformers_[0][2], preprocessor.named_transformers_["cat"].categories_)
)
NUMERIC_FIELDS: list[str] = list(preprocessor.transformers_[1][2])
FEATURES: list[str] = list(pipeline.feature_names_in_)
FEATURE_IMPORTANCE = sorted(
    zip(preprocessor.get_feature_names_out(), forest.feature_importances_),
    key=lambda item: item[1],
    reverse=True,
)[:10]

app = FastAPI(title="Customer Churn Prediction API", version="2.0.0")


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


def risk_level(probability: float) -> str:
    if probability > 0.7:
        return "High"
    if probability > 0.4:
        return "Medium"
    return "Low"


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate customer rows and return a clean frame with the model's columns."""
    data = raw[FEATURES].copy()
    problems: list[str] = []

    for column, allowed in CATEGORY_VALUES.items():
        data[column] = data[column].astype("string").str.strip()
        for row in data.index[~data[column].isin(allowed)][:MAX_ERRORS_SHOWN]:
            problems.append(f"Row {row + 1}: invalid or missing {column} (allowed: {', '.join(allowed)}).")

    for column in NUMERIC_FIELDS:
        numbers = pd.to_numeric(data[column], errors="coerce")
        blank = data[column].isna() | data[column].astype("string").str.strip().eq("")
        bad = numbers.isna() & ~(blank & (column == "TotalCharges"))  # only TotalCharges may be blank
        bad |= numbers < 0
        for row in data.index[bad][:MAX_ERRORS_SHOWN]:
            problems.append(f"Row {row + 1}: {column} must be a non-negative number.")
        data[column] = numbers

    if problems:
        raise HTTPException(status_code=422, detail=" ".join(problems[:MAX_ERRORS_SHOWN]))
    return data


def score(raw: pd.DataFrame) -> list[dict[str, Any]]:
    features = prepare_features(raw)
    probabilities = pipeline.predict_proba(features)[:, 1]
    return [
        {
            "churn_prediction": int(probability >= 0.5),
            "probability": float(probability),
            "risk_level": risk_level(float(probability)),
        }
        for probability in probabilities
    ]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
def predict_single(customer: CustomerInput) -> dict[str, Any]:
    return score(pd.DataFrame([customer.model_dump()]))[0]


@app.post("/api/predict-batch")
async def predict_batch(request: Request) -> dict[str, Any]:
    content = await request.body()
    if not content.strip():
        raise HTTPException(status_code=400, detail="Upload a non-empty CSV file.")
    try:
        customers = pd.read_csv(BytesIO(content), keep_default_na=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded CSV.") from None

    if customers.empty:
        raise HTTPException(status_code=400, detail="The CSV contains no customer rows.")
    if len(customers) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=413, detail=f"Upload at most {MAX_BATCH_ROWS} customers at a time.")
    missing = [field for field in FEATURES if field not in customers.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV is missing required columns: {', '.join(missing)}")

    results = score(customers)
    # JSON cannot hold NaN, so turn empty cells into null.
    originals = customers.astype(object).where(customers.notna(), None).to_dict(orient="records")
    churned = sum(result["churn_prediction"] for result in results)
    return {
        "results": [{**original, **result} for original, result in zip(originals, results)],
        "summary": {
            "total_customers": len(results),
            "predicted_churn": churned,
            "retained": len(results) - churned,
            "high_risk": sum(result["risk_level"] == "High" for result in results),
        },
    }


@app.get("/api/feature-importance")
def feature_importance() -> dict[str, list[dict[str, float | str]]]:
    return {
        "features": [
            {"feature": str(name), "importance": float(value)} for name, value in FEATURE_IMPORTANCE
        ]
    }


# On Vercel, `public/` is served by the CDN. Locally, serve it from FastAPI too.
if not os.environ.get("VERCEL"):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
