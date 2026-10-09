"""Train the churn model and save it as a single scikit-learn pipeline.

Usage:
    python train.py

The saved pipeline contains the preprocessing (category encoding + median
imputation) and the Random Forest for use in offline experiments.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
RANDOM_STATE = 42

CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURES = CATEGORICAL + NUMERIC


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    # TotalCharges contains blank strings for brand-new customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df["Churn"] == "Yes").astype(int)
    return df[FEATURES], y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Encode categories (sorted, fixed order) and impute missing numbers with the median."""
    categories = [sorted(X[col].unique()) for col in CATEGORICAL]
    return ColumnTransformer(
        [
            ("cat", OrdinalEncoder(categories=categories), CATEGORICAL),
            ("num", SimpleImputer(strategy="median"), NUMERIC),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Baselines, each evaluated on the same held-out test set.
    candidates = {
        "Logistic Regression": Pipeline([
            ("prep", build_preprocessor(X)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]),
        "Decision Tree": Pipeline([
            ("prep", build_preprocessor(X)),
            ("model", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
        ]),
    }
    results = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)

    # Random Forest with a small cross-validated search (balanced classes help recall).
    search = GridSearchCV(
        Pipeline([
            ("prep", build_preprocessor(X)),
            ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        param_grid={
            "model__n_estimators": [200],
            "model__max_depth": [8, 10, 12],
            "model__min_samples_leaf": [3, 5, 10],
        },
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    results["Random Forest (tuned)"] = evaluate(best, X_test, y_test)

    table = pd.DataFrame(results).T.round(3)
    print("Best parameters:", search.best_params_)
    print(table.to_string())

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(
        json.dumps({"best_params": search.best_params_, "test_metrics": results}, indent=2)
    )
    print(f"Saved {MODEL_PATH.relative_to(ROOT)} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
