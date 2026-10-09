# Customer Churn Prediction System

End-to-end ML project: predicts whether a telecom customer will churn, with a FastAPI backend and a responsive dashboard, ready for Vercel.

**Features:** single-customer prediction (probability + risk level), batch CSV scoring with downloadable results, top-10 feature importance.

## Structure

```text
├── api/index.py          FastAPI app (Vercel serverless function)
├── public/index.html     Dashboard (served statically by Vercel)
├── models/               churn_model.joblib (full sklearn Pipeline) + metrics.json
├── train.py              Reproducible training script
├── notebooks/            EDA + fair model comparison
├── data/                 Telco-Customer-Churn.csv
├── tests/test_api.py     API tests
├── vercel.json, .vercelignore, requirements.txt, .python-version
```

## Model

`train.py` builds one scikit-learn `Pipeline` (ordinal encoding of categories, median imputation for `TotalCharges`, Random Forest tuned with 5-fold CV). Preprocessing is inside the saved model, so the API needs no hand-written mappings. Test-set results (20% stratified hold-out):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.799 | 0.643 | 0.548 | 0.592 | 0.840 |
| Decision Tree | 0.784 | 0.604 | 0.545 | 0.573 | 0.825 |
| **Random Forest (deployed)** | 0.764 | 0.538 | 0.773 | 0.634 | 0.841 |

The Random Forest is deployed because it catches the most churners (recall 77%) at the best F1.

## Run locally

```bash
pip install -r requirements-dev.txt
python train.py                 # optional, re-creates models/
uvicorn api.index:app --reload  # http://127.0.0.1:8000
pytest
```

## Deploy on Vercel

1. Push this folder to a GitHub repository (repo root = this folder).
2. In Vercel: **Add New → Project →** import the repo. Framework preset: **Other**. No environment variables needed.
3. Deploy, then open `/api/health` — it should return `{"status":"ok"}`.

Or with the CLI: `npm i -g vercel && vercel --prod`.

If you retrain with a different scikit-learn version, update the pinned version in `requirements.txt` to match, otherwise the saved model may not load.

## API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/api/health` | GET | Health check |
| `/api/predict` | POST | One customer (JSON) |
| `/api/predict-batch` | POST | CSV body, up to 10,000 rows (Vercel body limit ≈ 4.5 MB) |
| `/api/feature-importance` | GET | Top 10 features |

Risk levels: Low ≤ 40%, Medium 40–70%, High > 70% churn probability.

## Author

Warda Ahad — BS Artificial Intelligence
