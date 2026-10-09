# Customer Churn Prediction System

A static customer-churn dashboard deployed on Vercel. The dashboard runs without a backend and provides illustrative mock scores for single customers and uploaded CSV files; it does not load or call the trained model.

**Demo features:** mock single-customer scores, in-browser CSV scoring with downloadable results, and illustrative signal bars.

## Structure

```text
├── public/index.html     Static dashboard and mock scoring logic
├── models/               Trained scikit-learn pipeline + evaluation metrics
├── train.py              Reproducible model training script
├── notebooks/            EDA + model comparison
├── data/                 Telco customer data and sample CSV
├── vercel.json           Static output configuration
└── requirements.txt      Python dependencies for model training
```

## Model research

`train.py` builds one scikit-learn `Pipeline` (ordinal encoding of categories, median imputation for `TotalCharges`, Random Forest tuned with 5-fold CV). Test-set results (20% stratified hold-out):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.799 | 0.643 | 0.548 | 0.592 | 0.840 |
| Decision Tree | 0.784 | 0.604 | 0.545 | 0.573 | 0.825 |
| **Random Forest (trained, not used by the demo)** | 0.764 | 0.538 | 0.773 | 0.634 | 0.841 |

The dashboard uses a simple browser-side heuristic to produce demo results. These values and the displayed signal bars are illustrative only, not predictions or feature importance from the trained Random Forest. CSV files are processed in the visitor's browser and are not sent to a server.

## Run locally

Open `public/index.html` in a browser to use the static demo. To retrain the model for research:

```bash
pip install -r requirements-dev.txt
python train.py
```

## Deploy on Vercel

1. Push this folder to a GitHub repository (repo root = this folder).
2. In Vercel, import the repository and use the **Other** framework preset.
3. Deploy. Vercel serves the static files from `public/`; no backend or environment variables are required.

Or with the CLI: `npm i -g vercel && vercel --prod`.

## Author

Warda Ahad — BS Artificial Intelligence
