# 📊 Customer Churn Prediction System

## 📌 Project Overview

The Customer Churn Prediction System is an end-to-end Machine Learning web application that predicts whether a customer is likely to churn. The project helps businesses identify at-risk customers and improve customer retention through data-driven insights.

The application consists of:

- 🤖 Machine Learning model for churn prediction
- ⚡ FastAPI backend
- 🎨 Streamlit frontend
- 📊 Single customer prediction
- 📂 Batch prediction using CSV
- 📈 Feature importance visualization

---

# 🚀 Technologies Used

- Python
- Scikit-learn
- FastAPI
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Joblib

---

# 📂 Project Structure

```text
customer-churn-prediction/
│
├── backend/
│   ├── main.py
│   ├── schemas.py
│   ├── final_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── frontend/
│   └── app.py
│
├── notebook/
│
├── requirements.txt
├── README.md
└── Telco-Customer-Churn.csv
```

---

# ✨ Features

## ✅ Single Customer Prediction

Predict whether an individual customer is likely to churn.

## ✅ Batch Prediction

Upload a CSV file and generate churn predictions for multiple customers.

## ✅ Risk Classification

Customers are categorized into:

- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk

## ✅ Feature Importance

Displays the most influential features used by the trained Machine Learning model.

---

# 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | API Health Check |
| /model-info | GET | Model Information |
| /predict | POST | Single Customer Prediction |
| /predict-batch | POST | Batch Prediction |
| /feature-importance | GET | Feature Importance |

---

# ⚙️ Running the Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend URL

```
http://127.0.0.1:8000
```

---

# ⚙️ Running the Frontend

```bash
cd frontend
streamlit run app.py
```

Frontend URL

```
http://localhost:8501
```

---

# 📊 Machine Learning Models Evaluated

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### ✅ Final Selected Model

**Random Forest Classifier**

---

# 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

---

# 🔮 Future Improvements

- SHAP Explainability
- Better UI/UX
- Authentication System
- Database Integration
- Docker Support
- CI/CD Pipeline

---

# 🚀 Deployment

- Backend deployed using Railway
- Frontend deployed using Streamlit Community Cloud
- REST API developed using FastAPI

---
## 🌐 Live Demo

Frontend:
https://customer-churn-prediction-vyy5gcd8dblht32juqikyb.streamlit.app/

Backend:
https://customer-churn-prediction-production-85d6.up.railway.app

---

# 👩‍💻 Author

**Warda Ahad**

BS Artificial Intelligence

---

_Last updated: July 7, 2026_
