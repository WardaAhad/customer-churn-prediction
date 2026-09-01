# Customer Churn Prediction System

A Machine Learning project that predicts whether a customer is likely to churn based on customer demographic, service, account, and billing information. The trained Machine Learning model is integrated into a Streamlit web application for interactive churn prediction.

## Project Overview

Customer churn prediction helps businesses identify customers who may stop using their services. This project uses the Telco Customer Churn dataset to train and evaluate Machine Learning models and provides a web interface for making predictions.

The project covers the complete Machine Learning workflow:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Data transformation
* Machine Learning model training
* Model evaluation
* Model saving using Joblib
* Customer churn prediction
* Streamlit application development

## Features

* Customer churn prediction
* Interactive Streamlit interface
* Trained Machine Learning model
* Feature scaling using a saved scaler
* Customer information input
* Churn prediction results
* Probability-based prediction
* Exploratory Data Analysis
* Complete model training notebook
* No separate backend API required

## Machine Learning Workflow

```text
Customer Data
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Feature Scaling
      ↓
Machine Learning Model
      ↓
Churn Prediction
      ↓
Prediction Result
```

## Input Information

The application uses customer information related to:

* Gender
* Senior citizen status
* Partner status
* Dependents
* Tenure
* Phone service
* Internet service
* Contract type
* Payment method
* Monthly charges
* Total charges
* Other subscribed services

## Project Structure

```text
Customer-Churn-Prediction/
│
├── frontend/
│   ├── app.py
│   ├── final_model.pkl
│   ├── scaler.pkl
│   └── requirements.txt
│
├── notebook/
│   └── Churn_Model.ipynb
│
├── Telco-Customer-Churn.csv
├── Customer Churn Prediction System.docx
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## Dataset

The project uses the **Telco Customer Churn Dataset**.

Dataset file:

```text
Telco-Customer-Churn.csv
```

The dataset contains customer demographic information, subscribed services, account information, billing details, and the customer's churn status.

## Model

The trained Machine Learning model is stored as:

```text
frontend/final_model.pkl
```

The feature scaler used during model training is stored as:

```text
frontend/scaler.pkl
```

Both files are loaded by the Streamlit application to generate predictions.

## Installation

Clone the repository:

```bash
git clone https://github.com/WardaAhad/customer-churn-prediction.git
```

Move into the project directory:

```bash
cd customer-churn-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit Application

Run:

```bash
streamlit run frontend/app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## Model Training

The complete model development process is available in:

```text
notebook/Churn_Model.ipynb
```

The notebook includes:

* Data loading
* Data cleaning
* Exploratory Data Analysis
* Feature preprocessing
* Feature transformation
* Model training
* Model evaluation
* Model saving

## Prediction Process

The Streamlit application takes customer information as input, applies the same preprocessing and scaling used during training, and passes the processed data to the trained Machine Learning model.

The application then displays the predicted customer churn result.

## Deployment

The Streamlit application can be deployed using Streamlit Community Cloud.

## Future Improvements

* Hyperparameter tuning
* Advanced Machine Learning models
* Improved feature engineering
* Model performance optimization
* Enhanced user interface
* Additional customer analytics
* Model monitoring

## Project Goal

The goal of this project is to build an end-to-end Customer Churn Prediction System that demonstrates how Machine Learning can be used to identify customers who are likely to leave a service and support data-driven customer retention strategies.
