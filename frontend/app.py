from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =====================================================
# LOAD MODEL DIRECTLY (no backend API needed)
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "final_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")


def predict_single(data: dict) -> dict:
    """Mirrors the old backend's /predict logic, run locally."""

    df = pd.DataFrame([data])

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    ).fillna(0)

    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    df = pd.get_dummies(df)

    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]

    scaled = scaler.transform(df)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    if prob > 0.7:
        risk = "High"
    elif prob > 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_prediction": int(pred),
        "probability": float(prob),
        "risk_level": risk,
    }


def predict_batch(customers: list) -> pd.DataFrame:
    """Mirrors the old backend's /predict-batch logic, run locally."""

    df = pd.DataFrame(customers)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    ).fillna(0)

    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    df = pd.get_dummies(df)

    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    prediction_df = df[model_features].copy()

    scaled = scaler.transform(prediction_df)

    preds = model.predict(scaled)
    probs = model.predict_proba(scaled)[:, 1]

    prediction_df["prediction"] = preds
    prediction_df["probability"] = probs

    return prediction_df


def get_feature_importance() -> pd.DataFrame:
    """Mirrors the old backend's /feature-importance logic, run locally."""

    importance = model.feature_importances_
    features = model.feature_names_in_

    result = [
        {"feature": f, "importance": float(i)}
        for f, i in zip(features, importance)
    ]
    result = sorted(result, key=lambda x: x["importance"], reverse=True)

    return pd.DataFrame(result[:10])


# =====================================================
# TITLE
# =====================================================

st.title("📊 Customer Churn Prediction System")


# =====================================================
# MODE SELECTION
# =====================================================

mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Single Prediction", "Batch Prediction"]
)


# =====================================================
# SINGLE PREDICTION
# =====================================================

if mode == "Single Prediction":

    st.subheader("🔮 Predict Single Customer")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    SeniorCitizen = st.number_input(
        "Senior Citizen",
        min_value=0,
        max_value=1,
        value=0
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        value=12
    )

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No"]
    )

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes", "No"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No"]
    )

    Contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer"
        ]
    )

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.50
    )

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=850.00
    )


    # =================================================
    # PREDICT BUTTON
    # =================================================

    if st.button("🔮 Predict Churn"):

        data = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": str(TotalCharges)
        }

        try:

            result = predict_single(data)

            st.write("## 📊 Prediction Result")

            st.json(result)

            # =====================================
            # RISK LEVEL
            # =====================================

            if "risk_level" in result:

                if result["risk_level"] == "High":

                    st.error("🔴 High Churn Risk")

                elif result["risk_level"] == "Medium":

                    st.warning("🟡 Medium Churn Risk")

                else:

                    st.success("🟢 Low Churn Risk")

        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


# =====================================================
# BATCH PREDICTION
# =====================================================

elif mode == "Batch Prediction":

    st.subheader("📂 Upload Customer CSV")

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("### 📄 Uploaded Data")

        st.dataframe(
            df.head()
        )


        customers = df.to_dict(
            orient="records"
        )


        # =================================================
        # BATCH PREDICTION BUTTON
        # =================================================

        if st.button("🚀 Run Batch Prediction"):

            try:

                result_df = predict_batch(customers)

                st.write(
                    "## 📊 Prediction Results"
                )


                st.dataframe(
                    result_df
                )


                # =====================================
                # CHURN DISTRIBUTION
                # =====================================

                if "prediction" in result_df.columns:

                    churn_counts = (
                        result_df["prediction"]
                        .value_counts()
                    )


                    st.write(
                        "## 📈 Churn Distribution"
                    )


                    st.bar_chart(
                        churn_counts
                    )


                # =====================================
                # HIGH RISK CUSTOMERS
                # =====================================

                if "probability" in result_df.columns:

                    high_risk = result_df[
                        result_df["probability"] > 0.7
                    ]


                    st.write(
                        "## 🔴 High Risk Customers"
                    )


                    st.dataframe(
                        high_risk
                    )


                # =====================================
                # FEATURE IMPORTANCE
                # =====================================

                st.subheader(
                    "📌 Feature Importance"
                )


                try:

                    feature_df = get_feature_importance()

                    if (
                        "feature" in feature_df.columns
                        and
                        "importance" in feature_df.columns
                    ):

                        fig, ax = plt.subplots(
                            figsize=(10, 6)
                        )


                        ax.barh(
                            feature_df["feature"],
                            feature_df["importance"]
                        )


                        ax.invert_yaxis()


                        ax.set_xlabel(
                            "Importance"
                        )


                        ax.set_ylabel(
                            "Feature"
                        )


                        ax.set_title(
                            "Feature Importance"
                        )


                        st.pyplot(fig)


                    else:

                        st.warning(
                            "Feature importance is unavailable."
                        )

                except Exception as e:

                    st.warning(
                        f"Could not load feature importance: {e}"
                    )

            except Exception as e:

                st.error(
                    f"❌ Batch prediction failed: {e}"
                )