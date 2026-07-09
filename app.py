"""Streamlit dashboard: Customer Churn Prediction & Business Intelligence."""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

CLEAN_PATH = "data/cleaned_churn.csv"
VIS_DIR = "visuals"
MODEL_DIR = "models"


@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_PATH)


@st.cache_resource
def load_model_assets():
    with open(f"{MODEL_DIR}/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/feature_columns.pkl", "rb") as f:
        cols = pickle.load(f)
    return model, scaler, cols


def show_image(path, caption):
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing: {path}. Run eda.py / train_model.py first.")


st.sidebar.title("Churn Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Customer Analytics", "Visual Insights", "Churn Prediction", "Model Performance"]
)

df = load_data() if os.path.exists(CLEAN_PATH) else None

# ---------------- Customer Analytics ----------------
if page == "Customer Analytics":
    st.title("Customer Analytics")
    if df is None:
        st.error("Run data_preprocessing.py first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", len(df))
        c2.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
        c3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
        c4.metric("Avg Tenure (months)", f"{df['tenure'].mean():.1f}")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

# ---------------- Visual Insights ----------------
elif page == "Visual Insights":
    st.title("Visual Insights")
    col1, col2 = st.columns(2)
    with col1:
        show_image(f"{VIS_DIR}/churn_distribution.png", "Churn Distribution")
        show_image(f"{VIS_DIR}/monthlycharges_vs_churn.png", "Monthly Charges vs Churn")
        show_image(f"{VIS_DIR}/feature_correlation_with_churn.png", "Feature Correlation with Churn")
    with col2:
        show_image(f"{VIS_DIR}/tenure_vs_churn.png", "Tenure vs Churn")
        show_image(f"{VIS_DIR}/contract_vs_churn.png", "Churn Rate by Contract Type")
        show_image(f"{VIS_DIR}/correlation_heatmap.png", "Correlation Heatmap")

# ---------------- Churn Prediction ----------------
elif page == "Churn Prediction":
    st.title("Churn Prediction")
    if not os.path.exists(f"{MODEL_DIR}/best_model.pkl"):
        st.error("Run train_model.py first.")
    else:
        model, scaler, feature_cols = load_model_assets()
        st.write("Enter customer details:")

        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges", 18.0, 150.0, 70.0)
            total_charges = st.number_input("Total Charges", 0.0, 10000.0, float(tenure * monthly_charges))
            senior = st.selectbox("Senior Citizen", [0, 1])
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c3:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
            )
            num_services = st.slider("Number of Active Services", 0, 5, 2)

        if st.button("Predict Churn"):
            row = {c: 0 for c in feature_cols}
            row["tenure"] = tenure
            row["MonthlyCharges"] = monthly_charges
            row["TotalCharges"] = total_charges
            row["SeniorCitizen"] = senior
            row["gender"] = 1 if gender == "Male" else 0
            row["Partner"] = 1 if partner == "Yes" else 0
            row["Dependents"] = 1 if dependents == "Yes" else 0
            row["PaperlessBilling"] = 1 if paperless == "Yes" else 0
            row["AvgMonthlySpend"] = total_charges / (tenure + 1)
            row["NumServices"] = num_services

            for key, val in [
                (f"Contract_{contract}", 1),
                (f"InternetService_{internet}", 1),
                (f"PaymentMethod_{payment}", 1),
            ]:
                if key in row:
                    row[key] = val

            X_input = pd.DataFrame([row])[feature_cols]

            if hasattr(model, "coef_"):
                X_scaled = scaler.transform(X_input)
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0][1]
            else:
                pred = model.predict(X_input)[0]
                prob = model.predict_proba(X_input)[0][1]

            if pred == 1:
                st.error(f"Prediction: Customer likely to CHURN (probability: {prob*100:.1f}%)")
            else:
                st.success(f"Prediction: Customer likely to STAY (churn probability: {prob*100:.1f}%)")
            st.progress(min(int(prob * 100), 100))

# ---------------- Model Performance ----------------
elif page == "Model Performance":
    st.title("Model Performance")
    results_path = f"{MODEL_DIR}/model_results.csv"
    if os.path.exists(results_path):
        results = pd.read_csv(results_path)
        st.subheader("Metrics Comparison")
        st.dataframe(results.style.highlight_max(subset=["accuracy", "precision", "recall", "f1"], color="lightgreen"),
                     use_container_width=True)
        st.bar_chart(results.set_index("model")[["accuracy", "precision", "recall", "f1"]])
    else:
        st.error("Run train_model.py first.")

    col1, col2 = st.columns(2)
    with col1:
        show_image(f"{VIS_DIR}/confusion_matrix_logistic_regression.png", "Logistic Regression - Confusion Matrix")
    with col2:
        show_image(f"{VIS_DIR}/confusion_matrix_random_forest.png", "Random Forest - Confusion Matrix")

    show_image(f"{VIS_DIR}/feature_importance_rf.png", "Top Feature Importances (Random Forest)")
