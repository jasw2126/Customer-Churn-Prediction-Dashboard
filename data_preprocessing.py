"""Data loading, cleaning, encoding, feature engineering for Telco Customer Churn dataset."""
import pandas as pd
import numpy as np
import os

RAW_PATH = "data/churn.csv"
CLEAN_PATH = "data/cleaned_churn.csv"


def generate_synthetic_data(n=2000, seed=42):
    """Fallback synthetic Telco-style churn dataset if raw CSV not found."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": rng.integers(0, 73, n),
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
        ),
        "MonthlyCharges": np.round(rng.uniform(18, 120, n), 2),
    })
    df["TotalCharges"] = np.round(df["tenure"] * df["MonthlyCharges"] * rng.uniform(0.9, 1.0, n), 2)
    churn_prob = np.where(df["Contract"] == "Month-to-month", 0.42,
                  np.where(df["Contract"] == "One year", 0.15, 0.05))
    df["Churn"] = [rng.choice(["Yes", "No"], p=[p, 1 - p]) for p in churn_prob]

    # inject missing values & duplicates for realistic cleaning step
    mask = rng.choice(df.index, size=int(0.02 * n), replace=False)
    df.loc[mask, "TotalCharges"] = np.nan
    df = pd.concat([df, df.sample(20, random_state=seed)], ignore_index=True)
    return df


def load_data():
    if os.path.exists(RAW_PATH):
        return pd.read_csv(RAW_PATH)
    print("Raw dataset not found. Generating synthetic Telco-style dataset...")
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(RAW_PATH, index=False)
    return df


def clean_data(df):
    df = df.drop_duplicates()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = df.dropna()
    return df


def feature_engineering(df):
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    )
    df["NumServices"] = df[[
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "TechSupport"
    ]].apply(lambda row: sum(v not in ["No", "No internet service", "No phone service"] for v in row), axis=1)
    return df


def encode_data(df):
    df = df.copy()
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "TechSupport",
        "Contract", "PaymentMethod", "TenureGroup"
    ]
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)
    return df


def run_pipeline():
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    df_encoded = encode_data(df)
    if "customerID" in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=["customerID"])
    os.makedirs("data", exist_ok=True)
    df_encoded.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned dataset saved to {CLEAN_PATH} | Shape: {df_encoded.shape}")
    return df_encoded


if __name__ == "__main__":
    run_pipeline()
