"""Exploratory Data Analysis for Customer Churn dataset."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CLEAN_PATH = "data/cleaned_churn.csv"
OUT_DIR = "visuals"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def load():
    return pd.read_csv(CLEAN_PATH)


def churn_distribution(df):
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Churn", data=df, palette="Set2")
    plt.title("Churn Distribution")
    plt.xlabel("Churn (0=No, 1=Yes)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/churn_distribution.png")
    plt.close()


def tenure_vs_churn(df):
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=20, palette="Set2")
    plt.title("Tenure Distribution by Churn")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/tenure_vs_churn.png")
    plt.close()


def monthly_charges_vs_churn(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette="Set2")
    plt.title("Monthly Charges by Churn")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/monthlycharges_vs_churn.png")
    plt.close()


def contract_impact(df):
    contract_cols = [c for c in df.columns if c.startswith("Contract_")]
    if not contract_cols:
        return
    rates = {c.replace("Contract_", ""): df[df[c] == 1]["Churn"].mean() for c in contract_cols}
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(rates.keys()), y=list(rates.values()), palette="Set2")
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/contract_vs_churn.png")
    plt.close()


def feature_importance_corr(df):
    corr_with_target = df.corr(numeric_only=True)["Churn"].drop("Churn").sort_values()
    plt.figure(figsize=(6, 8))
    corr_with_target.plot(kind="barh", color="teal")
    plt.title("Feature Correlation with Churn")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_correlation_with_churn.png")
    plt.close()


def correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlation_heatmap.png")
    plt.close()


def run_eda():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    churn_distribution(df)
    tenure_vs_churn(df)
    monthly_charges_vs_churn(df)
    contract_impact(df)
    feature_importance_corr(df)
    correlation_heatmap(df)
    print(f"EDA complete. Visuals saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    run_eda()
