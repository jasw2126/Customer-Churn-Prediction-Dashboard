"""Train Logistic Regression & Random Forest churn models, evaluate, save best."""
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

CLEAN_PATH = "data/cleaned_churn.csv"
MODEL_DIR = "models"
VIS_DIR = "visuals"


def load_data():
    df = pd.read_csv(CLEAN_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def evaluate(name, y_test, y_pred):
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print(f"\n{name} Results:")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}")
    print(classification_report(y_test, y_pred))
    return metrics


def plot_confusion_matrix(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(f"{VIS_DIR}/confusion_matrix_{fname}.png")
    plt.close()


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    results.append(evaluate("Logistic Regression", y_test, y_pred_lr))
    plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test, y_pred_rf))
    plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")

    # Feature importance (RF)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
    plt.figure(figsize=(7, 6))
    importances.plot(kind="barh", color="darkorange")
    plt.gca().invert_yaxis()
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/feature_importance_rf.png")
    plt.close()

    # Pick best model by F1 score
    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["f1"].idxmax()]
    best_model = rf if best_row["model"] == "Random Forest" else lr
    best_name = best_row["model"]

    with open(f"{MODEL_DIR}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODEL_DIR}/feature_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)
    results_df.to_csv(f"{MODEL_DIR}/model_results.csv", index=False)

    print(f"\nBest model: {best_name} (F1={best_row['f1']:.4f}) saved to {MODEL_DIR}/best_model.pkl")
    return results_df


if __name__ == "__main__":
    train()
