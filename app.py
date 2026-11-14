"""Streamlit dashboard: Customer Churn Prediction & Business Intelligence."""
import os
import pickle

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🔮",
    layout="wide",
)

CLEAN_PATH = "data/cleaned_churn.csv"
VIS_DIR = "visuals"
MODEL_DIR = "models"

# ---------------- Violet palette ----------------
BG = "#0F0A1F"
PANEL = "#1A1233"
PANEL_2 = "#241A45"
BORDER = "#3B2D6B"
VIOLET = "#A78BFA"
VIOLET_DEEP = "#7C3AED"
LILAC = "#DDD6FE"
ROSE = "#F0ABFC"
TEXT = "#EDE9FE"
MUTED = "#A99CD0"

# ---------------- Styling ----------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, p, label, li, button, input, select {{
    font-family: 'Outfit', sans-serif !important;
}}
h1, h2, h3, .hero-title, .kpi-value {{
    font-family: 'Fraunces', serif !important;
    letter-spacing: -0.01em;
}}

.stApp {{
    background:
        radial-gradient(900px 500px at 85% -10%, rgba(124,58,237,.28), transparent 60%),
        radial-gradient(700px 400px at -10% 30%, rgba(240,171,252,.12), transparent 60%),
        {BG};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; max-width: 1250px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {PANEL} 0%, {BG} 100%);
    border-right: 1px solid {BORDER};
}}
.side-brand {{
    font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 700;
    color: {LILAC}; margin-bottom: .2rem;
}}
.side-sub {{ color: {MUTED}; font-size: .85rem; margin-bottom: 1.2rem; }}

/* Hero */
.hero {{
    background: linear-gradient(135deg, {VIOLET_DEEP} 0%, #5B21B6 45%, #2E1065 100%);
    border: 1px solid {BORDER};
    border-radius: 22px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 20px 60px rgba(124,58,237,.25);
}}
.hero-title {{
    font-size: 2.5rem; font-weight: 700; color: #fff; line-height: 1.15; margin: 0;
}}
.hero-sub {{ color: {LILAC}; font-weight: 300; font-size: 1.05rem; margin-top: .6rem; }}

/* KPI cards */
.kpi {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
}}
.kpi-label {{ color: {MUTED}; font-size: .9rem; }}
.kpi-value {{ color: {LILAC}; font-size: 2.1rem; font-weight: 700; margin-top: .2rem; }}

/* Section headings */
h2, h3 {{ color: {LILAC} !important; }}

/* Prediction result */
.result {{
    border-radius: 18px; padding: 1.4rem 1.6rem; border: 1px solid {BORDER};
    background: {PANEL_2}; margin-top: 1rem;
}}
.result-title {{ font-family: 'Fraunces', serif; font-size: 1.5rem; color: #fff; }}
.result-sub {{ color: {MUTED}; margin-top: .2rem; }}
.meter {{ height: 12px; border-radius: 99px; background: {BG}; margin-top: 1rem; overflow: hidden; }}
.meter > div {{ height: 100%; border-radius: 99px; }}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {VIOLET_DEEP}, {VIOLET});
    color: #fff; border: 0; border-radius: 12px;
    padding: .6rem 1.6rem; font-weight: 500;
}}
.stButton > button:hover {{ filter: brightness(1.1); color: #fff; }}

[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- Helpers ----------------
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
        st.image(Image.open(path), caption=caption, width="stretch")
    else:
        st.warning(f"Missing: {path}. Run eda.py / train_model.py first.")


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><p class="hero-title">{title}</p>'
        f'<p class="hero-sub">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def kpi(col, label, value):
    col.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def style_chart(chart):
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor=BORDER, domainColor=BORDER
        )
        .configure_legend(labelColor=MUTED, titleColor=MUTED)
    )


# ---------------- Sidebar ----------------
st.sidebar.markdown('<div class="side-brand">🔮 Churn Intelligence</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="side-sub">Predict who is leaving, and why.</div>', unsafe_allow_html=True
)
page = st.sidebar.radio(
    "Navigate",
    ["Customer Analytics", "Visual Insights", "Churn Prediction", "Model Performance"],
)

df = load_data() if os.path.exists(CLEAN_PATH) else None

# ---------------- Customer Analytics ----------------
if page == "Customer Analytics":
    hero(
        "Customer Churn Prediction & Business Intelligence",
        "Explore customer behaviour, spot churn patterns and predict risk with machine learning.",
    )
    if df is None:
        st.error("Run data_preprocessing.py first.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "Total Customers", f"{len(df):,}")
        kpi(k2, "Churn Rate", f"{df['Churn'].mean() * 100:.1f}%")
        kpi(k3, "Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
        kpi(k4, "Avg Tenure (months)", f"{df['tenure'].mean():.1f}")

        st.write("")
        st.subheader("Churn by tenure")
        tmp = df.copy()
        tmp["Tenure group"] = pd.cut(
            tmp["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
        )
        grp = (
            tmp.groupby("Tenure group", observed=True)["Churn"].mean().mul(100).reset_index()
        )
        grp.columns = ["Tenure group (months)", "Churn rate (%)"]
        chart = (
            alt.Chart(grp)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color=VIOLET)
            .encode(
                x=alt.X("Tenure group (months):N", sort=None, axis=alt.Axis(labelAngle=0)),
                y="Churn rate (%):Q",
                tooltip=["Tenure group (months)", alt.Tooltip("Churn rate (%):Q", format=".1f")],
            )
            .properties(height=300)
        )
        st.altair_chart(style_chart(chart), width="stretch")

        st.subheader("Dataset preview")
        st.dataframe(df.head(20), width="stretch")

        with st.expander("Summary statistics"):
            st.dataframe(df.describe(), width="stretch")

# ---------------- Visual Insights ----------------
elif page == "Visual Insights":
    hero("Visual Insights", "The EDA charts behind the churn story.")
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
    hero("Churn Prediction", "Enter a customer profile and get an instant churn risk score.")
    if not os.path.exists(f"{MODEL_DIR}/best_model.pkl"):
        st.error("Run train_model.py first.")
    else:
        model, scaler, feature_cols = load_model_assets()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges", 18.0, 150.0, 70.0)
            total_charges = st.number_input(
                "Total Charges", 0.0, 10000.0, float(tenure * monthly_charges)
            )
            senior = st.selectbox("Senior Citizen", [0, 1])
        with c2:
            st.subheader("Customer")
            gender = st.selectbox("Gender", ["Male", "Female"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c3:
            st.subheader("Services")
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            )
            num_services = st.slider("Number of Active Services", 0, 5, 2)

        if st.button("Predict churn"):
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

            pct = prob * 100
            if pct < 30:
                level, color = "Low risk", "#34D399"
            elif pct < 60:
                level, color = "Medium risk", "#FBBF24"
            else:
                level, color = "High risk", "#F472B6"

            headline = (
                "This customer is likely to churn" if pred == 1 else "This customer is likely to stay"
            )
            st.markdown(
                f"""
<div class="result">
  <div class="result-title">{headline}</div>
  <div class="result-sub">{level} · churn probability {pct:.1f}%</div>
  <div class="meter"><div style="width:{min(pct, 100):.0f}%; background:{color};"></div></div>
</div>
""",
                unsafe_allow_html=True,
            )

# ---------------- Model Performance ----------------
elif page == "Model Performance":
    hero("Model Performance", "How each model scores on accuracy, precision, recall and F1.")
    results_path = f"{MODEL_DIR}/model_results.csv"
    if os.path.exists(results_path):
        results = pd.read_csv(results_path)
        metrics = ["accuracy", "precision", "recall", "f1"]

        st.subheader("Metrics comparison")
        st.dataframe(
            results.style.highlight_max(subset=metrics, color="#6D28D9").format(
                {m: "{:.3f}" for m in metrics}
            ),
            width="stretch",
        )

        long = results.melt(id_vars="model", value_vars=metrics, var_name="Metric", value_name="Score")
        chart = (
            alt.Chart(long)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Metric:N", axis=alt.Axis(labelAngle=0), title=None),
                xOffset="model:N",
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "model:N",
                    scale=alt.Scale(range=[VIOLET, ROSE, "#6D28D9", LILAC]),
                    title="Model",
                ),
                tooltip=["model", "Metric", alt.Tooltip("Score:Q", format=".3f")],
            )
            .properties(height=330)
        )
        st.altair_chart(style_chart(chart), width="stretch")
    else:
        st.error("Run train_model.py first.")

    col1, col2 = st.columns(2)
    with col1:
        show_image(
            f"{VIS_DIR}/confusion_matrix_logistic_regression.png",
            "Logistic Regression - Confusion Matrix",
        )
    with col2:
        show_image(
            f"{VIS_DIR}/confusion_matrix_random_forest.png",
            "Random Forest - Confusion Matrix",
        )

    show_image(f"{VIS_DIR}/feature_importance_rf.png", "Top Feature Importances (Random Forest)")
