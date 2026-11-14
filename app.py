"""Streamlit dashboard: Customer Churn Prediction & Business Intelligence."""
import os
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🔮",
    layout="wide",
)
alt.data_transformers.disable_max_rows()

CLEAN_PATH = "data/cleaned_churn.csv"
VIS_DIR = "visuals"
MODEL_DIR = "models"

# ---------------- Violet palette ----------------
P = {
    "BG": "#0F0A1F",
    "PANEL": "#1A1233",
    "PANEL_2": "#241A45",
    "BORDER": "#3B2D6B",
    "VIOLET": "#A78BFA",
    "VIOLET_DEEP": "#7C3AED",
    "LILAC": "#DDD6FE",
    "ROSE": "#F0ABFC",
    "TEXT": "#EDE9FE",
    "MUTED": "#A99CD0",
}
BG, PANEL, PANEL_2, BORDER = P["BG"], P["PANEL"], P["PANEL_2"], P["BORDER"]
VIOLET, VIOLET_DEEP, LILAC, ROSE = P["VIOLET"], P["VIOLET_DEEP"], P["LILAC"], P["ROSE"]
TEXT, MUTED = P["TEXT"], P["MUTED"]

STAY_COLOR, CHURN_COLOR = "#8B5CF6", "#F0ABFC"
COLOR_SCALE = alt.Scale(domain=["Stayed", "Churned"], range=[STAY_COLOR, CHURN_COLOR])

# ---------------- Styling ----------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Outfit:wght@300;400;500;600&display=swap');

html, body, .stApp, .stMarkdown, p, label, li, button, input, select, textarea {
    font-family: 'Outfit', sans-serif !important;
}
span[data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }
h1, h2, h3, .hero-title, .kpi-value, .side-brand, .result-title {
    font-family: 'Fraunces', serif !important;
    letter-spacing: -0.01em;
}

.stApp {
    background:
        radial-gradient(900px 500px at 85% -10%, rgba(124,58,237,.28), transparent 60%),
        radial-gradient(700px 400px at -10% 30%, rgba(240,171,252,.12), transparent 60%),
        $BG;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1250px; }
h2, h3 { color: $LILAC !important; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, $PANEL 0%, $BG 100%);
    border-right: 1px solid $BORDER;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
.brand { display: flex; align-items: center; gap: .8rem; margin: .4rem 0 1.4rem 0; }
.brand svg { flex-shrink: 0; filter: drop-shadow(0 6px 14px rgba(124,58,237,.55)); }
.side-brand { font-size: 1.35rem; font-weight: 700; color: $LILAC; line-height: 1.1; }
.side-sub { color: $MUTED; font-size: .8rem; margin-top: .2rem; }
.side-section {
    color: $MUTED; font-size: .78rem; font-weight: 500; letter-spacing: .08em;
    text-transform: uppercase; margin: 1.4rem 0 .5rem 0;
    padding-bottom: .4rem; border-bottom: 1px solid $BORDER;
}

/* nav pills */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: .35rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    width: 100%; padding: .65rem .9rem; border-radius: 12px;
    border: 1px solid transparent; transition: all .15s ease; cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: $PANEL_2; border-color: $BORDER;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, $VIOLET_DEEP, #5B21B6);
    border-color: $VIOLET; box-shadow: 0 8px 22px rgba(124,58,237,.35);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p { font-size: .98rem; }

/* match counter */
.match {
    background: $PANEL_2; border: 1px solid $BORDER; border-radius: 14px;
    padding: .8rem 1rem; margin: .8rem 0;
}
.match b { font-family: 'Fraunces', serif; font-size: 1.5rem; color: $LILAC; }
.match span { color: $MUTED; font-size: .85rem; }
.match .bar { height: 6px; border-radius: 99px; background: $BG; margin-top: .55rem; overflow: hidden; }
.match .bar > div { height: 100%; background: linear-gradient(90deg, $VIOLET_DEEP, $ROSE); }

/* footer */
.foot {
    margin-top: 1.6rem; padding: .9rem 1rem; border-radius: 14px;
    background: $PANEL_2; border: 1px solid $BORDER;
}
.foot .who { color: $LILAC; font-weight: 500; font-size: .92rem; }
.foot .what { color: $MUTED; font-size: .78rem; margin-bottom: .6rem; }
.foot a {
    display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 10px; margin-right: .5rem;
    background: $BG; border: 1px solid $BORDER; transition: all .15s ease;
}
.foot a:hover { border-color: $VIOLET; transform: translateY(-2px); }
.foot svg { fill: $LILAC; }

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, $VIOLET_DEEP 0%, #5B21B6 45%, #2E1065 100%);
    border: 1px solid $BORDER; border-radius: 22px;
    padding: 2.4rem 2.6rem; margin-bottom: 1.6rem;
    box-shadow: 0 20px 60px rgba(124,58,237,.25);
}
.hero .hero-title {
    font-size: 2.6rem !important; font-weight: 700; color: #fff !important;
    line-height: 1.15; margin: 0 !important;
}
.hero .hero-sub {
    color: $LILAC !important; font-weight: 300; font-size: 1.1rem !important;
    margin: .7rem 0 0 0 !important;
}

/* ---------- KPI / insight cards ---------- */
.kpi {
    background: $PANEL; border: 1px solid $BORDER; border-radius: 16px;
    padding: 1.1rem 1.3rem; height: 100%;
}
.kpi-label { color: $MUTED; font-size: .9rem; }
.kpi-value { color: $LILAC; font-size: 2.1rem; font-weight: 700; margin-top: .2rem; }
.kpi-note { color: $MUTED; font-size: .8rem; margin-top: .1rem; }

/* chart panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: $PANEL; border: 1px solid $BORDER !important;
    border-radius: 18px; padding: .4rem .6rem;
}

/* prediction result */
.result {
    border-radius: 18px; padding: 1.4rem 1.6rem; border: 1px solid $BORDER;
    background: $PANEL_2; margin-top: 1rem;
}
.result-title { font-size: 1.5rem; color: #fff; }
.result-sub { color: $MUTED; margin-top: .2rem; }
.meter { height: 12px; border-radius: 99px; background: $BG; margin-top: 1rem; overflow: hidden; }
.meter > div { height: 100%; border-radius: 99px; }

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, $VIOLET_DEEP, $VIOLET);
    color: #fff; border: 0; border-radius: 12px; padding: .6rem 1.6rem; font-weight: 500;
}
.stButton > button:hover { filter: brightness(1.1); color: #fff; }
section[data-testid="stSidebar"] .stButton > button {
    background: transparent; border: 1px solid $BORDER; color: $LILAC; width: 100%;
}
section[data-testid="stSidebar"] .stButton > button:hover { border-color: $VIOLET; }

[data-testid="stDataFrame"] { border: 1px solid $BORDER; border-radius: 12px; overflow: hidden; }

/* static PNG charts -> dark mode look */
[data-testid="stImage"] img {
    filter: invert(.92) hue-rotate(180deg); border-radius: 12px;
}
</style>
"""
for _k in sorted(P, key=len, reverse=True):
    CSS = CSS.replace("$" + _k, P[_k])
st.markdown(CSS, unsafe_allow_html=True)


# ---------------- Data helpers ----------------
def derive_category(d, raw_col, prefix, base):
    """Rebuild a readable category from one-hot columns (or use the raw column)."""
    if raw_col in d.columns:
        return d[raw_col].astype(str)
    out = pd.Series(base, index=d.index)
    for c in [c for c in d.columns if c.startswith(prefix)]:
        out[d[c] == 1] = c[len(prefix):]
    return out


@st.cache_data
def load_data():
    d = pd.read_csv(CLEAN_PATH)
    bool_cols = d.select_dtypes("bool").columns
    d[bool_cols] = d[bool_cols].astype(int)
    d["ChurnLabel"] = d["Churn"].map({0: "Stayed", 1: "Churned"})
    d["ContractType"] = derive_category(d, "Contract", "Contract_", "Month-to-month")
    d["InternetType"] = derive_category(d, "InternetService", "InternetService_", "DSL")
    d["TenureGroup"] = pd.cut(
        d["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]
    ).astype(str)
    return d


@st.cache_resource
def load_model_assets():
    with open(f"{MODEL_DIR}/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/feature_columns.pkl", "rb") as f:
        cols = pickle.load(f)
    return model, scaler, cols


HELPER_COLS = ["ChurnLabel", "ContractType", "InternetType", "TenureGroup"]


# ---------------- UI helpers ----------------
def show_image(path, caption):
    if os.path.exists(path):
        with st.container(border=True):
            st.image(Image.open(path), caption=caption, width="stretch")
    else:
        st.warning(f"Missing: {path}. Run eda.py / train_model.py first.")


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><p class="hero-title">{title}</p>'
        f'<p class="hero-sub">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def kpi(col, label, value, note=""):
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    col.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def style_chart(chart):
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor=BORDER, domainColor=BORDER,
            labelFont="Outfit", titleFont="Outfit", gridOpacity=0.5,
        )
        .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFont="Outfit")
        .configure_title(
            color=LILAC, font="Fraunces", fontSize=17, anchor="start", offset=14,
        )
    )


def show_chart(chart):
    with st.container(border=True):
        if chart is None:
            st.info("Not enough variety in the filtered data to draw this chart.")
        else:
            st.altair_chart(style_chart(chart), theme=None, width="stretch")


# ---------------- Charts ----------------
def donut(d):
    c = d["ChurnLabel"].value_counts().reset_index()
    c.columns = ["Status", "Customers"]
    return (
        alt.Chart(c)
        .mark_arc(innerRadius=75, outerRadius=118, cornerRadius=7, stroke=PANEL, strokeWidth=3)
        .encode(
            theta=alt.Theta("Customers:Q", stack=True),
            color=alt.Color("Status:N", scale=COLOR_SCALE, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Status", "Customers"],
        )
        .properties(title="Churn distribution", height=300)
    )


def rate_bar(d, col, title, order=None):
    g = d.groupby(col)["Churn"].mean().mul(100).reset_index()
    g.columns = [col, "ChurnRate"]
    base = alt.Chart(g).encode(
        x=alt.X(f"{col}:N", sort=order if order else "-y", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("ChurnRate:Q", title="Churn rate (%)"),
    )
    bars = base.mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        color=alt.Color("ChurnRate:Q", scale=alt.Scale(range=[VIOLET, ROSE]), legend=None),
        tooltip=[col, alt.Tooltip("ChurnRate:Q", format=".1f", title="Churn rate (%)")],
    )
    labels = base.mark_text(dy=-10, color=TEXT, fontSize=13).encode(
        text=alt.Text("ChurnRate:Q", format=".1f")
    )
    return (bars + labels).properties(title=title, height=300)


def tenure_density(d):
    return (
        alt.Chart(d[["tenure", "ChurnLabel"]])
        .transform_density("tenure", groupby=["ChurnLabel"], as_=["tenure", "density"], extent=[0, 72])
        .mark_area(opacity=0.5, line={"strokeWidth": 2}, interpolate="monotone")
        .encode(
            x=alt.X("tenure:Q", title="Tenure (months)"),
            y=alt.Y("density:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("ChurnLabel:N", scale=COLOR_SCALE, legend=alt.Legend(orient="top", title=None)),
        )
        .properties(title="Tenure distribution by churn", height=300)
    )


def charges_box(d):
    return (
        alt.Chart(d[["MonthlyCharges", "ChurnLabel"]])
        .mark_boxplot(size=70, median={"color": "#ffffff"}, rule={"color": MUTED}, ticks={"color": MUTED})
        .encode(
            x=alt.X("ChurnLabel:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("MonthlyCharges:Q", title="Monthly charges ($)", scale=alt.Scale(zero=False)),
            color=alt.Color("ChurnLabel:N", scale=COLOR_SCALE, legend=None),
        )
        .properties(title="Monthly charges vs churn", height=300)
    )


def corr_bar(d):
    if d["Churn"].nunique() < 2:
        return None
    c = d.select_dtypes("number").corr()["Churn"].drop("Churn").dropna()
    c = c.reindex(c.abs().sort_values(ascending=False).head(12).index)
    cdf = pd.DataFrame({"Feature": c.index, "Correlation": c.values})
    return (
        alt.Chart(cdf)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            y=alt.Y("Feature:N", sort=None, title=None),
            x=alt.X("Correlation:Q"),
            color=alt.condition(alt.datum.Correlation > 0, alt.value(ROSE), alt.value(VIOLET)),
            tooltip=["Feature", alt.Tooltip("Correlation:Q", format=".3f")],
        )
        .properties(title="Feature correlation with churn", height=380)
    )


def corr_heatmap(d):
    if d["Churn"].nunique() < 2:
        return None
    num = d.select_dtypes("number")
    num = num.loc[:, num.std() > 0]
    corr = num.corr()
    top = corr["Churn"].drop("Churn").abs().sort_values(ascending=False).head(9).index.tolist() + ["Churn"]
    m = corr.loc[top, top]
    m.index.name = "Feature"
    m = m.reset_index().melt("Feature", var_name="Feature2", value_name="Correlation")
    base = alt.Chart(m).encode(
        x=alt.X("Feature2:N", title=None, sort=top, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("Feature:N", title=None, sort=top),
    )
    rect = base.mark_rect(cornerRadius=4, stroke=BG, strokeWidth=2).encode(
        color=alt.Color(
            "Correlation:Q",
            scale=alt.Scale(domain=[-1, 0, 1], range=["#6D28D9", PANEL_2, ROSE]),
            legend=None,
        ),
        tooltip=["Feature", "Feature2", alt.Tooltip("Correlation:Q", format=".2f")],
    )
    text = base.mark_text(fontSize=10).encode(
        text=alt.Text("Correlation:Q", format=".2f"),
        color=alt.condition("abs(datum.Correlation) > 0.45", alt.value("#ffffff"), alt.value(MUTED)),
    )
    return (rect + text).properties(title="Correlation heatmap", height=380)


# ---------------- Sidebar ----------------
LOGO = """
<svg width="46" height="46" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#C4B5FD"/><stop offset="1" stop-color="#7C3AED"/></linearGradient></defs>
<rect width="48" height="48" rx="14" fill="url(#lg)"/>
<path d="M12 31 L20 22 L26 28 L36 16" stroke="#fff" stroke-width="3.2" fill="none"
 stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="36" cy="16" r="3.2" fill="#fff"/></svg>
"""
GITHUB = (
    '<svg width="18" height="18" viewBox="0 0 24 24"><path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 '
    '9.39 7.86 10.91.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.54-3.88-1.54-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72'
    '.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28'
    '-5.24-5.68 0-1.25.45-2.28 1.18-3.08-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.78 0c2.2-1.49 '
    '3.17-1.18 3.17-1.18.62 1.58.23 2.75.11 3.04.74.8 1.18 1.83 1.18 3.08 0 4.41-2.69 5.38-5.25 5.67.41.36.78 1.06.78 '
    '2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5z"/></svg>'
)
LINKEDIN = (
    '<svg width="18" height="18" viewBox="0 0 24 24"><path d="M20.45 20.45h-3.56v-5.57c0-1.33-.03-3.04-1.85-3.04-1.86 '
    '0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 '
    '7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.56V9h3.56v11.45z"/></svg>'
)

st.sidebar.markdown(
    f'<div class="brand">{LOGO}<div><div class="side-brand">Churn Intelligence</div>'
    f'<div class="side-sub">Predict who is leaving, and why.</div></div></div>',
    unsafe_allow_html=True,
)

NAV = {
    "📊  Customer Analytics": "Customer Analytics",
    "🔍  Visual Insights": "Visual Insights",
    "🎯  Churn Prediction": "Churn Prediction",
    "🏆  Model Performance": "Model Performance",
}
st.sidebar.markdown('<div class="side-section">Navigate</div>', unsafe_allow_html=True)
page = NAV[st.sidebar.radio("Navigate", list(NAV.keys()), label_visibility="collapsed")]

df = load_data() if os.path.exists(CLEAN_PATH) else None
fdf = df

if df is not None:
    contracts = sorted(df["ContractType"].unique())
    internets = sorted(df["InternetType"].unique())
    t_min, t_max = int(df["tenure"].min()), int(df["tenure"].max())
    c_min, c_max = int(np.floor(df["MonthlyCharges"].min())), int(np.ceil(df["MonthlyCharges"].max()))

    defaults = {
        "f_contract": contracts,
        "f_internet": internets,
        "f_tenure": (t_min, t_max),
        "f_charges": (c_min, c_max),
        "f_senior": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def reset_filters():
        for k, v in defaults.items():
            st.session_state[k] = v

    st.sidebar.markdown('<div class="side-section">🎛️ Filters</div>', unsafe_allow_html=True)
    sel_contract = st.sidebar.multiselect("📄 Contract type", contracts, key="f_contract")
    sel_internet = st.sidebar.multiselect("🌐 Internet service", internets, key="f_internet")
    sel_tenure = st.sidebar.slider("⏳ Tenure (months)", t_min, t_max, key="f_tenure")
    sel_charges = st.sidebar.slider("💵 Monthly charges ($)", c_min, c_max, key="f_charges")
    senior_only = st.sidebar.toggle("👴 Senior citizens only", key="f_senior")
    st.sidebar.button("↺  Reset filters", on_click=reset_filters)

    mask = (
        df["ContractType"].isin(sel_contract)
        & df["InternetType"].isin(sel_internet)
        & df["tenure"].between(*sel_tenure)
        & df["MonthlyCharges"].between(*sel_charges)
    )
    if senior_only:
        mask &= df["SeniorCitizen"] == 1
    fdf = df[mask]

    share = len(fdf) / len(df) * 100
    st.sidebar.markdown(
        f'<div class="match"><b>{len(fdf):,}</b> <span>of {len(df):,} customers</span>'
        f'<div class="bar"><div style="width:{share:.0f}%"></div></div></div>',
        unsafe_allow_html=True,
    )
    if page not in ("Customer Analytics", "Visual Insights"):
        st.sidebar.caption("Filters apply to the Analytics and Visual Insights pages.")

st.sidebar.markdown(
    f'<div class="foot"><div class="who">Built by Jaswin</div>'
    f'<div class="what">Customer churn prediction project</div>'
    f'<a href="https://github.com/jasw2126" target="_blank" title="GitHub">{GITHUB}</a>'
    f'<a href="https://www.linkedin.com/in/jaswin-b-a201a3291/" target="_blank" title="LinkedIn">{LINKEDIN}</a></div>',
    unsafe_allow_html=True,
)

# ---------------- Customer Analytics ----------------
if page == "Customer Analytics":
    hero(
        "Customer Churn Prediction & Business Intelligence",
        "Explore customer behaviour, spot churn patterns and predict risk with machine learning.",
    )
    if df is None:
        st.error("Run data_preprocessing.py first.")
    elif fdf.empty:
        st.warning("No customers match these filters. Try widening them or reset the filters.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "Total Customers", f"{len(fdf):,}")
        kpi(k2, "Churn Rate", f"{fdf['Churn'].mean() * 100:.1f}%")
        kpi(k3, "Avg Monthly Charges", f"${fdf['MonthlyCharges'].mean():.2f}")
        kpi(k4, "Avg Tenure (months)", f"{fdf['tenure'].mean():.1f}")

        st.write("")
        churned = fdf[fdf["Churn"] == 1]
        risk_by_contract = fdf.groupby("ContractType")["Churn"].mean()
        top_contract = risk_by_contract.idxmax()
        i1, i2, i3 = st.columns(3)
        kpi(i1, "Customers lost", f"{len(churned):,}", "Marked as churned in this view")
        kpi(i2, "Revenue at risk", f"${churned['MonthlyCharges'].sum():,.0f}/mo", "Monthly charges of churned customers")
        kpi(i3, "Riskiest contract", top_contract, f"{risk_by_contract.max() * 100:.1f}% churn rate")

        st.write("")
        a, b = st.columns(2)
        with a:
            show_chart(rate_bar(fdf, "TenureGroup", "Churn rate by tenure (months)", order=["0-12", "13-24", "25-48", "49-72"]))
        with b:
            show_chart(rate_bar(fdf, "ContractType", "Churn rate by contract type"))
        c, d_ = st.columns(2)
        with c:
            show_chart(rate_bar(fdf, "InternetType", "Churn rate by internet service"))
        with d_:
            show_chart(donut(fdf))

        st.subheader("Dataset preview")
        st.dataframe(fdf.drop(columns=HELPER_COLS).head(20), width="stretch")
        with st.expander("Summary statistics"):
            st.dataframe(fdf.drop(columns=HELPER_COLS).describe(), width="stretch")

# ---------------- Visual Insights ----------------
elif page == "Visual Insights":
    hero("Visual Insights", "The patterns behind the churn story, redrawn live from your filters.")
    if df is None:
        st.error("Run data_preprocessing.py first.")
    elif fdf.empty:
        st.warning("No customers match these filters. Try widening them or reset the filters.")
    else:
        a, b = st.columns(2)
        with a:
            show_chart(donut(fdf))
        with b:
            show_chart(tenure_density(fdf))
        c, d_ = st.columns(2)
        with c:
            show_chart(charges_box(fdf))
        with d_:
            show_chart(rate_bar(fdf, "ContractType", "Churn rate by contract type"))
        e, f = st.columns(2)
        with e:
            show_chart(corr_bar(fdf))
        with f:
            show_chart(corr_heatmap(fdf))

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
            .properties(title="Model comparison", height=330)
        )
        show_chart(chart)
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

    # Feature importance: drawn natively from the saved model when possible
    drawn = False
    if os.path.exists(f"{MODEL_DIR}/best_model.pkl"):
        try:
            model, _, feature_cols = load_model_assets()
            if hasattr(model, "feature_importances_"):
                vals = model.feature_importances_
            elif hasattr(model, "coef_"):
                vals = np.abs(model.coef_[0])
            else:
                vals = None
            if vals is not None:
                imp = pd.DataFrame({"Feature": feature_cols, "Importance": vals})
                imp = imp.sort_values("Importance", ascending=False).head(12)
                fi = (
                    alt.Chart(imp)
                    .mark_bar(cornerRadiusEnd=6)
                    .encode(
                        y=alt.Y("Feature:N", sort=None, title=None),
                        x=alt.X("Importance:Q"),
                        color=alt.Color("Importance:Q", scale=alt.Scale(range=[VIOLET, ROSE]), legend=None),
                        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
                    )
                    .properties(title=f"Top features ({type(model).__name__})", height=380)
                )
                show_chart(fi)
                drawn = True
        except Exception:
            drawn = False
    if not drawn:
        show_image(f"{VIS_DIR}/feature_importance_rf.png", "Top Feature Importances (Random Forest)")
