import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLUSTER_PATH = os.path.join(BASE_DIR, "data", "processed", "climatology_clusters.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifiers.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.csv")

# ---------- Branding ----------
AUTHOR = "Sameer Ansari"
PROJECT = "Best Time to Travel Predictor"

# ---------- Config ----------
FEATURES = [
    "tavg_hist", "rain_total_hist", "rainy_days_hist", "wind_hist",
    "m_sin", "m_cos", "lat", "lon",
]
PREFERENCES = ["cool", "warm", "dry"]
PREF_DESC = {
    "cool": "Mild weather, 14 to 24 °C",
    "warm": "Warm weather, 24 to 32 °C",
    "dry": "Dry weather, 15 to 32 °C with few rainy days",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title=PROJECT, page_icon="🧭", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0f766e 0%, #1e3a8a 55%, #312e81 100%);
    border-radius: 20px; padding: 2.2rem 2.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.hero-title { font-size: 2.4rem; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; }
.hero-sub { font-size: 1.05rem; color: #cbd5e1; margin-top: 0.3rem; }
.hero-badge {
    display: inline-block; margin-top: 1rem; padding: 0.35rem 0.9rem;
    border-radius: 999px; background: rgba(255,255,255,0.15);
    color: #ffffff; font-size: 0.85rem; font-weight: 600;
    border: 1px solid rgba(255,255,255,0.25);
}

/* KPI cards */
.kpi {
    background: #111827; border: 1px solid #1f2937; border-radius: 16px;
    padding: 1.2rem 1.4rem; height: 100%;
}
.kpi-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; font-weight: 600; }
.kpi-value { font-size: 1.7rem; font-weight: 800; color: #f8fafc; margin-top: 0.3rem; }
.kpi-hint { font-size: 0.85rem; color: #34d399; margin-top: 0.25rem; }

/* Month cards */
.month-card {
    background: linear-gradient(160deg, #0f172a, #1e293b);
    border: 1px solid #334155; border-radius: 16px; padding: 1.1rem; text-align: center;
}
.month-rank { font-size: 0.75rem; color: #5eead4; font-weight: 700; letter-spacing: 1px; }
.month-name { font-size: 1.6rem; font-weight: 800; color: #f1f5f9; margin: 0.2rem 0; }
.month-score { font-size: 0.9rem; color: #94a3b8; }

/* Section titles */
.section-title { font-size: 1.3rem; font-weight: 700; color: #f1f5f9; margin: 0.5rem 0 0.2rem 0; }
.section-sub { font-size: 0.9rem; color: #94a3b8; margin-bottom: 1rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0b1120; border-right: 1px solid #1e293b; }
.side-brand { font-size: 1.1rem; font-weight: 800; color: #5eead4; }
.side-note { font-size: 0.8rem; color: #94a3b8; line-height: 1.5; }

/* Footer */
.footer {
    text-align: center; color: #64748b; font-size: 0.85rem;
    margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid #1e293b;
}
.footer b { color: #5eead4; }

/* Tabs */
button[data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ---------- Functions ----------
@st.cache_data
def load_climatology():
    return pd.read_csv(CLUSTER_PATH)


@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)


def comfort_index(df, w_temp=5, w_rain=3, w_wind=2):
    """Travel Comfort Index (0-100), same logic as the notebook."""
    temp_score = (100 - 8 * np.maximum(0, np.abs(df["tavg"] - 24) - 4)).clip(0, 100)
    rain_score = 100 * (1 - (df["rainy_days"] / 15).clip(0, 1))
    wind_score = 100 * (1 - ((df["wind"] - 20) / 30).clip(0, 1))
    total = max(w_temp + w_rain + w_wind, 1)
    return ((w_temp * temp_score + w_rain * rain_score + w_wind * wind_score) / total).round(1)


def visit_probability(model, df):
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(df))
    return model.predict_proba(df[FEATURES])[:, classes.index(1)]


def heatmap(data, value, zmax, colorscale, fmt):
    pv = (data.pivot(index="city", columns="month", values=value)
              .reindex(columns=range(1, 13)))
    pv.columns = MONTHS
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=MONTHS, y=pv.index.tolist(),
        zmin=0, zmax=zmax, colorscale=colorscale,
        texttemplate=fmt, textfont={"size": 13, "color": "#0f172a"},
        hovertemplate="<b>%{y}</b>, %{x}: %{z:.2f}<extra></extra>",
        showscale=False, xgap=3, ygap=3,
    ))
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter"),
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=False, side="top"),
    )
    return fig


# ---------- Data ----------
clim = load_climatology().copy()
models = load_models()

# ---------- Sidebar ----------
st.sidebar.markdown('<div class="side-brand">Travel Predictor</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="side-note">Choose your travel style and see which cities and months suit you best.</div>',
                    unsafe_allow_html=True)
st.sidebar.markdown("---")

pref = st.sidebar.radio("Preferred weather", PREFERENCES,
                        format_func=lambda p: p.title())
st.sidebar.caption(PREF_DESC[pref])

month_name = st.sidebar.selectbox("Month of interest", MONTHS, index=10)
month_no = MONTHS.index(month_name) + 1

st.sidebar.markdown("---")
st.sidebar.markdown("**Comfort weights**")
w_temp = st.sidebar.slider("Temperature", 0, 10, 5)
w_rain = st.sidebar.slider("Rain", 0, 10, 3)
w_wind = st.sidebar.slider("Wind", 0, 10, 2)

st.sidebar.markdown("---")
st.sidebar.markdown(f'<div class="side-note">Built by <b>{AUTHOR}</b><br>'
                    'Summer Training Project, Data Science</div>', unsafe_allow_html=True)

# ---------- Scores ----------
clim["ml_score"] = visit_probability(models[pref], clim)
clim["comfort"] = comfort_index(clim, w_temp, w_rain, w_wind)
clim["month_name"] = clim["month"].map(lambda m: MONTHS[m - 1])

# ---------- Hero ----------
st.markdown(f"""
<div class="hero">
    <div class="hero-title">{PROJECT}</div>
    <div class="hero-sub">Data-driven travel planning for 9 Indian cities, based on 10 years (2015-2024) of climate data.</div>
    <div class="hero-badge">Designed and built by {AUTHOR}</div>
</div>
""", unsafe_allow_html=True)

# ---------- KPI row ----------
month_df = clim[clim["month"] == month_no].sort_values(["ml_score", "comfort"], ascending=False)
best_ml = month_df.iloc[0]
best_comfort = month_df.sort_values("comfort", ascending=False).iloc[0]
good_count = int((month_df["ml_score"] >= 0.5).sum())

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(f"""<div class="kpi">
        <div class="kpi-label">Best city for {month_name} ({pref.title()})</div>
        <div class="kpi-value">{best_ml['city']}</div>
        <div class="kpi-hint">ML visit probability {best_ml['ml_score']:.0%}</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi">
        <div class="kpi-label">Highest comfort in {month_name}</div>
        <div class="kpi-value">{best_comfort['city']}</div>
        <div class="kpi-hint">Comfort index {best_comfort['comfort']:.0f} / 100</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi">
        <div class="kpi-label">Good cities in {month_name}</div>
        <div class="kpi-value">{good_count} of {clim['city'].nunique()}</div>
        <div class="kpi-hint">Cities with ML score of 50% or more</div>
    </div>""", unsafe_allow_html=True)

st.write("")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Heatmaps", "City explorer", "Month ranking", "Model performance"])

# Tab 1: Heatmaps
with tab1:
    st.markdown(f'<div class="section-title">Visit probability and comfort, all cities</div>'
                f'<div class="section-sub">Preference: {pref.title()} | Weights: temp {w_temp}, rain {w_rain}, wind {w_wind}</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            heatmap(clim, "ml_score", 1, [[0, "#1e293b"], [0.5, "#0d9488"], [1, "#5eead4"]], "%{z:.2f}"),
            use_container_width=True)
        st.caption("ML visit probability (Random Forest)")
    with c2:
        st.plotly_chart(
            heatmap(clim, "comfort", 100, [[0, "#7f1d1d"], [0.5, "#ca8a04"], [1, "#16a34a"]], "%{z:.0f}"),
            use_container_width=True)
        st.caption("Travel Comfort Index (0-100)")

# Tab 2: City explorer
with tab2:
    st.markdown('<div class="section-title">City explorer</div>'
                '<div class="section-sub">Pick a city to see its best months and full monthly profile.</div>',
                unsafe_allow_html=True)
    city = st.selectbox("City", sorted(clim["city"].unique()), key="city_select")
    city_df = clim[clim["city"] == city].sort_values("month")

    top3 = city_df.nlargest(3, "comfort")
    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            st.markdown(f"""<div class="month-card">
                <div class="month-rank">RANK {i + 1}</div>
                <div class="month-name">{row['month_name']}</div>
                <div class="month-score">Comfort {row['comfort']:.0f} | ML {row['ml_score']:.0%}</div>
            </div>""", unsafe_allow_html=True)

    st.write("")
    table = city_df[["month_name", "tavg", "rain_total", "rainy_days", "wind",
                     "ml_score", "comfort", "cluster_name"]].rename(columns={
        "month_name": "Month", "tavg": "Avg temp (°C)", "rain_total": "Rain (mm)",
        "rainy_days": "Rainy days", "wind": "Wind (km/h)", "ml_score": "ML score",
        "comfort": "Comfort", "cluster_name": "Weather cluster"}).round(2)
    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={
            "ML score": st.column_config.ProgressColumn("ML score", min_value=0, max_value=1, format="%.2f"),
            "Comfort": st.column_config.ProgressColumn("Comfort", min_value=0, max_value=100, format="%.0f"),
        },
    )

# Tab 3: Month ranking
with tab3:
    st.markdown(f'<div class="section-title">Cities ranked for {month_name}</div>'
                f'<div class="section-sub">Sorted by ML visit probability, then comfort score.</div>',
                unsafe_allow_html=True)
    ranking = month_df[["city", "avg_temp" if "avg_temp" in month_df else "tavg", "rain_total",
                        "rainy_days", "ml_score", "comfort", "cluster_name"]].rename(columns={
        "city": "City", "tavg": "Avg temp (°C)", "avg_temp": "Avg temp (°C)",
        "rain_total": "Rain (mm)", "rainy_days": "Rainy days", "ml_score": "ML score",
        "comfort": "Comfort", "cluster_name": "Weather cluster"}).round(2)
    st.dataframe(
        ranking, hide_index=True, use_container_width=True,
        column_config={
            "ML score": st.column_config.ProgressColumn("ML score", min_value=0, max_value=1, format="%.2f"),
            "Comfort": st.column_config.ProgressColumn("Comfort", min_value=0, max_value=100, format="%.0f"),
        },
    )

# Tab 4: Model performance
with tab4:
    st.markdown('<div class="section-title">Model performance</div>'
                '<div class="section-sub">Time-based test set: years 2023 and 2024.</div>',
                unsafe_allow_html=True)
    metrics = pd.read_csv(METRICS_PATH)

    fig = go.Figure()
    palette = {"Baseline": "#64748b", "LogisticRegression": "#38bdf8", "RandomForest": "#34d399"}
    for model_name in metrics["model"].unique():
        sub = metrics[metrics["model"] == model_name]
        fig.add_trace(go.Bar(
            name=model_name, x=sub["preference"].str.title(), y=sub["f1"],
            marker_color=palette.get(model_name, "#a78bfa"),
            text=sub["f1"].round(3), textposition="outside",
        ))
    fig.update_layout(
        barmode="group", height=400, yaxis_title="F1 score", yaxis_range=[0, 1.05],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter"),
        legend=dict(orientation="h", y=1.1, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_yaxes(gridcolor="#1e293b")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(metrics, hide_index=True, use_container_width=True)

# ---------- Footer ----------
st.markdown(f"""
<div class="footer">
    Built with Python, Scikit-learn and Streamlit · Data: Open-Meteo Archive API<br>
    Designed and developed by <b>{AUTHOR}</b> · Summer Training Project, Data Science
</div>
""", unsafe_allow_html=True)
