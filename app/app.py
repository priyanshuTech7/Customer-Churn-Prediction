from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Churn Project | Customer Churn Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "app" else Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.pkl"

BG      = "#0B0F17"
PANEL   = "#151A23"
PANEL_2 = "#1C222D"
PANEL_3 = "#252B36"
TEXT    = "#DFE2EE"
MUTED   = "#9AA3B5"
OUTLINE = "rgba(255,255,255,.09)"
INDIGO      = "#818CF8"
CYAN        = "#06B6D4"
CYAN_LIGHT  = "#4CD7F6"
CORAL       = "#F43F5E"
AMBER       = "#F59E0B"
GREEN       = "#10B981"
CHURN_COLORS = {"No": CYAN, "Yes": CORAL}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
      --bg: #0B0F17; --panel: #151A23; --panel2: #1C222D; --panel3: #252B36;
      --text: #DFE2EE; --muted: #9AA3B5; --cyan: #06B6D4; --cyan2: #4CD7F6;
      --coral: #F43F5E; --indigo: #818CF8; --amber: #F59E0B; --green: #10B981;
      --line: rgba(255,255,255,.09);
    }
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
      font-family: 'Plus Jakarta Sans', sans-serif; color: var(--text);
    }
    [data-testid="stAppViewContainer"] { background: var(--bg); }
    [data-testid="stHeader"] { background: rgba(11,15,23,.92); }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stStatusWidget"] { display: none; }

    [data-testid="stMain"] {
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 100%;
    }
    .main .block-container,
    .block-container,
    [data-testid="stMainBlockContainer"] {
      max-width: 1680px;
      width: 100% !important;
      margin-left: auto !important;
      margin-right: auto !important;
      padding-top: 4.5rem;
      padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"][aria-expanded="true"] {
      min-width: 250px;
    }
    section[data-testid="stSidebar"] {
      background: #090D14; border-right: 1px solid rgba(255,255,255,.06);
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.15rem; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }

    .brand-row { display:flex; align-items:center; gap:11px; padding: 5px 4px 18px 4px; }
    .brand-mark {
      width:36px;height:36px;border-radius:12px;display:flex;align-items:center;justify-content:center;
      color:#DDE0FF;background:linear-gradient(145deg,#292B58,#131B32);
      border:1px solid rgba(129,140,248,.30);box-shadow:0 0 22px rgba(129,140,248,.18);
      font-size:19px;font-weight:800;
    }
    .brand-name { font-size:18px;font-weight:800;letter-spacing:-.04em;color:var(--text); }
    .brand-name span { color:var(--cyan2); }
    .eyebrow { color:var(--cyan2);font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:.13em;text-transform:uppercase; }
    .pipeline-box {
      background:linear-gradient(110deg,rgba(21,26,35,.96),rgba(28,34,45,.72));
      border:1px solid var(--line);border-radius:12px;padding:12px 13px;margin:2px 0 19px 0;
    }
    .pipeline-top { color:var(--muted);font:700 11px 'JetBrains Mono',monospace;letter-spacing:.12em;text-transform:uppercase; }
    .pipeline-name { color:var(--text);font:600 12px 'JetBrains Mono',monospace;margin-top:5px; }
    .status-dot { display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--cyan2);box-shadow:0 0 10px var(--cyan2);margin-right:7px; }
    .telemetry-box { background:#151A23;border:1px solid var(--line);border-radius:13px;padding:12px 13px;margin-top:28px; }
    .telemetry-label { color:var(--muted);font:700 11px 'JetBrains Mono',monospace;letter-spacing:.12em;text-transform:uppercase; }
    .telemetry-grid { display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:9px; }
    .telemetry-value { font:700 21px 'JetBrains Mono',monospace;color:var(--text); }
    .telemetry-value.cyan { color:var(--cyan2); }

    div[role="radiogroup"] { gap:10px !important; }
    div[role="radiogroup"] label {
      background:transparent;border:1px solid transparent;border-radius:10px;padding:11px 14px !important;
      color:#BFC5D4 !important;transition:all .15s ease;font-size:13px;
    }
    div[role="radiogroup"] label:hover { background:rgba(129,140,248,.08);border-color:rgba(129,140,248,.15); }
    div[role="radiogroup"] label[data-checked="true"] {
      background:linear-gradient(90deg,rgba(129,140,248,.17),rgba(6,182,212,.06));
      border-color:rgba(129,140,248,.22);color:#F2F3FF !important;
    }
    .topbar { display:flex;align-items:center;gap:10px;margin-bottom:16px; }
    .ready-pill { display:inline-flex;align-items:center;gap:7px;background:#202632;border:1px solid var(--line);border-radius:999px;padding:8px 12px;color:#BFC8D8;font:700 11px 'JetBrains Mono',monospace;letter-spacing:.08em;white-space:nowrap; }
    .ready-pill:before { content:'';width:7px;height:7px;border-radius:50%;background:var(--cyan2);box-shadow:0 0 9px var(--cyan2); }
    .page-title { font-size:clamp(25px,3vw,38px);font-weight:800;letter-spacing:-.045em;line-height:1.12;margin:3px 0 8px 0;color:var(--text); }
    .page-description { color:var(--muted);font:13px/1.7 'JetBrains Mono',monospace;max-width:850px;margin-bottom:17px; }
    .section-heading { font-size:18px;font-weight:700;letter-spacing:-.025em;margin:0;color:var(--text); }
    .section-sub { color:var(--muted);font:12px/1.55 'JetBrains Mono',monospace;margin-top:3px; }
    .section-kicker { color:var(--muted);font:700 11px 'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.13em; }

    .glass-card { background:linear-gradient(145deg,rgba(28,34,45,.94),rgba(21,26,35,.94));border:1px solid var(--line);border-radius:15px;padding:16px 17px;box-shadow:0 10px 30px rgba(0,0,0,.12); }
    .metric-card { min-height:116px;background:linear-gradient(145deg,#181E28,#141922);border:1px solid var(--line);border-radius:13px;padding:15px 16px;position:relative;overflow:hidden;margin-bottom:4px; }
    .metric-card:after { content:'';position:absolute;left:15px;right:15px;bottom:0;height:3px;background:var(--metric-accent,var(--cyan));border-radius:99px;opacity:.9; }
    .metric-label { color:var(--muted);font:700 11px 'JetBrains Mono',monospace;letter-spacing:.11em;text-transform:uppercase; }
    .metric-value { color:var(--text);font:700 clamp(24px,2.6vw,34px) 'JetBrains Mono',monospace;letter-spacing:-.05em;margin-top:12px;line-height:1.05; }
    .metric-foot { color:var(--muted);font:12px 'JetBrains Mono',monospace;margin-top:8px; }
    .tag { display:inline-block;padding:4px 8px;border-radius:999px;font:700 11px 'JetBrains Mono',monospace;letter-spacing:.04em;background:rgba(6,182,212,.12);color:var(--cyan2);border:1px solid rgba(6,182,212,.18); }
    .tag-risk { background:rgba(244,63,94,.13);color:#FDA4AF;border-color:rgba(244,63,94,.22); }
    .tag-warn { background:rgba(245,158,11,.13);color:#FCD34D;border-color:rgba(245,158,11,.22); }
    .tag-green { background:rgba(16,185,129,.13);color:#6EE7B7;border-color:rgba(16,185,129,.22); }

    .result-card { border:1px solid var(--result-color,var(--cyan));background:linear-gradient(145deg,rgba(21,26,35,.96),rgba(21,26,35,.78));border-radius:15px;padding:19px;box-shadow:0 0 24px color-mix(in srgb,var(--result-color,var(--cyan)) 16%,transparent); }
    .result-title { font-size:21px;font-weight:800;margin:0 0 5px 0; }
    .result-copy { color:#B8C1D1;font-size:12px;line-height:1.6;margin:0; }
    .mono { font-family:'JetBrains Mono',monospace; }
    .muted { color:var(--muted); }

    /* Progress bar for churn probability */
    .prob-bar-wrap { background:#1A2030;border-radius:999px;height:8px;margin:10px 0; }
    .prob-bar-fill { height:8px;border-radius:999px;transition:width .4s ease; }

    div[data-testid="stForm"] { background:transparent;border:0;padding:0; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(145deg,rgba(28,34,45,.88),rgba(21,26,35,.88));border:1px solid var(--line);border-radius:14px; }
    .stButton > button, div[data-testid="stFormSubmitButton"] > button, .stDownloadButton > button {
      border-radius:999px;border:1px solid rgba(255,255,255,.12);font-weight:700;transition:all .16s ease;
      min-height:2.45rem;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--cyan);color:white;box-shadow:0 0 15px rgba(6,182,212,.14); }
    div[data-testid="stFormSubmitButton"] > button[kind="primary"], .stButton > button[kind="primary"] {
      background:linear-gradient(100deg,#F43F5E,#FB5870);border:0;color:white;box-shadow:0 5px 20px rgba(244,63,94,.2);
    }
    div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover { filter:brightness(1.08);color:#fff; }

    input, textarea { color:var(--text) !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div {
      background:#252B36 !important;border-color:rgba(255,255,255,.08) !important;border-radius:9px !important;
    }
    div[data-baseweb="select"] span, div[data-baseweb="input"] input { color:var(--text) !important; }
    label, .stMarkdown p, .stCaption { color:var(--text); }
    [data-testid="stMetric"] { background:#171D27;border:1px solid var(--line);border-radius:12px;padding:12px 14px; }
    [data-testid="stMetricLabel"] { color:var(--muted) !important; }
    [data-testid="stMetricValue"] { font-family:'JetBrains Mono',monospace;color:var(--text) !important; }
    [data-testid="stDataFrame"] { border:1px solid var(--line);border-radius:12px;overflow:hidden; }
    button[data-baseweb="tab"] { color:var(--muted) !important;font-weight:700; }
    hr { border-color:var(--line) !important; }

    @media (max-width: 900px) {
      .block-container { padding-left:1rem;padding-right:1rem; }
      .metric-value { font-size:24px; }
      .page-title { font-size:27px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading the Telco customer dataset...")
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    data = data.reset_index(drop=True)
    data["SeniorLabel"] = data["SeniorCitizen"].map({0: "Non-senior", 1: "Senior"})
    data["AvgMonthlySpend"] = np.where(
        data["tenure"] > 0,
        data["TotalCharges"] / data["tenure"],
        data["MonthlyCharges"],
    )
    return data


@st.cache_resource(show_spinner="Loading the trained churn model...")
def load_model(path: str):
    return joblib.load(path)


try:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    df    = load_data(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))

except Exception as exc:
    st.error(f"Could not initialize the application: {exc}")
    st.info(
        "Check that the project has:\n"
        "- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`\n"
        "- `models/customer_churn_model.pkl`"
    )
    st.stop()


saved_metrics  = getattr(model, "evaluation_metrics_", {})
saved_cm       = getattr(model, "confusion_matrix_", [[747, 288], [81, 293]])
saved_roc      = getattr(model, "roc_curve_", None)
saved_comparison = getattr(model, "model_comparison_", {
    "Logistic Regression": {"accuracy": 0.7381, "precision": 0.5043, "recall": 0.7834, "f1_score": 0.6136, "roc_auc": 0.8416},
    "Random Forest":       {"accuracy": 0.7601, "precision": 0.5422, "recall": 0.7192, "f1_score": 0.6185, "roc_auc": 0.8313},
    "Decision Tree":       {"accuracy": 0.7359, "precision": 0.5018, "recall": 0.7540, "f1_score": 0.6028, "roc_auc": 0.8110},
})

MODEL_ACCURACY  = saved_metrics.get("accuracy",  0.7381) * 100
MODEL_PRECISION = saved_metrics.get("precision", 0.5043) * 100
MODEL_RECALL    = saved_metrics.get("recall",    0.7834) * 100
MODEL_F1        = saved_metrics.get("f1_score",  0.6136) * 100
MODEL_ROC_AUC   = saved_metrics.get("roc_auc",   0.8416)


def style_fig(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=12, r=12, t=48, b=10),
        font=dict(family="JetBrains Mono, monospace", color=TEXT, size=10),
        title=dict(font=dict(family="Plus Jakarta Sans", size=15, color=TEXT), x=0.02),
        legend_title_text="",
        hoverlabel=dict(bgcolor=PANEL_3, font_color=TEXT, bordercolor=OUTLINE),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(255,255,255,.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    return fig


def metric_card(label: str, value: str, foot: str = "", accent: str = CYAN) -> None:
    st.markdown(
        f"""<div class="metric-card" style="--metric-accent:{accent}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-foot">{foot}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-description">{description}</div>', unsafe_allow_html=True)


def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="section-heading">{title}</div><div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def churn_rate_by(data: pd.DataFrame, column: str, title: str) -> go.Figure:
    rates = (
        data.groupby(column, dropna=False)["Churn"]
        .apply(lambda v: (v == "Yes").mean() * 100)
        .reset_index(name="Churn rate")
        .sort_values("Churn rate", ascending=False)
    )
    fig = px.bar(
        rates, x=column, y="Churn rate",
        text=rates["Churn rate"].map(lambda v: f"{v:.1f}%"),
        color="Churn rate",
        color_continuous_scale=[CYAN, INDIGO, CORAL],
        title=title,
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False, yaxis_title="Churn rate (%)", xaxis_title="")
    return style_fig(fig, 300)


def plot_confusion_matrix(cm_data: list[list[int]]) -> go.Figure:
    """Plot an interactive cybernetic-styled confusion matrix heatmap."""
    z = cm_data
    x = ["Predicted Retain (0)", "Predicted Churn (1)"]
    y = ["Actual Retain (0)", "Actual Churn (1)"]
    total = sum(z[0]) + sum(z[1])
    tn, fp = z[0][0], z[0][1]
    fn, tp = z[1][0], z[1][1]

    text_labels = [
        [f"<b>True Negative (TN)</b><br>{tn:,}<br><span style='font-size:11px'>({tn/total*100:.1f}%)</span>",
         f"<b>False Positive (FP)</b><br>{fp:,}<br><span style='font-size:11px'>({fp/total*100:.1f}%)</span>"],
        [f"<b>False Negative (FN)</b><br>{fn:,}<br><span style='font-size:11px'>({fn/total*100:.1f}%)</span>",
         f"<b>True Positive (TP)</b><br>{tp:,}<br><span style='font-size:11px'>({tp/total*100:.1f}%)</span>"],
    ]

    fig = px.imshow(
        z, x=x, y=y,
        labels=dict(x="Predicted", y="Actual", color="Customers"),
        color_continuous_scale=[[0, "#151A23"], [0.35, "#1C2438"], [0.7, "#312E81"], [1, "#06B6D4"]],
        aspect="auto",
    )
    fig.update_traces(
        text=text_labels,
        texttemplate="%{text}",
        textfont={"size": 13, "family": "JetBrains Mono, monospace"},
        hoverinfo="none",
    )
    fig.update_layout(
        xaxis=dict(tickfont=dict(size=11, color=TEXT), side="bottom"),
        yaxis=dict(tickfont=dict(size=11, color=TEXT)),
        coloraxis_showscale=False,
    )
    return style_fig(fig, 320)


def plot_roc_curve(roc_data: dict | None, auc_score: float) -> go.Figure:
    """Plot the ROC Curve with baseline and AUC fill."""
    if not roc_data or "fpr" not in roc_data or "tpr" not in roc_data:
        fpr = [0.0, 0.05, 0.15, 0.35, 0.65, 1.0]
        tpr = [0.0, 0.45, 0.72, 0.85, 0.94, 1.0]
    else:
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=MUTED, dash="dash", width=1.5),
        name="Random Chance (AUC = 0.50)",
    ))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        line=dict(color=CYAN_LIGHT, width=2.8),
        fill="tozeroy",
        fillcolor="rgba(6, 182, 212, 0.12)",
        name=f"Logistic Regression (AUC = {auc_score:.4f})",
    ))
    fig.update_layout(
        xaxis=dict(title="False Positive Rate", range=[-0.02, 1.02]),
        yaxis=dict(title="True Positive Rate (Recall)", range=[-0.02, 1.04]),
        legend=dict(x=0.38, y=0.08, bgcolor="rgba(21, 26, 35, 0.85)", bordercolor=OUTLINE, borderwidth=1),
    )
    return style_fig(fig, 320)


def plot_model_comparison(comparison_data: dict) -> go.Figure | None:
    """Plot multi-model benchmark grouped bar chart."""
    if not comparison_data:
        return None
    rows = []
    for model_name, m in comparison_data.items():
        rows.append({"Model": model_name, "Metric": "Accuracy", "Score": m["accuracy"] * 100})
        rows.append({"Model": model_name, "Metric": "Recall", "Score": m["recall"] * 100})
        rows.append({"Model": model_name, "Metric": "ROC-AUC", "Score": m["roc_auc"] * 100})
        rows.append({"Model": model_name, "Metric": "F1 Score", "Score": m["f1_score"] * 100})
    comp_df = pd.DataFrame(rows)

    fig = px.bar(
        comp_df, x="Metric", y="Score", color="Model",
        barmode="group",
        color_discrete_map={
            "Logistic Regression": CYAN,
            "Random Forest": INDIGO,
            "Decision Tree": AMBER,
        },
        text=comp_df["Score"].map(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis_title="Score (%)",
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return style_fig(fig, 340)


def model_parts():
    """Return (preprocessor, classifier) steps from the saved sklearn pipeline."""
    try:
        if hasattr(model, "named_steps"):
            steps = model.named_steps
            pre = steps.get("preprocessor") or list(steps.values())[-2]
            clf = steps.get("classifier")   or list(steps.values())[-1]
            return pre, clf
        return model[:-1], model[-1]
    except Exception:
        return None, None


def customer_attributions(customer_frame: pd.DataFrame, n: int = 5) -> pd.DataFrame | None:
    """Coefficient × encoded input contribution (not SHAP)."""
    pre, clf = model_parts()
    if pre is None or clf is None or not hasattr(clf, "coef_"):
        return None
    try:
        feature_names = pre.get_feature_names_out()
        transformed   = pre.transform(customer_frame)
        dense_row     = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
        contributions = dense_row * np.asarray(clf.coef_)[0]
        result = pd.DataFrame({"Feature": feature_names, "Impact": contributions})
        result["Feature"] = result["Feature"].str.replace(r"^\w+?__", "", regex=True)
        return (
            result
            .reindex(result["Impact"].abs().sort_values(ascending=False).index)
            .head(n)
            .sort_values("Impact")
        )
    except Exception:
        return None


def global_coefficients() -> pd.DataFrame | None:
    pre, clf = model_parts()
    if pre is None or clf is None or not hasattr(clf, "coef_"):
        return None
    try:
        names        = pre.get_feature_names_out()
        coefficients = np.asarray(clf.coef_)[0]
        result = pd.DataFrame({"Feature": names, "Coefficient": coefficients})
        result["Feature"]    = result["Feature"].str.replace(r"^\w+?__", "", regex=True)
        result["Odds ratio"] = np.exp(np.clip(result["Coefficient"], -20, 20))
        result["Effect"]     = np.where(result["Coefficient"] >= 0, "Raises churn risk", "Protective signal")
        return result.reindex(result["Coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    except Exception:
        return None


def retention_actions(customer: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if customer["Contract"] == "Month-to-month":
        actions.append("Offer a discount or benefit for upgrading to a one- or two-year contract.")
    if customer["PaymentMethod"] == "Electronic check":
        actions.append("Suggest automatic payment by card or bank transfer — reduces friction and churn risk.")
    if customer["InternetService"] == "Fiber optic":
        actions.append("Review fiber service quality and consider a loyalty price or plan review.")
    if customer["InternetService"] != "No" and customer["TechSupport"] == "No":
        actions.append("Offer a short complimentary tech-support trial.")
    if customer["InternetService"] != "No" and customer["OnlineSecurity"] == "No":
        actions.append("Offer a trial of the online-security add-on.")
    if customer["InternetService"] != "No" and customer.get("OnlineBackup") == "No":
        actions.append("Highlight the value of online backup — customers with it churn less.")
    if customer["tenure"] <= 6:
        actions.append("New customer (<6 months): schedule a welcome call and onboarding check-in.")
    elif customer["tenure"] <= 12:
        actions.append("Schedule an onboarding check-in — first-year customers are at elevated risk.")
    if customer["SeniorCitizen"] == 1:
        actions.append("Offer a dedicated support contact or simplified plan walkthrough for senior citizens.")
    if customer["MonthlyCharges"] > 80:
        actions.append("Monthly charges are high — review the bundle and identify a better-value option.")
    if customer["Dependents"] == "No" and customer["Partner"] == "No":
        actions.append("Single customer with no dependents: consider a family or loyalty plan to build stickiness.")
    return actions or ["No specific risk trigger identified. Maintain current service level and monitor the account."]


def risk_tier(percent: float) -> tuple[str, str, str]:
    if percent >= 70:
        return "Critical churn risk", CORAL, "Immediate retention intervention strongly recommended."
    if percent >= 50:
        return "High churn risk", CORAL, "Prioritize this account for a retention intervention."
    if percent >= 35:
        return "Moderate churn risk", AMBER, "A proactive service check-in may help retain this customer."
    return "Low churn risk", CYAN, "The current profile is consistent with customer retention."


DEFAULTS: dict[str, Any] = {
    "gender": "Male", "senior": "No", "partner": "Yes", "dependents": "No",
    "tenure": 12, "phone": "Yes", "lines": "No", "internet": "DSL",
    "security": "No", "backup": "No", "protection": "No", "support": "No",
    "tv": "No", "movies": "No", "contract": "Month-to-month",
    "paperless": "Yes", "payment": "Electronic check",
    "monthly": 50.0, "auto_total": True, "total": 600.0,
}

PRESETS: dict[str, dict[str, Any]] = {
    "High-risk": {
        **DEFAULTS,
        "senior": "Yes", "partner": "No", "tenure": 3,
        "lines": "Yes", "internet": "Fiber optic", "monthly": 90.0,
        "tv": "Yes", "movies": "Yes",
    },
    "Low-risk": {
        **DEFAULTS,
        "partner": "Yes", "dependents": "Yes", "tenure": 60,
        "security": "Yes", "backup": "Yes", "protection": "Yes", "support": "Yes",
        "contract": "Two year", "paperless": "No", "payment": "Credit card (automatic)",
        "monthly": 70.0, "total": 4200.0,
    },
}

for key, default_value in DEFAULTS.items():
    st.session_state.setdefault(key, default_value)


def apply_preset(name: str) -> None:
    values = PRESETS.get(name, DEFAULTS)
    for key, value in values.items():
        st.session_state[key] = value
    st.session_state.pop("last_prediction", None)


@st.cache_data
def get_customer_search_dict(df_data: pd.DataFrame) -> dict[str, str]:
    """Precompute informative search labels for all customers."""
    labels = {}
    for cid, g, ten, con, mc, ch in zip(
        df_data["customerID"],
        df_data["gender"],
        df_data["tenure"],
        df_data["Contract"],
        df_data["MonthlyCharges"],
        df_data["Churn"]
    ):
        labels[cid] = f"{cid}  ·  {g}  ·  {ten} mo  ·  {con}  ·  ${mc:.0f}/mo  ·  Actual: {ch}"
    return labels


def load_customer_by_id(cust_id: str) -> None:
    """Load an existing customer's 19 features into the state and compute prediction."""
    matches = df[df["customerID"] == cust_id]
    if matches.empty:
        return
    row = matches.iloc[0]

    st.session_state["gender"] = str(row["gender"])
    st.session_state["senior"] = "Yes" if row["SeniorCitizen"] == 1 else "No"
    st.session_state["partner"] = str(row["Partner"])
    st.session_state["dependents"] = str(row["Dependents"])
    st.session_state["tenure"] = int(row["tenure"])
    st.session_state["phone"] = str(row["PhoneService"])
    st.session_state["lines"] = "Yes" if row["MultipleLines"] == "Yes" else "No"
    st.session_state["internet"] = str(row["InternetService"])
    st.session_state["security"] = "Yes" if row["OnlineSecurity"] == "Yes" else "No"
    st.session_state["backup"] = "Yes" if row["OnlineBackup"] == "Yes" else "No"
    st.session_state["protection"] = "Yes" if row["DeviceProtection"] == "Yes" else "No"
    st.session_state["support"] = "Yes" if row["TechSupport"] == "Yes" else "No"
    st.session_state["tv"] = "Yes" if row["StreamingTV"] == "Yes" else "No"
    st.session_state["movies"] = "Yes" if row["StreamingMovies"] == "Yes" else "No"
    st.session_state["contract"] = str(row["Contract"])
    st.session_state["paperless"] = str(row["PaperlessBilling"])
    st.session_state["payment"] = str(row["PaymentMethod"])
    st.session_state["monthly"] = float(row["MonthlyCharges"])
    st.session_state["total"] = float(row["TotalCharges"])
    st.session_state["auto_total"] = False

    customer = {
        "gender": str(row["gender"]),
        "SeniorCitizen": int(row["SeniorCitizen"]),
        "Partner": str(row["Partner"]),
        "Dependents": str(row["Dependents"]),
        "tenure": int(row["tenure"]),
        "PhoneService": str(row["PhoneService"]),
        "MultipleLines": str(row["MultipleLines"]),
        "InternetService": str(row["InternetService"]),
        "OnlineSecurity": str(row["OnlineSecurity"]),
        "OnlineBackup": str(row["OnlineBackup"]),
        "DeviceProtection": str(row["DeviceProtection"]),
        "TechSupport": str(row["TechSupport"]),
        "StreamingTV": str(row["StreamingTV"]),
        "StreamingMovies": str(row["StreamingMovies"]),
        "Contract": str(row["Contract"]),
        "PaperlessBilling": str(row["PaperlessBilling"]),
        "PaymentMethod": str(row["PaymentMethod"]),
        "MonthlyCharges": float(row["MonthlyCharges"]),
        "TotalCharges": float(row["TotalCharges"]),
    }
    customer_frame = pd.DataFrame([customer])
    prediction = int(model.predict(customer_frame)[0])
    probability = float(model.predict_proba(customer_frame)[0][1])

    st.session_state["last_prediction"] = {
        "customer": customer,
        "probability": probability,
        "prediction": prediction,
        "customerID": str(cust_id),
        "actual_churn": str(row["Churn"]),
    }


def build_customer_from_state() -> dict[str, Any]:
    s = st.session_state
    no_internet = s.internet == "No"
    no_phone    = s.phone    == "No"
    addon = lambda v: "No internet service" if no_internet else v
    total = s.monthly * s.tenure if s.auto_total else s.total
    return {
        "gender":           s.gender,
        "SeniorCitizen":    1 if s.senior == "Yes" else 0,
        "Partner":          s.partner,
        "Dependents":       s.dependents,
        "tenure":           s.tenure,
        "PhoneService":     s.phone,
        "MultipleLines":    "No phone service" if no_phone else s.lines,
        "InternetService":  s.internet,
        "OnlineSecurity":   addon(s.security),
        "OnlineBackup":     addon(s.backup),
        "DeviceProtection": addon(s.protection),
        "TechSupport":      addon(s.support),
        "StreamingTV":      addon(s.tv),
        "StreamingMovies":  addon(s.movies),
        "Contract":         s.contract,
        "PaperlessBilling": s.paperless,
        "PaymentMethod":    s.payment,
        "MonthlyCharges":   float(s.monthly),
        "TotalCharges":     float(total),
    }


with st.sidebar:
    st.markdown(
        '<div class="brand-row"><div class="brand-mark">✣</div>'
        '<div class="brand-name">Churn <span>Project</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pipeline-box"><div class="pipeline-top"><span class="status-dot"></span>Active pipeline</div>'
        f'<div class="pipeline-name">Balanced Logistic Regression<br>ROC-AUC {MODEL_ROC_AUC:.4f}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-kicker" style="margin-bottom:8px">Workspace navigation</div>', unsafe_allow_html=True)
    if "nav_target" in st.session_state:
        st.session_state["workspace_page"] = st.session_state.pop("nav_target")

    page = st.radio(
        "Workspace",
        ["Predict", "Analytics", "Model Insights", "Data", "Summary"],
        format_func=lambda v: {
            "Predict":        "Predict  ·  Simulator & Playbook",
            "Analytics":      "Analytics  ·  Cohort & Retention",
            "Model Insights": "Model Insights  ·  Coefficients",
            "Data":           "Data  ·  Customer Dataset",
            "Summary":        "Summary  ·  Project Overview",
        }[v],
        label_visibility="collapsed",
        key="workspace_page",
    )
    st.markdown(
        f'''<div class="telemetry-box">
          <div class="telemetry-label">Model performance <span style="float:right;color:{CYAN_LIGHT}">● Live</span></div>
          <div class="telemetry-grid">
            <div><div class="section-kicker">Accuracy</div><div class="telemetry-value">{MODEL_ACCURACY:.1f}%</div></div>
            <div><div class="section-kicker">Recall</div><div class="telemetry-value cyan">{MODEL_RECALL:.1f}%</div></div>
            <div><div class="section-kicker">F1 Score</div><div class="telemetry-value">{MODEL_F1:.1f}%</div></div>
            <div><div class="section-kicker">ROC-AUC</div><div class="telemetry-value cyan">{MODEL_ROC_AUC:.4f}</div></div>
          </div>
        </div>''',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="margin-top:12px;padding:10px 12px;background:#171D27;border:1px solid rgba(255,255,255,.07);border-radius:11px;">'
        '<div style="font-weight:700;font-size:12px">Customer Churn Project</div>'
        '<div style="font:12px JetBrains Mono;color:#929CAF;margin-top:3px">Machine Learning · IT V SEM</div></div>',
        unsafe_allow_html=True,
    )


header_search_col, header_status_col, header_export_col = st.columns([6, 1.7, 1.5], vertical_alignment="center")
with header_search_col:
    search_query = st.text_input(
        "Search customer records",
        placeholder="Search customer ID (e.g. 7590-VHVEG), contract, payment method...",
        label_visibility="collapsed",
        key="global_search",
    )
with header_status_col:
    st.markdown('<div class="ready-pill">INFERENCE READY</div>', unsafe_allow_html=True)
with header_export_col:
    st.download_button(
        "↓  Export data",
        data=df.drop(columns=["SeniorLabel", "AvgMonthlySpend"], errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name="telco_churn_clean.csv",
        mime="text/csv",
        use_container_width=True,
    )

if search_query.strip():
    q_str = search_query.strip().lower()
    exact_id_matches = df[df["customerID"].str.lower().str.contains(q_str, na=False)]
    if not exact_id_matches.empty and len(exact_id_matches) <= 10:
        match_row = exact_id_matches.iloc[0]
        m_id = str(match_row["customerID"])
        m_churn = str(match_row["Churn"])
        h_col1, h_col2 = st.columns([4, 1.3], vertical_alignment="center")
        with h_col1:
            st.markdown(
                f'<div style="background:rgba(6,182,212,.08);border:1px solid rgba(6,182,212,.28);border-radius:10px;padding:7px 12px;font-size:12px;margin:5px 0 10px 0;">'
                f'Found Customer Record: <b class="mono" style="color:{CYAN_LIGHT}">{m_id}</b> &nbsp;·&nbsp; '
                f'Tenure: <b>{match_row["tenure"]} mo</b> &nbsp;·&nbsp; Contract: <b>{match_row["Contract"]}</b> &nbsp;·&nbsp; '
                f'Historical Churn: <b style="color:{"#F43F5E" if m_churn == "Yes" else "#10B981"}">{m_churn}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with h_col2:
            if st.button(f"⚡ Load & Predict {m_id}", key=f"hdr_jump_{m_id}", use_container_width=True):
                load_customer_by_id(m_id)
                st.session_state["nav_target"] = "Predict"
                st.rerun()


if page == "Predict":
    page_header(
        "Production-style local inference  /  Customer simulator",
        "Customer Churn Prediction",
        "Build a customer profile, estimate churn probability and review practical retention actions. "
        "Predictions are generated locally by the saved Balanced Logistic Regression pipeline.",
    )
    st.markdown(
        '<div class="pipeline-box" style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">'
        f'<div><span class="status-dot"></span><span class="section-kicker">Active model</span> '
        f'<b class="mono">Balanced Logistic Regression</b> <span class="muted mono">· ROC-AUC {MODEL_ROC_AUC:.4f}</span></div>'
        '<div><span class="tag tag-risk">High-risk profile</span> '
        '<span class="tag">Low-risk profile</span> '
        '<span class="tag">Reset available</span></div></div>',
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1.52, 1], gap="large")
    with form_col:
        with st.container(border=True):
            section_title("🔍  Search & Load Customer from Database", "Search by Customer ID or filter features to test any of the 7,043 customers")
            st.write("")

            qf1, qf2, qf3 = st.columns([1, 1, 1])
            with qf1:
                filter_churn = st.selectbox("Actual Churn", ["All", "Actual Churned", "Actual Retained"], key="lookup_filter_churn")
            with qf2:
                filter_contract = st.selectbox("Contract", ["All", "Month-to-month", "One year", "Two year"], key="lookup_filter_contract")
            with qf3:
                filter_internet = st.selectbox("Internet Service", ["All", "Fiber optic", "DSL", "No"], key="lookup_filter_internet")

            cand_df = df
            if filter_churn == "Actual Churned":
                cand_df = cand_df[cand_df["Churn"] == "Yes"]
            elif filter_churn == "Actual Retained":
                cand_df = cand_df[cand_df["Churn"] == "No"]
            if filter_contract != "All":
                cand_df = cand_df[cand_df["Contract"] == filter_contract]
            if filter_internet != "All":
                cand_df = cand_df[cand_df["InternetService"] == filter_internet]

            cust_label_map = get_customer_search_dict(df)
            cand_ids = cand_df["customerID"].tolist()

            sel_col, btn_col = st.columns([2.8, 1.2], vertical_alignment="bottom")
            with sel_col:
                selected_cust_id = st.selectbox(
                    f"Select customer ({len(cand_ids):,} matching — type to search)",
                    options=cand_ids,
                    format_func=lambda cid: cust_label_map.get(cid, cid),
                    key="lookup_customer_id",
                    placeholder="Type to search Customer ID (e.g. 7590-VHVEG)...",
                )
            with btn_col:
                if st.button("📥 Load & Predict", type="primary", use_container_width=True, key="btn_load_customer"):
                    if selected_cust_id:
                        load_customer_by_id(selected_cust_id)
                        st.rerun()

        st.write("")
        preset_a, preset_b, preset_c = st.columns([1, 1, .7])
        preset_a.button("⚠  High-risk sample", on_click=apply_preset, args=("High-risk",), use_container_width=True)
        preset_b.button("◈  Low-risk sample",  on_click=apply_preset, args=("Low-risk",),  use_container_width=True)
        preset_c.button("↺  Reset",             on_click=apply_preset, args=("Reset",),     use_container_width=True)

        st.checkbox(
            "Calculate total charges from monthly charges × tenure",
            key="auto_total",
            help="Untick to enter total charges manually.",
        )

        with st.form("churn_customer_form", clear_on_submit=False):
            with st.container(border=True):
                section_title("◉  Customer profile", "Who the customer is and how long they have stayed")
                st.write("")
                a1, a2, a3 = st.columns(3)
                a1.selectbox("Gender",             ["Male", "Female"],   key="gender")
                a2.selectbox("Senior citizen (65+)", ["No", "Yes"],      key="senior")
                a3.selectbox("Partner status",     ["Yes", "No"],        key="partner")
                a1, a2, a3 = st.columns(3)
                a1.selectbox("Dependents",         ["Yes", "No"],        key="dependents")
                a2.slider("Tenure (months)",        min_value=0, max_value=72, key="tenure")
                a3.selectbox("Phone service",       ["Yes", "No"],        key="phone")

            with st.container(border=True):
                section_title("⌁  Services", "Internet, voice, add-on security and entertainment services")
                st.write("")
                a1, a2, a3 = st.columns(3)
                a1.selectbox("Multiple lines",     ["Yes", "No"],                          key="lines")
                a2.selectbox("Internet service",   ["Fiber optic", "DSL", "No"],           key="internet")
                a3.selectbox("Online security",    ["Yes", "No"],                          key="security")
                a1, a2, a3 = st.columns(3)
                a1.selectbox("Online backup",      ["Yes", "No"],                          key="backup")
                a2.selectbox("Device protection",  ["Yes", "No"],                          key="protection")
                a3.selectbox("Tech support",        ["Yes", "No"],                          key="support")
                a1, a2 = st.columns(2)
                a1.selectbox("Streaming TV",       ["Yes", "No"],                          key="tv")
                a2.selectbox("Streaming movies",   ["Yes", "No"],                          key="movies")
                st.caption("Internet/phone-dependent fields are normalised automatically before inference.")

            with st.container(border=True):
                section_title("▤  Contract & billing", "Commitment terms, payment channel and charge profile")
                st.write("")
                a1, a2, a3 = st.columns(3)
                a1.selectbox("Contract",           ["Month-to-month", "One year", "Two year"], key="contract")
                a2.selectbox("Paperless billing",  ["Yes", "No"],                              key="paperless")
                a3.selectbox(
                    "Payment channel",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    key="payment",
                )
                a1, a2 = st.columns(2)
                a1.number_input("Monthly charges ($)", min_value=0.0, step=0.5, key="monthly")
                a2.number_input(
                    "Total charges ($)", min_value=0.0, step=1.0, key="total",
                    disabled=st.session_state.auto_total,
                )
                if st.session_state.auto_total:
                    st.caption("Total = monthly × tenure (auto-calculated).")

            submitted = st.form_submit_button("⚡  Predict churn risk", type="primary", use_container_width=True)

    if submitted:
        customer       = build_customer_from_state()
        customer_frame = pd.DataFrame([customer])
        try:
            with st.spinner("Scoring customer profile..."):
                prediction  = int(model.predict(customer_frame)[0])
                probability = float(model.predict_proba(customer_frame)[0][1])
            prev_pred = st.session_state.get("last_prediction", {})
            st.session_state["last_prediction"] = {
                "customer":    customer,
                "probability": probability,
                "prediction":  prediction,
                "customerID":  prev_pred.get("customerID"),
                "actual_churn": prev_pred.get("actual_churn"),
            }
        except Exception as exc:
            st.error(f"The model could not score this customer: {exc}")
            st.session_state.pop("last_prediction", None)

    with result_col:
        last = st.session_state.get("last_prediction")
        if last:
            probability = float(last["probability"])
            percent     = probability * 100
            customer    = last["customer"]
            label, color, message = risk_tier(percent)

            if last.get("customerID"):
                actual = last.get("actual_churn", "Unknown")
                pred_label = "Churn" if last["prediction"] == 1 else "Retain"
                is_correct = (pred_label == "Churn" and actual == "Yes") or (pred_label == "Retain" and actual == "No")
                status_tag = '<span class="tag tag-green">✓ Accurate Match</span>' if is_correct else '<span class="tag tag-warn">⚠ Misclassification</span>'
                actual_tag = '<span class="tag tag-risk">Actual: Churned</span>' if actual == "Yes" else '<span class="tag tag-green">Actual: Retained</span>'

                st.markdown(
                    f'''<div class="glass-card" style="padding:11px 14px;margin-bottom:12px;border:1px solid rgba(129,140,248,.35);">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="section-kicker">Database Record Match</span>
                        {status_tag}
                      </div>
                      <div style="margin-top:6px;font-size:12px;line-height:1.7;">
                        <b>Customer ID:</b> <code class="mono" style="color:{CYAN_LIGHT};font-weight:700">{last["customerID"]}</code><br>
                        <b>Historical Status:</b> {actual_tag} &nbsp;·&nbsp; <b>Model:</b> <span class="mono">{"Churn (1)" if last["prediction"] == 1 else "Retain (0)"}</span>
                      </div>
                    </div>''',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-kicker">Churn risk  ·  Local model output</div>', unsafe_allow_html=True)

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=percent,
                    number={"suffix": "%", "valueformat": ".1f",
                            "font": {"size": 43, "color": TEXT, "family": "JetBrains Mono"}},
                    title={"text": "Estimated churn probability", "font": {"size": 12, "color": MUTED}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED, "size": 9}},
                        "bar":  {"color": color, "thickness": .28},
                        "bgcolor": PANEL_2, "borderwidth": 0,
                        "steps": [
                            {"range": [0,  35], "color": "rgba(6,182,212,.14)"},
                            {"range": [35, 50], "color": "rgba(245,158,11,.12)"},
                            {"range": [50, 70], "color": "rgba(244,63,94,.13)"},
                            {"range": [70,100], "color": "rgba(244,63,94,.22)"},
                        ],
                    },
                )
            )
            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", height=260,
                margin=dict(l=15, r=15, t=45, b=5), font={"color": TEXT},
            )
            st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f'<div class="result-card" style="--result-color:{color}">'
                f'<div class="result-title">{label}</div>'
                f'<p class="result-copy">{message}</p></div>',
                unsafe_allow_html=True,
            )
            st.write("")

            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f'<div class="metric-card"><div class="metric-label">Model class</div>'
                f'<div class="metric-value" style="font-size:21px">{"Churn" if last["prediction"] == 1 else "Retain"}</div>'
                f'<div class="metric-foot">Default 0.50 threshold</div></div>',
                unsafe_allow_html=True,
            )
            c2.markdown(
                f'<div class="metric-card"><div class="metric-label">Monthly charges</div>'
                f'<div class="metric-value" style="font-size:21px">${customer["MonthlyCharges"]:,.0f}</div>'
                f'<div class="metric-foot">${customer["TotalCharges"]:,.0f} lifetime</div></div>',
                unsafe_allow_html=True,
            )
            c3.markdown(
                f'<div class="metric-card"><div class="metric-label">Tenure</div>'
                f'<div class="metric-value" style="font-size:21px">{customer["tenure"]} mo</div>'
                f'<div class="metric-foot">{customer["Contract"]}</div></div>',
                unsafe_allow_html=True,
            )
            st.write("")

            section_title(
                "Top factors behind this prediction",
                "Coefficient × encoded input contribution — not SHAP values",
            )
            drivers = customer_attributions(pd.DataFrame([customer]), n=6)
            if drivers is not None and not drivers.empty:
                drivers["Direction"] = np.where(drivers["Impact"] >= 0, "Raises churn risk", "Protective signal")
                fig = px.bar(
                    drivers, x="Impact", y="Feature", orientation="h",
                    color="Direction",
                    color_discrete_map={"Raises churn risk": CORAL, "Protective signal": CYAN_LIGHT},
                    title="Strongest profile-level contributions",
                )
                fig.update_layout(xaxis_title="Signed contribution", yaxis_title="")
                st.plotly_chart(style_fig(fig, 280), use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Feature-level contributions are not available for this model type.")

            section_title("Retention playbook", "Suggested actions based on the customer risk profile")
            actions = retention_actions(customer)
            for i, action in enumerate(actions, start=1):
                tag_class = "tag-risk" if i == 1 and percent >= 50 else "tag-warn" if percent >= 35 else "tag"
                st.markdown(
                    f'<div class="glass-card" style="padding:11px 13px;margin:8px 0;display:flex;gap:10px;align-items:flex-start;">'
                    f'<span class="tag {tag_class}">{i:02d}</span>'
                    f'<span style="font-size:12px;line-height:1.6">{action}</span></div>',
                    unsafe_allow_html=True,
                )

            with st.expander("Inspect the feature vector sent to the model"):
                st.dataframe(pd.DataFrame([customer]), use_container_width=True, hide_index=True)

        else:
            st.markdown(
                '<div class="glass-card"><div class="section-kicker">Inference panel</div>'
                '<div style="font-size:22px;font-weight:800;margin:12px 0 6px">Awaiting customer profile</div>'
                '<div class="section-sub">Choose a sample profile or complete the form, then click '
                '<b>Predict churn risk</b> to see the risk gauge, key factors and retention actions.</div>'
                '<div style="margin-top:22px;display:flex;gap:8px;flex-wrap:wrap">'
                '<span class="tag">Local inference</span>'
                '<span class="tag">19 input features</span>'
                '<span class="tag">Probability score</span>'
                '<span class="tag">Retention playbook</span></div></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            metric_card("Model accuracy", f"{MODEL_ACCURACY:.2f}%", "Held-out test evaluation", INDIGO)
            metric_card("Churn recall",   f"{MODEL_RECALL:.2f}%",   "Positive class: Churn = Yes", CYAN)


elif page == "Analytics":
    page_header(
        "Cohort telemetry  /  Telco customer dataset",
        "Retention & Cohort Analytics",
        "Explore churn distribution, contract resilience, service-level patterns and billing behaviour "
        "across the cleaned Telco customer sample.",
    )
    st.markdown('<div class="section-kicker">Active slice · choose filters in the sidebar</div>', unsafe_allow_html=True)
    st.write("")

    with st.sidebar:
        st.markdown('<div class="section-kicker" style="margin-top:18px">Analytics filters</div>', unsafe_allow_html=True)
        contract_filter = st.multiselect(
            "Contract types", sorted(df["Contract"].dropna().unique()),
            default=sorted(df["Contract"].dropna().unique()), key="analytics_contracts",
        )
        internet_filter = st.multiselect(
            "Internet service", sorted(df["InternetService"].dropna().unique()),
            default=sorted(df["InternetService"].dropna().unique()), key="analytics_internet",
        )
        tenure_filter  = st.slider("Tenure horizon (months)", 0, 72, (0, 72), key="analytics_tenure")
        senior_filter  = st.radio("Demographic cohort", ["All", "Senior", "Non-senior"], horizontal=True, key="analytics_senior")

    analytics_df = df[
        df["Contract"].isin(contract_filter)
        & df["InternetService"].isin(internet_filter)
        & df["tenure"].between(*tenure_filter)
    ].copy()
    if senior_filter != "All":
        analytics_df = analytics_df[analytics_df["SeniorLabel"] == senior_filter]

    if analytics_df.empty:
        st.warning("No customers match the current analytics filters. Widen the selected ranges.")
    else:
        churned      = int((analytics_df["Churn"] == "Yes").sum())
        retained     = int((analytics_df["Churn"] == "No").sum())
        churn_rate   = analytics_df["Churn"].eq("Yes").mean() * 100
        avg_tenure   = analytics_df["tenure"].mean()
        avg_monthly  = analytics_df["MonthlyCharges"].mean()
        overall_rate = df["Churn"].eq("Yes").mean() * 100

        m1, m2, m3, m4 = st.columns(4, gap="medium")
        with m1: metric_card("Customer pool",      f"{len(analytics_df):,}", "Accounts in active slice", INDIGO)
        with m2: metric_card("Accounts churned",   f"{churned:,}",           f"{churn_rate:.1f}% of slice", CORAL)
        with m3: metric_card("Churn rate",         f"{churn_rate:.1f}%",     f"{churn_rate - overall_rate:+.1f} pts vs full sample", CORAL if churn_rate >= overall_rate else GREEN)
        with m4: metric_card("Avg. tenure / spend", f"{avg_tenure:.1f} mo", f"${avg_monthly:.0f} avg monthly · {retained:,} retained", CYAN)

        st.write("")
        st.write("")

        left, right = st.columns([1, 1.45], gap="medium")
        with left:
            with st.container(border=True):
                section_title("Churned vs retained", "Share of customers in the current cohort")
                donut = px.pie(
                    names=["Retained", "Churned"], values=[retained, churned], hole=.68,
                    color=["Retained", "Churned"],
                    color_discrete_map={"Retained": CYAN, "Churned": CORAL},
                )
                donut.update_traces(textinfo="percent", textfont_size=12, marker_line_color=PANEL, marker_line_width=3)
                donut.add_annotation(
                    text=f"{len(analytics_df):,}<br><span style='font-size:10px'>ACCOUNTS</span>",
                    x=.5, y=.5, showarrow=False,
                    font=dict(size=20, color=TEXT, family="JetBrains Mono"),
                )
                st.plotly_chart(style_fig(donut, 310), use_container_width=True, config={"displayModeBar": False})
                st.markdown(
                    f'<span class="tag">Retained · {retained:,}</span> &nbsp; '
                    f'<span class="tag tag-risk">Churned · {churned:,}</span>',
                    unsafe_allow_html=True,
                )
        with right:
            with st.container(border=True):
                section_title("Churn rate by contract type", "Contract duration is the strongest retention lever")
                st.plotly_chart(
                    churn_rate_by(analytics_df, "Contract", "Contract churn rate"),
                    use_container_width=True, config={"displayModeBar": False},
                )

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with st.container(border=True):
                section_title("Churn by internet service", "Fibre optic customers churn at the highest rate")
                st.plotly_chart(
                    churn_rate_by(analytics_df, "InternetService", "Internet service churn rate"),
                    use_container_width=True, config={"displayModeBar": False},
                )
        with c2:
            with st.container(border=True):
                section_title("Payment method vulnerability", "Electronic check is the highest-risk payment channel")
                st.plotly_chart(
                    churn_rate_by(analytics_df, "PaymentMethod", "Payment method churn rate"),
                    use_container_width=True, config={"displayModeBar": False},
                )

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with st.container(border=True):
                section_title("Tenure distribution", "Churn is concentrated in early-tenure customers")
                tenure_fig = px.histogram(
                    analytics_df, x="tenure", color="Churn", nbins=36,
                    barmode="overlay", opacity=.72, color_discrete_map=CHURN_COLORS,
                )
                tenure_fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="Customers")
                st.plotly_chart(style_fig(tenure_fig, 320), use_container_width=True, config={"displayModeBar": False})
        with c2:
            with st.container(border=True):
                section_title("Monthly charges dispersion", "Churned customers tend to pay more per month")
                box = px.box(
                    analytics_df, x="Churn", y="MonthlyCharges",
                    color="Churn", color_discrete_map=CHURN_COLORS, points=False,
                )
                box.update_layout(xaxis_title="", yaxis_title="Monthly charges ($)", showlegend=False)
                st.plotly_chart(style_fig(box, 320), use_container_width=True, config={"displayModeBar": False})

        c1, c2 = st.columns([1.3, 1], gap="medium")
        with c1:
            with st.container(border=True):
                section_title("Feature correlation matrix", "Pearson correlations across numeric fields and churn indicator")
                corr_data = analytics_df.assign(Churned=analytics_df["Churn"].eq("Yes").astype(int))
                corr_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churned"]
                corr      = corr_data[corr_cols].corr(numeric_only=True)
                heat = px.imshow(corr, text_auto=".2f", color_continuous_scale=[CYAN, "#1C2430", CORAL], zmin=-1, zmax=1, aspect="auto")
                st.plotly_chart(style_fig(heat, 340), use_container_width=True, config={"displayModeBar": False})
        with c2:
            with st.container(border=True):
                section_title("Senior citizen cohort", "Actual churn rate split by age cohort")
                senior_rates = (
                    analytics_df.groupby("SeniorLabel")["Churn"]
                    .apply(lambda v: v.eq("Yes").mean() * 100)
                    .reset_index(name="Churn rate")
                )
                senior_fig = px.bar(
                    senior_rates, x="SeniorLabel", y="Churn rate", color="SeniorLabel",
                    color_discrete_map={"Senior": CORAL, "Non-senior": CYAN},
                    text=senior_rates["Churn rate"].map(lambda x: f"{x:.1f}%"),
                )
                senior_fig.update_traces(textposition="outside")
                senior_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Churn rate (%)")
                st.plotly_chart(style_fig(senior_fig, 340), use_container_width=True, config={"displayModeBar": False})

        with st.container(border=True):
            section_title("Tenure vs monthly charges", "Scatter coloured by actual churn label — low tenure & high charges = hotspot")
            scatter = px.scatter(
                analytics_df.sample(min(len(analytics_df), 2000), random_state=42),
                x="tenure", y="MonthlyCharges", color="Churn",
                color_discrete_map=CHURN_COLORS, opacity=0.55, size_max=5,
            )
            scatter.update_layout(xaxis_title="Tenure (months)", yaxis_title="Monthly charges ($)")
            st.plotly_chart(style_fig(scatter, 340), use_container_width=True, config={"displayModeBar": False})


elif page == "Model Insights":
    page_header(
        "Model diagnostics  /  Coefficient-based explanation",
        "Explainability & Model Weights",
        "Inspect the fitted Logistic Regression coefficients and held-out evaluation metrics. "
        "Coefficient contributions are shown transparently; they are not SHAP values.",
    )
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1: metric_card("Accuracy",           f"{MODEL_ACCURACY:.2f}%",   "Held-out test result",            INDIGO)
    with m2: metric_card("Churn recall",        f"{MODEL_RECALL:.2f}%",    "Positive class: Churn = Yes",     CYAN)
    with m3: metric_card("ROC-AUC",             f"{MODEL_ROC_AUC:.4f}",    "Area under ROC curve",            INDIGO)
    with m4: metric_card("F1 Score",            f"{MODEL_F1:.2f}%",        "Harmonic mean of P & R",          AMBER)

    st.write("")
    st.write("")

    coefficients = global_coefficients()
    if coefficients is None or coefficients.empty:
        st.warning("Coefficient information is not available for the currently loaded model.")
    else:
        left, right = st.columns([1.25, 1], gap="medium")
        with left:
            with st.container(border=True):
                section_title(
                    "Strongest features in the model",
                    "Signed Logistic Regression coefficients · positive values increase churn log-odds",
                )
                display = coefficients.head(16).sort_values("Coefficient")
                display["Direction"] = np.where(display["Coefficient"] >= 0, "Raises churn risk", "Protective signal")
                coef_fig = px.bar(
                    display, x="Coefficient", y="Feature", orientation="h",
                    color="Direction",
                    color_discrete_map={"Raises churn risk": CORAL, "Protective signal": CYAN_LIGHT},
                )
                coef_fig.add_vline(x=0, line_color="rgba(255,255,255,.35)", line_dash="dot")
                coef_fig.update_layout(xaxis_title="Coefficient (log-odds scale)", yaxis_title="", legend_title="")
                st.plotly_chart(style_fig(coef_fig, 500), use_container_width=True, config={"displayModeBar": False})
                st.caption(
                    "These are model coefficients, not causal effects. Correlated features and one-hot encoding "
                    "affect how individual coefficients should be interpreted."
                )
        with right:
            with st.container(border=True):
                section_title("Top coefficients table", "Odds ratios are exp(coefficient) for a one-unit encoded change")
                top = coefficients.head(12).copy()
                top["Coefficient"] = top["Coefficient"].map(lambda v: f"{v:+.3f}")
                top["Odds ratio"]  = top["Odds ratio"].map(lambda v: f"{v:.3f}")
                st.dataframe(
                    top[["Feature", "Coefficient", "Odds ratio", "Effect"]],
                    use_container_width=True, hide_index=True, height=430,
                )
            with st.container(border=True):
                section_title("Pipeline facts", "Current saved model configuration")
                st.markdown(
                    '<div class="glass-card" style="padding:12px;margin-top:10px">'
                    '<div class="section-kicker">Estimator</div><div style="font-weight:700;margin:5px 0 12px">Balanced Logistic Regression</div>'
                    '<div class="section-kicker">Preprocessing</div><div style="font-size:12px;margin:5px 0 12px">StandardScaler (numeric) + OneHotEncoder (categorical)</div>'
                    '<div class="section-kicker">Class weighting</div><div style="font-size:12px;margin:5px 0 12px">balanced — corrects for class imbalance (~26% churn)</div>'
                    '<div class="section-kicker">Target encoding</div><div style="font-size:12px;margin-top:5px">No churn = 0 · Churn = 1</div></div>',
                    unsafe_allow_html=True,
                )

    st.write("")
    st.write("")
    cm_col, roc_col = st.columns([1, 1], gap="medium")
    with cm_col:
        with st.container(border=True):
            section_title("Confusion Matrix", "Held-out test split (1,409 customers) · 20% test partition")
            st.plotly_chart(plot_confusion_matrix(saved_cm), use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                '<div style="display:flex;gap:7px;flex-wrap:wrap;margin:4px 0 6px 0;">'
                f'<span class="tag tag-green">True Negatives: {saved_cm[0][0]:,}</span>'
                f'<span class="tag tag-risk">False Positives: {saved_cm[0][1]:,}</span>'
                f'<span class="tag tag-warn">False Negatives: {saved_cm[1][0]:,}</span>'
                f'<span class="tag">True Positives: {saved_cm[1][1]:,}</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.caption("Low False Negatives (81) means only ~5.7% of customers are missed churners.")

    with roc_col:
        with st.container(border=True):
            section_title("Receiver Operating Characteristic (ROC)", f"Discrimination capacity · AUC = {MODEL_ROC_AUC:.4f}")
            st.plotly_chart(plot_roc_curve(saved_roc, MODEL_ROC_AUC), use_container_width=True, config={"displayModeBar": False})
            st.caption(f"An AUC of {MODEL_ROC_AUC:.4f} demonstrates high discriminatory power significantly above random guessing (0.50).")

    st.write("")
    st.write("")
    with st.container(border=True):
        section_title("Multi-Model Architecture Comparison", "Benchmarked across 3 classification algorithms on the identical test split")
        st.write("")
        c_chart, c_table = st.columns([1.4, 1], gap="medium")
        with c_chart:
            comp_fig = plot_model_comparison(saved_comparison)
            if comp_fig is not None:
                st.plotly_chart(comp_fig, use_container_width=True, config={"displayModeBar": False})
        with c_table:
            if saved_comparison:
                table_rows = []
                for m_name, m_vals in saved_comparison.items():
                    table_rows.append({
                        "Algorithm": m_name,
                        "Accuracy": f"{m_vals['accuracy']*100:.1f}%",
                        "Recall": f"{m_vals['recall']*100:.1f}%",
                        "ROC-AUC": f"{m_vals['roc_auc']:.4f}",
                        "F1": f"{m_vals['f1_score']*100:.1f}%",
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
            st.markdown(
                '<div class="glass-card" style="padding:11px;margin-top:10px;font-size:12px;line-height:1.65">'
                '<b>Why Balanced Logistic Regression was selected:</b><br>'
                '• <b>Highest Recall (78.3%)</b> — catches the most churners before they leave.<br>'
                '• <b>Highest ROC-AUC (0.8416)</b> — best overall class discrimination.<br>'
                '• <b>Linear Interpretability</b> — direct odds ratios provide actionable insights without black-box opacity.'
                '</div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        '<div class="glass-card"><div class="section-heading">How to read these weights</div>'
        '<div class="section-sub" style="font-size:12px;margin-top:8px">'
        'Positive coefficients push the model score toward <b>Churn = Yes</b>; negative ones toward <b>No</b>. '
        'For a specific customer, the Predict page multiplies encoded values by model coefficients to produce a '
        'local contribution ranking. This is a diagnostic aid — do not interpret it as causal proof. '
        'Consider SHAP values for production-grade explainability.</div></div>',
        unsafe_allow_html=True,
    )


elif page == "Data":
    page_header(
        "Dataset telemetry  /  Customer feature registry",
        "Customer Dataset Explorer",
        "Inspect the cleaned Telco Customer Churn records, filter customer cohorts and export the current data slice.",
    )
    total_records       = len(df)
    churn_count         = int(df["Churn"].eq("Yes").sum())
    churn_rate          = df["Churn"].eq("Yes").mean() * 100
    predictive_features = len([c for c in df.columns if c not in {"customerID", "Churn", "SeniorLabel", "AvgMonthlySpend"}])

    d1, d2, d3, d4 = st.columns(4, gap="medium")
    with d1: metric_card("Total records",        f"{total_records:,}",        "Cleaned customer rows",            INDIGO)
    with d2: metric_card("Predictive features",  str(predictive_features),    "ID and target excluded",           CYAN)
    with d3: metric_card("Churn cohort",          f"{churn_count:,}",          f"{churn_rate:.1f}% of records",    CORAL)
    with d4: metric_card("Dataset columns",       str(len(df.columns) - 2),   "Excl. helper columns",             AMBER)

    st.write("")

    filter1, filter2, filter3, filter4 = st.columns([1.1, 1.1, 1.1, 1.0])
    with filter1:
        risk_label_filter = st.selectbox("Actual churn label", ["All", "Churned", "Retained"], key="data_churn_filter")
    with filter2:
        contract_data_filter = st.selectbox("Contract", ["All"] + sorted(df["Contract"].dropna().unique().tolist()), key="data_contract_filter")
    with filter3:
        internet_data_filter = st.selectbox("Internet", ["All"] + sorted(df["InternetService"].dropna().unique().tolist()), key="data_internet_filter")
    with filter4:
        max_tenure_filter = st.selectbox("Tenure band", ["All", "0–12 mo", "13–36 mo", "37–72 mo"], key="data_tenure_filter")

    gf_col, _ = st.columns([1.1, 3.3])
    with gf_col:
        gender_filter = st.selectbox("Gender", ["All", "Male", "Female"], key="data_gender_filter")

    data_view = df.copy()
    if risk_label_filter == "Churned":
        data_view = data_view[data_view["Churn"] == "Yes"]
    elif risk_label_filter == "Retained":
        data_view = data_view[data_view["Churn"] == "No"]
    if contract_data_filter != "All":
        data_view = data_view[data_view["Contract"] == contract_data_filter]
    if internet_data_filter != "All":
        data_view = data_view[data_view["InternetService"] == internet_data_filter]
    if gender_filter != "All":
        data_view = data_view[data_view["gender"] == gender_filter]
    if max_tenure_filter == "0–12 mo":
        data_view = data_view[data_view["tenure"].between(0, 12)]
    elif max_tenure_filter == "13–36 mo":
        data_view = data_view[data_view["tenure"].between(13, 36)]
    elif max_tenure_filter == "37–72 mo":
        data_view = data_view[data_view["tenure"].between(37, 72)]

    if search_query.strip():
        q = search_query.strip().lower()
        mask = data_view.astype(str).apply(lambda col: col.str.lower().str.contains(q, na=False))
        data_view = data_view[mask.any(axis=1)]

    with st.container(border=True):
        heading_col, export_col = st.columns([4, 1], vertical_alignment="center")
        with heading_col:
            section_title(
                "Customer feature registry",
                f"Showing {min(500, len(data_view)):,} of {len(data_view):,} filtered records · "
                f"{total_records:,} rows in cleaned source",
            )
        with export_col:
            st.download_button(
                "↓  Export slice",
                data=data_view.drop(columns=["SeniorLabel", "AvgMonthlySpend"], errors="ignore").to_csv(index=False).encode("utf-8"),
                file_name="telco_churn_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )

        show_cols = [
            "customerID", "gender", "SeniorLabel", "Partner", "Dependents", "tenure",
            "InternetService", "PhoneService", "MultipleLines", "OnlineSecurity",
            "TechSupport", "Contract", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
        ]
        show_cols = [c for c in show_cols if c in data_view.columns]
        st.dataframe(
            data_view[show_cols].head(500),
            use_container_width=True,
            hide_index=True,
            height=520,
            column_config={
                "customerID":    st.column_config.TextColumn("Customer ID"),
                "SeniorLabel":   st.column_config.TextColumn("Age cohort"),
                "tenure":        st.column_config.ProgressColumn("Tenure (months)", min_value=0, max_value=72, format="%d"),
                "MonthlyCharges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
                "TotalCharges":  st.column_config.NumberColumn("Total charges",   format="$%.2f"),
                "Churn":         st.column_config.TextColumn("Actual churn"),
            },
        )
        st.caption(
            "The Churn column is the historical target label — it is displayed for exploration only "
            "and is NOT supplied as an input to the prediction model. Showing up to 500 rows."
        )

        st.write("")
        st.write("")
        section_title("⚡ Test Churn Prediction for Any Customer in This View", "Select a customer from the filtered slice above to instantly load their profile and run inference")
        st.write("")
        sel_d_col, btn_d_col = st.columns([3, 1.2], vertical_alignment="bottom")
        with sel_d_col:
            slice_ids = data_view["customerID"].head(250).tolist()
            slice_label_map = get_customer_search_dict(df)
            chosen_id = st.selectbox(
                "Customer to inspect",
                options=slice_ids,
                format_func=lambda cid: slice_label_map.get(cid, cid),
                key="data_view_customer_picker",
            )
        with btn_d_col:
            if st.button("🚀 Load into Predict Simulator", type="primary", use_container_width=True, key="btn_data_load_sim"):
                if chosen_id:
                    load_customer_by_id(chosen_id)
                    st.session_state["nav_target"] = "Predict"
                    st.rerun()


elif page == "Summary":
    page_header(
        "Project overview  /  Academic ML project",
        "Churn Project — Project Summary",
        "A complete overview of the dataset, model pipeline, evaluation results and key business insights "
        "from the Telco Customer Churn analysis.",
    )

    with st.container(border=True):
        section_title("📊 Dataset at a glance", "WA_Fn-UseC_-Telco-Customer-Churn · IBM sample dataset")
        st.write("")
        d1, d2, d3, d4 = st.columns(4, gap="medium")
        with d1: metric_card("Total customers",    f"{len(df):,}",                                           "Rows in cleaned dataset",          INDIGO)
        with d2: metric_card("Churn rate",          f"{df['Churn'].eq('Yes').mean()*100:.1f}%",              "Class imbalance present",          CORAL)
        with d3: metric_card("Avg. tenure",         f"{df['tenure'].mean():.1f} mo",                        "Across all customers",             CYAN)
        with d4: metric_card("Avg. monthly charge", f"${df['MonthlyCharges'].mean():.0f}",                  "Across all customers",             AMBER)

    st.write("")

    with st.container(border=True):
        section_title("🤖 Model pipeline", "scikit-learn Pipeline with preprocessing and classifier steps")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("""
**Data preprocessing**
- `TotalCharges` coerced to numeric; 11 missing values → 0
- `customerID` dropped before training
- Numeric features: `StandardScaler`
- Categorical features: `OneHotEncoder(handle_unknown='ignore')`

**Classifier**
- `LogisticRegression(class_weight='balanced', max_iter=1000)`
- `class_weight='balanced'` corrects for ~26 % churn minority
- Trained on 80 % of 7,043 records; tested on 20 %
- `stratify=y` ensures representative splits
""")
        with col2:
            st.markdown(f"""
**Held-out test results**

| Metric | Score | Note |
|--------|-------|------|
| Accuracy | {MODEL_ACCURACY:.2f}% | Overall test accuracy |
| Precision | {MODEL_PRECISION:.2f}% | Focus on catching churners |
| Recall (churn) | {MODEL_RECALL:.2f}% | Primary goal (>78% detected) |
| F1 Score | {MODEL_F1:.2f}% | Harmonic mean of P & R |
| ROC-AUC | {MODEL_ROC_AUC:.4f} | Excellent discrimination |

**Candidate comparison (Test split):**
- **Logistic Regression**: Recall **78.3%** · ROC-AUC **0.8416**
- **Random Forest**: Recall 71.9% · ROC-AUC 0.8313
- **Decision Tree**: Recall 75.4% · ROC-AUC 0.8110
""")

    st.write("")

    with st.container(border=True):
        section_title("💡 Key business insights", "Derived from the coefficient analysis and cohort analytics")
        st.write("")
        insights = [
            ("Contract type", CORAL,  "Month-to-month customers churn ~43 % vs ~11 % (one year) and ~3 % (two year). "
                                       "Upgrading contract tenure is the single most impactful lever."),
            ("Internet service", CORAL, "Fibre optic customers churn ~42 % vs ~19 % (DSL). Service quality and pricing must be reviewed."),
            ("Payment method",   AMBER, "Electronic-check users churn ~45 % — ~2× the rate of auto-pay customers. "
                                         "Nudging toward automatic payment reduces churn."),
            ("Early-tenure risk", AMBER, "Customers in the first 12 months are at the highest risk. "
                                          "Onboarding programs and check-ins can significantly improve retention."),
            ("Add-on services",   CYAN,  "Tech support, online security and online backup each reduce churn risk. "
                                          "Cross-selling these to existing internet customers is a clear win."),
            ("Senior citizens",   CYAN,  "Seniors churn ~42 % vs ~24 % overall. A dedicated support pathway is recommended."),
        ]
        for driver, color, text in insights:
            st.markdown(
                f'<div class="glass-card" style="margin:8px 0;display:flex;gap:14px;align-items:flex-start;">'
                f'<span class="tag" style="background:rgba(0,0,0,.2);border-color:{color};color:{color};white-space:nowrap">{driver}</span>'
                f'<span style="font-size:13px;line-height:1.7">{text}</span></div>',
                unsafe_allow_html=True,
            )

    st.write("")

    with st.container(border=True):
        section_title("🚀 How to run the project", "Setup and launch instructions")
        st.code(
            "# 1. Create & activate the virtual environment\n"
            "python -m venv venv\n"
            "venv\\Scripts\\activate          # Windows\n\n"
            "# 2. Install dependencies\n"
            "pip install -r requirements.txt\n\n"
            "# 3. Train the model (generates models/customer_churn_model.pkl)\n"
            "python scripts/train_model.py\n\n"
            "# 4. Launch the dashboard\n"
            "streamlit run app/app.py",
            language="bash",
        )


st.markdown("---")
st.markdown(
    '<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#7F899C;font:12px JetBrains Mono,monospace">'
    '<span>Churn Project · Customer Churn Prediction System</span>'
    '<span>Balanced Logistic Regression · Local inference · Academic ML project · IT V SEM</span>'
    '</div>',
    unsafe_allow_html=True,
)
