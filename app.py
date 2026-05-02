"""
Credit Risk Decisioning Lab
Interactive Cost-Sensitive ML Dashboard
Author: Sebastian Vu — UTS MSc Data Analytics
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
MODELS  = os.path.join(BASE, "models")
DATA    = os.path.join(BASE, "data", "train.csv")
RESULTS = os.path.join(BASE, "results")

# ── Constants ─────────────────────────────────────────────────────────────────
COST_FN    = 1_922.58
COST_FP    = 253.74
COST_RATIO = COST_FN / COST_FP    # 7.576×
T_STAR     = 1 / (1 + COST_RATIO) # ≈ 0.117

# Models available in repo (Random Forest excluded — too large for GitHub)
MODEL_KEYS = {
    "LightGBM ⭐":         "lightgbm",
    "XGBoost":             "xgboost",
    "AdaBoost":            "adaboost",
    "SVM":                 "svm",
    "Logistic Regression": "logistic_regression",
    "K-Nearest Neighbors": "knn",
    "Random Forest":       "random_forest",   # large — handled gracefully
}

# Accountants dropped (drop_first=True, alphabetically first)
OCCUPATION_TYPES = [
    "Accountants", "Cleaning staff", "Cooking staff", "Core staff",
    "Drivers", "HR staff", "High skill tech staff", "IT staff",
    "Laborers", "Low-skill Laborers", "Managers", "Medicine staff",
    "Private service staff", "Realty agents", "Sales staff",
    "Secretaries", "Security staff", "Unknown", "Waiters/barmen staff",
]

PAL = {
    "blue":   "#4299e1",
    "green":  "#68d391",
    "red":    "#fc8181",
    "yellow": "#f6e05e",
    "purple": "#b794f4",
    "gray":   "#4a5568",
    "bg":     "#0d1321",
    "border": "#1e2a3a",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0a0d14; color: #e2e8f0; }

.stTabs [data-baseweb="tab-list"] {
    background: #0d1321; border-bottom: 1px solid #1e2a3a; gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    letter-spacing: 1.5px; color: #4a5568; padding: 12px 22px;
    text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
    color: #4299e1 !important; border-bottom: 2px solid #4299e1 !important;
    background: transparent !important;
}
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #fff !important; font-size: 22px !important; }
h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #4299e1 !important; }
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #3182ce) !important;
    color: white !important; border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 12px 24px !important; border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(49,130,206,0.3) !important;
}
.card {
    background: #0d1321; border: 1px solid #1e2a3a;
    border-radius: 10px; padding: 18px;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    letter-spacing: 2.5px; color: #4a5568; text-transform: uppercase;
    border-bottom: 1px solid #1e2a3a; padding-bottom: 6px; margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1321", "axes.facecolor": "#0d1321",
    "axes.edgecolor":   "#1e2a3a", "axes.labelcolor": "#718096",
    "xtick.color":      "#4a5568", "ytick.color":     "#4a5568",
    "text.color":       "#e2e8f0", "grid.color":      "#1e2a3a",
    "grid.alpha":       0.5,       "font.family":     "monospace",
    "axes.spines.top":  False,     "axes.spines.right": False,
})


def sfig(figsize=(8, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1321")
    return fig, ax


# ── Dataframe styler helpers ──────────────────────────────────────────────────
def style_highlight_max(df, subset):
    """Highlight max per column with readable green text on dark background."""
    def _apply(s):
        is_max = s == s.max()
        return [
            "background-color: #0f3d1a; color: #68d391; font-weight: 700" if v else ""
            for v in is_max
        ]
    return df.style.apply(_apply, subset=subset)


def style_highlight_min(df, subset):
    """Highlight min per column with readable green text on dark background."""
    def _apply(s):
        is_min = s == s.min()
        return [
            "background-color: #0f3d1a; color: #68d391; font-weight: 700" if v else ""
            for v in is_min
        ]
    return df.style.apply(_apply, subset=subset)


# ── Data & model helpers ──────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    scaler    = joblib.load(os.path.join(MODELS, "scaler.pkl"))
    feat_cols = joblib.load(os.path.join(MODELS, "feature_columns.pkl"))
    return scaler, feat_cols


@st.cache_data
def load_raw():
    return pd.read_csv(DATA)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the notebook preprocessing pipeline exactly."""
    df = df.copy()
    df.drop(columns=["customer_id", "name"], errors="ignore", inplace=True)
    df = df[df["no_of_days_employed"] <= 36_500].reset_index(drop=True)

    gmode = df["gender"].mode()[0]
    df["gender"]     = df["gender"].replace("XNA", gmode).fillna(gmode).map({"M": 1, "F": 0})
    df["owns_car"]   = df["owns_car"].fillna("N").map({"Y": 1, "N": 0})
    df["owns_house"] = df["owns_house"].fillna("N").map({"Y": 1, "N": 0})

    for c in ["no_of_children", "no_of_days_employed", "yearly_debt_payments",
              "migrant_worker", "total_family_members", "credit_score"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    df["debt_to_income_ratio"]     = df["yearly_debt_payments"] / df["net_yearly_income"].replace(0, np.nan)
    df["children_family_ratio"]    = df["no_of_children"] / df["total_family_members"].replace(0, np.nan)
    df["income_per_family_member"] = df["net_yearly_income"] / df["total_family_members"].replace(0, np.nan)

    df.drop(columns=["credit_limit", "total_family_members", "owns_house"],
            errors="ignore", inplace=True)

    if "occupation_type" in df.columns:
        df = pd.get_dummies(df, columns=["occupation_type"], drop_first=True, dtype=int)

    return df


@st.cache_data
def get_test_set():
    """Reproduce the exact test split from the notebook (seed=42, stratified)."""
    raw = load_raw()
    df  = preprocess(raw)
    X   = df.drop(columns=["credit_card_default"])
    y   = df["credit_card_default"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler, feat_cols = load_assets()
    for c in feat_cols:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[feat_cols]
    # Return DataFrame so models with feature names don't warn
    Xs = pd.DataFrame(scaler.transform(X_test), columns=feat_cols)
    return Xs, y_test.values


def model_path(key: str, tuned: bool) -> str:
    prefix = "tuned" if tuned else "baseline"
    return os.path.join(MODELS, f"{prefix}_{key}.pkl")


def model_available(key: str, tuned: bool) -> bool:
    return os.path.exists(model_path(key, tuned))


@st.cache_resource
def load_model(key: str, tuned: bool = True):
    return joblib.load(model_path(key, tuned))


@st.cache_data
def get_predictions(model_name: str, tuned: bool = True):
    key = MODEL_KEYS[model_name]
    if not model_available(key, tuned):
        return None, None
    model = load_model(key, tuned)
    Xs, y = get_test_set()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xs)[:, 1]
    else:
        scores = model.decision_function(Xs.values)
        probs  = 1 / (1 + np.exp(-scores))
    return probs, y


def threshold_sweep(probs: np.ndarray, y: np.ndarray, n: int = 600):
    ts    = np.linspace(0.001, 0.999, n)
    costs = np.empty(n)
    fns   = np.empty(n, dtype=int)
    fps   = np.empty(n, dtype=int)
    for i, t in enumerate(ts):
        pred    = (probs >= t).astype(int)
        fns[i]  = int(((pred == 0) & (y == 1)).sum())
        fps[i]  = int(((pred == 1) & (y == 0)).sum())
        costs[i] = fns[i] * COST_FN + fps[i] * COST_FP
    return ts, costs, fns, fps


@st.cache_resource
def train_reference_models():
    """Train the 5 LightGBM variants from the notebook (mirrors retrain_no_smote.py).

    Returns dict: {variant_name: (probs_array, y_test_array)} or None if unavailable.
    Uses a fresh StandardScaler fit on X_train — identical to notebook pipeline.
    Cached per session (~30s first run, instant after).
    """
    try:
        from imblearn.over_sampling import SMOTE
        _has_smote = True
    except ImportError:
        _has_smote = False

    # ── Reproduce exact split ──────────────────────────────────────────────
    raw = load_raw()
    df  = preprocess(raw)
    X   = df.drop(columns=["credit_card_default"])
    y   = df["credit_card_default"].values

    # Align columns to the saved feature list (adds missing dummies as 0)
    _, feat_cols_saved = load_assets()
    for c in feat_cols_saved:
        if c not in X.columns:
            X[c] = 0
    X = X[feat_cols_saved]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fresh scaler fit on train — identical to notebook
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_train)
    X_te_s = sc.transform(X_test)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    BASE = dict(
        learning_rate=0.1, n_estimators=200, max_depth=-1,
        num_leaves=50, subsample=0.8, random_state=42, verbose=-1,
    )

    results = {}

    # 1. Raw — no reweighting, no SMOTE
    m = LGBMClassifier(**BASE)
    m.fit(X_tr_s, y_train)
    results["Raw"] = (m.predict_proba(X_te_s)[:, 1], y_test)

    # 2. Class-Weighted — scale_pos_weight = n_neg / n_pos
    m = LGBMClassifier(**BASE, scale_pos_weight=n_neg / n_pos)
    m.fit(X_tr_s, y_train)
    results["Class-Weighted"] = (m.predict_proba(X_te_s)[:, 1], y_test)

    # 3. SMOTE — oversample minority on train only
    if _has_smote:
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(X_tr_s, y_train)
        m = LGBMClassifier(**BASE)
        m.fit(X_sm, y_sm)
        results["SMOTE"] = (m.predict_proba(X_te_s)[:, 1], y_test)
    else:
        results["SMOTE"] = None   # imbalanced-learn not installed

    # 4. Calibrated — isotonic calibration (5-fold CV) on raw model
    base_m = LGBMClassifier(**BASE)
    cal_m  = CalibratedClassifierCV(base_m, cv=5, method="isotonic")
    cal_m.fit(X_tr_s, y_train)
    results["Calibrated"] = (cal_m.predict_proba(X_te_s)[:, 1], y_test)

    # 5. Custom Cost Obj — cost-weighted gradient objective
    # LightGBM 4.x API: objective in params, signature is (y_pred, train_set)
    def _cost_obj(y_pred, train_set):
        y_true = train_set.get_label()
        p      = 1.0 / (1.0 + np.exp(-y_pred))
        w      = np.where(y_true == 1, COST_RATIO, 1.0)
        grad   = w * (p - y_true)
        hess   = w * p * (1.0 - p)
        return grad, hess

    dtrain  = lgb.Dataset(X_tr_s, label=y_train)
    booster = lgb.train(
        {"objective": _cost_obj, "learning_rate": 0.1, "num_leaves": 50,
         "subsample": 0.8, "verbose": -1, "seed": 42},
        dtrain, num_boost_round=200,
    )
    raw_sc         = booster.predict(X_te_s, raw_score=True)
    probs_custom   = 1.0 / (1.0 + np.exp(-raw_sc))
    results["Custom Cost Obj"] = (probs_custom, y_test)

    return results


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧪 Credit Risk Decisioning Lab")
st.markdown(
    "<p style='color:#718096;font-family:IBM Plex Mono;font-size:11px;margin-top:-8px;'>"
    "7 Models &nbsp;·&nbsp; GridSearchCV &nbsp;·&nbsp; Cost-Sensitive Threshold Optimisation"
    " &nbsp;·&nbsp; 7.58× FN/FP Asymmetry &nbsp;·&nbsp; Sebastian Vu · UTS MSc Data Analytics</p>",
    unsafe_allow_html=True,
)

tab_eda, tab_arena, tab_cost, tab_predict = st.tabs([
    "📊  Data Explorer",
    "🏆  Model Arena",
    "💰  Cost Lab",
    "👤  Single Applicant",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    raw       = load_raw()
    total     = len(raw)
    n_default = int(raw["credit_card_default"].sum())
    dr        = n_default / total

    # ── KPI strip ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    for col, val, label, color in [
        (k1, f"{total:,}",     "Total Records",  PAL["blue"]),
        (k2, f"{n_default:,}", "Total Defaults",  PAL["red"]),
        (k3, f"{dr:.1%}",      "Default Rate",    PAL["yellow"]),
        (k4, "19",             "Raw Features",    PAL["purple"]),
    ]:
        col.markdown(f"""
        <div class="card" style="text-align:center;margin-bottom:8px">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:26px;font-weight:700;color:{color}">{val}</div>
          <div style="font-size:11px;color:#718096;margin-top:4px">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Row 1: Class balance + Missing values ──────────────────────────────
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("### Class Distribution")
        fig, ax = sfig(figsize=(5, 3))
        counts = raw["credit_card_default"].value_counts().sort_index()
        bars   = ax.bar(
            ["No Default (0)", "Default (1)"], counts.values,
            color=[PAL["green"], PAL["red"]], width=0.5,
            edgecolor="#0d1321", linewidth=0.5,
        )
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                    f"{v:,}  ({v/total:.1%})",
                    ha="center", va="bottom", fontsize=9, color="#a0aec0")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption("Heavy class imbalance (~8.7% defaults) — SMOTE applied during training.")

    with c2:
        st.markdown("### Missing Values per Feature")
        missing = raw.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)
        if len(missing):
            fig, ax = sfig(figsize=(5, 3))
            pct = missing.values / total * 100
            ax.barh(missing.index, pct, color=PAL["yellow"],
                    edgecolor="#0d1321", linewidth=0.5)
            ax.set_xlabel("Missing (%)")
            ax.grid(axis="x", alpha=0.2)
            for i, (v, p) in enumerate(zip(missing.values, pct)):
                ax.text(p + 0.05, i, f"{v:,}  ({p:.1f}%)",
                        va="center", fontsize=8, color="#a0aec0")
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption("All nulls imputed: median (numeric), mode (categorical).")

    st.divider()

    # ── Row 2: Correlation + Default by occupation ─────────────────────────
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown("### Feature Correlation with Default")
        num_cols = raw.select_dtypes(include=np.number).columns.tolist()
        num_cols = [c for c in num_cols if c not in ["customer_id", "credit_card_default"]]
        corrs    = (raw[num_cols + ["credit_card_default"]]
                    .corr()["credit_card_default"]
                    .drop("credit_card_default")
                    .sort_values())
        colors = [PAL["red"] if v > 0 else PAL["green"] for v in corrs.values]
        fig, ax = sfig(figsize=(5, 4))
        ax.barh(corrs.index, corrs.values, color=colors,
                edgecolor="#0d1321", linewidth=0.5)
        ax.axvline(0, color=PAL["gray"], linewidth=0.8)
        ax.set_xlabel("Pearson Correlation with Default")
        ax.grid(axis="x", alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        top3 = corrs.abs().nlargest(3).index.tolist()
        st.caption(f"Strongest predictors: **{'**, **'.join(top3)}**")

    with c4:
        st.markdown("### Default Rate by Occupation")
        if "occupation_type" in raw.columns:
            occ = (raw.groupby("occupation_type")["credit_card_default"]
                   .agg(["mean", "count"])
                   .rename(columns={"mean": "rate", "count": "n"})
                   .sort_values("rate", ascending=True))
            bar_colors = [
                PAL["red"] if v > 0.12 else PAL["yellow"] if v > 0.08 else PAL["green"]
                for v in occ["rate"].values
            ]
            fig, ax = sfig(figsize=(5, 4))
            ax.barh(occ.index, occ["rate"] * 100, color=bar_colors,
                    edgecolor="#0d1321", linewidth=0.5)
            ax.xaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_xlabel("Default Rate (%)")
            ax.grid(axis="x", alpha=0.2)
            plt.yticks(fontsize=7.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            top_occ = occ["rate"].idxmax()
            st.caption(f"Highest-risk: **{top_occ}** ({occ.loc[top_occ,'rate']:.1%} default rate)")

    st.divider()

    # ── Row 3: Feature distributions split by class ────────────────────────
    st.markdown("### Feature Distributions — Default vs No Default")
    feats_to_plot = [f for f in [
        "credit_score", "net_yearly_income", "credit_limit_used(%)",
        "prev_defaults", "default_in_last_6months", "no_of_days_employed",
    ] if f in raw.columns]

    cols3 = st.columns(3)
    for i, feat in enumerate(feats_to_plot):
        with cols3[i % 3]:
            fig, ax = sfig(figsize=(3.8, 2.6))
            d0 = raw.loc[raw["credit_card_default"] == 0, feat].dropna()
            d1 = raw.loc[raw["credit_card_default"] == 1, feat].dropna()
            ax.hist(d0, bins=30, alpha=0.6, color=PAL["green"], label="No Default", density=True)
            ax.hist(d1, bins=30, alpha=0.6, color=PAL["red"],   label="Default",    density=True)
            ax.set_title(feat, fontsize=8.5, color="#a0aec0")
            ax.legend(fontsize=7, framealpha=0)
            ax.grid(axis="y", alpha=0.2)
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL ARENA
# ══════════════════════════════════════════════════════════════════════════════
with tab_arena:
    st.markdown("### 7 Models — Baseline vs Tuned (GridSearchCV, 5-fold CV, ROC-AUC scoring)")

    try:
        base_df = pd.read_csv(os.path.join(RESULTS, "baseline_model_results.csv"))
        tune_df = pd.read_csv(os.path.join(RESULTS, "tuned_model_results.csv"))
        have_csvs = True
    except Exception:
        have_csvs = False

    if have_csvs:
        float_cols_b = base_df.select_dtypes(float).columns.tolist()
        float_cols_t = tune_df.select_dtypes(float).columns.tolist()

        cb, ct = st.columns(2, gap="large")
        with cb:
            st.markdown('<div class="section-label">Baseline — default hyperparameters</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                style_highlight_max(base_df, subset=float_cols_b)
                    .format({c: "{:.4f}" for c in float_cols_b}),
                use_container_width=True, hide_index=True,
            )
        with ct:
            st.markdown('<div class="section-label">Tuned — GridSearchCV best params</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                style_highlight_max(tune_df, subset=float_cols_t)
                    .format({c: "{:.4f}" for c in float_cols_t}),
                use_container_width=True, hide_index=True,
            )

        st.divider()

        # ── ROC-AUC grouped bar chart ──────────────────────────────────────
        st.markdown("### ROC-AUC: Baseline vs Tuned")
        auc_b = next((c for c in base_df.columns if "roc" in c.lower() or "auc" in c.lower()), None)
        auc_t = next((c for c in tune_df.columns if "roc" in c.lower() or "auc" in c.lower()), None)
        if auc_b and auc_t:
            n_m   = len(base_df)
            x     = np.arange(n_m)
            w     = 0.38
            fig, ax = sfig(figsize=(9, 3.5))
            ax.bar(x - w / 2, base_df[auc_b], w,
                   label="Baseline", color=PAL["blue"],  alpha=0.72, edgecolor="#0d1321")
            ax.bar(x + w / 2, tune_df[auc_t].iloc[:n_m], w,
                   label="Tuned",    color=PAL["green"], alpha=0.88, edgecolor="#0d1321")
            ax.set_xticks(x)
            ax.set_xticklabels(base_df["Model"], rotation=25, ha="right", fontsize=9)
            ax.set_ylabel("ROC-AUC")
            ax.set_ylim(0.90, 1.005)
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
            ax.legend(fontsize=9, framealpha=0)
            ax.grid(axis="y", alpha=0.2)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.divider()

        # ── Grouped bar: all metrics for tuned models ──────────────────────
        st.markdown("### Tuned Models — Full Metric Breakdown")
        metric_order = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        plot_cols    = [c for c in metric_order if c in tune_df.columns]
        if plot_cols:
            n_m   = len(tune_df)
            n_met = len(plot_cols)
            x     = np.arange(n_met)
            w     = 0.8 / n_m
            cmap  = plt.cm.get_cmap("tab10", n_m)
            fig, ax = sfig(figsize=(9, 3.5))
            for i, (_, row) in enumerate(tune_df.iterrows()):
                ax.bar(x + i * w - (n_m * w / 2),
                       [row[c] for c in plot_cols], w,
                       label=row["Model"], color=cmap(i),
                       alpha=0.85, edgecolor="#0d1321", linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(plot_cols, fontsize=9)
            ax.set_ylim(0.5, 1.02)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=7.5, framealpha=0, ncol=4,
                      bbox_to_anchor=(0.5, -0.28), loc="upper center")
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.divider()

    # ── Narrative bridge ───────────────────────────────────────────────────
    st.markdown("### But wait — is ROC-AUC the whole story?")
    n1, n2 = st.columns(2, gap="large")
    with n1:
        st.markdown(f"""
        <div style="padding:20px;background:#0d1321;border-left:3px solid {PAL['yellow']};
                    border-radius:0 8px 8px 0;font-size:13px;color:#a0aec0;line-height:1.8;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                      color:{PAL['yellow']};letter-spacing:1px;margin-bottom:10px;">
            THE PROBLEM WITH ACCURACY
          </div>
          Every model achieves <b style="color:#fff">ROC-AUC &gt; 0.94</b>.
          By standard ML benchmarks, they're all "excellent".<br><br>
          But in credit risk, <b style="color:{PAL['red']}">missing one defaulter
          costs ${COST_FN:,.2f}</b> while
          <b style="color:{PAL['yellow']}">rejecting a good customer costs only
          ${COST_FP:,.2f}</b>.<br><br>
          That's a <b style="color:#fff">{COST_RATIO:.1f}× cost asymmetry</b>
          that ROC-AUC is completely blind to.
        </div>""", unsafe_allow_html=True)
    with n2:
        st.markdown(f"""
        <div style="padding:20px;background:#0d1321;border-left:3px solid {PAL['green']};
                    border-radius:0 8px 8px 0;font-size:13px;color:#a0aec0;line-height:1.8;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                      color:{PAL['green']};letter-spacing:1px;margin-bottom:10px;">
            THE COST-OPTIMAL SOLUTION
          </div>
          Standard threshold (t = 0.50)
          &nbsp;→&nbsp; <b style="color:{PAL['red']}">~$290k total cost</b><br>
          Cost-optimal threshold (t ≈ 0.005)
          &nbsp;→&nbsp; <b style="color:{PAL['green']}">~$90k total cost</b><br><br>
          <b style="color:#fff">~70% cost reduction</b> — same model,
          smarter decision threshold.<br><br>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#4a5568;">
            → Go to 💰 Cost Lab to explore this interactively
          </span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COST LAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_cost:
    st.markdown("### 💰 Cost Lab — Interactive Threshold Optimisation")
    st.markdown(
        "<p style='color:#718096;font-size:13px;margin-top:-10px;'>"
        "Select a model, move the threshold slider, and watch financial cost change in real time.</p>",
        unsafe_allow_html=True,
    )

    # ── Threshold explanation box ──────────────────────────────────────────
    with st.expander("📖  Why t* differs across models — read this first", expanded=False):
        st.markdown(f"""
        There are **3 different threshold concepts** in this app — they are NOT the same thing:

        | | Value | What it is |
        |---|---|---|
        | **Theoretical t\\*** | `{1/(1+COST_RATIO):.3f}` | Bayes-optimal formula: `C_FP / (C_FP + C_FN)`. Assumes perfect model calibration. |
        | **Empirical t\\*** (raw model) | `~0.001` | Sweep result on the **raw imbalanced model** (no SMOTE, no tuning). Low because class imbalance biases probability outputs downward. |
        | **Empirical t\\*** (tuned model) | `~0.143` | Sweep result on the **GridSearchCV + SMOTE model**. SMOTE balances training data → probabilities are better calibrated → t\\* moves closer to the theoretical value. |

        **Why the tuned model shows lower total cost ($51k) vs raw model ($290k):**
        The tuned LightGBM (ROC-AUC = 0.9994) is a dramatically better classifier.
        Even at t = 0.50 it already makes very few errors. The raw model needs aggressive
        threshold lowering to catch defaulters that it naturally misses.

        **Reference Models tab** replicates the notebook's 5 training strategies (no GridSearchCV)
        so costs match the notebook's ~$290k baseline and ~$90k optimal figures exactly.
        """)

    # ── Analysis mode toggle ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Analysis Mode</div>', unsafe_allow_html=True)
    cost_mode = st.radio(
        "Analysis Mode",
        ["📦  Tuned / Baseline  (pkl models)", "🔬  Reference Models  — 5 LightGBM Variants"],
        horizontal=True, key="cost_mode", label_visibility="collapsed",
    )

    # ── Branch: resolve (probs, y_true) based on mode ─────────────────────
    probs       = None
    y_true      = None
    model_label = ""

    if cost_mode == "📦  Tuned / Baseline  (pkl models)":
        sc1, sc2 = st.columns([1.5, 1])
        with sc1:
            cost_model = st.selectbox("Model", list(MODEL_KEYS.keys()), key="cost_model")
        with sc2:
            cost_variant = st.radio(
                "Variant", ["Tuned (GridSearchCV)", "Baseline"],
                horizontal=True, key="cost_variant",
            )
        cost_tuned = cost_variant == "Tuned (GridSearchCV)"
        cost_key   = MODEL_KEYS[cost_model]

        if not model_available(cost_key, cost_tuned):
            st.warning(
                f"**{cost_model} ({'Tuned' if cost_tuned else 'Baseline'})** model file not found. "
                "Random Forest is excluded from this repo due to file size (52MB / 35MB). "
                "Select a different model or regenerate the pkl from the training notebook.",
                icon="⚠️",
            )
        else:
            with st.spinner("Loading model & computing predictions…"):
                probs, y_true = get_predictions(cost_model, cost_tuned)
            model_label = f"{cost_model} ({'Tuned' if cost_tuned else 'Baseline'})"

    else:  # ── Reference Models ────────────────────────────────────────────
        st.info(
            "**Reference models** are trained on-the-fly with the same pipeline as the notebook "
            "(no GridSearchCV). Costs match notebook values: ~$290k at t=0.50, ~$90k at t\\*. "
            "Training takes ~30s on first load — cached for the rest of your session.",
            icon="🔬",
        )
        REF_NAMES = ["Raw", "Class-Weighted", "SMOTE", "Calibrated", "Custom Cost Obj"]
        sel_variant = st.selectbox(
            "LightGBM Variant",
            REF_NAMES,
            format_func=lambda v: {
                "Raw":             "Raw — no reweighting, no SMOTE",
                "Class-Weighted":  "Class-Weighted — scale_pos_weight = n_neg/n_pos",
                "SMOTE":           "SMOTE — oversample minority class on train set",
                "Calibrated":      "Calibrated — isotonic calibration (5-fold CV)",
                "Custom Cost Obj": "Custom Cost Obj — cost-weighted gradient objective",
            }[v],
            key="cost_ref_variant",
        )
        with st.spinner("Training reference models — cached after first run (~30s)…"):
            ref_results = train_reference_models()

        if ref_results.get(sel_variant) is None:
            st.warning(
                "**SMOTE** requires the `imbalanced-learn` package. "
                "Run `pip install imbalanced-learn` and restart the app.",
                icon="⚠️",
            )
        else:
            probs, y_true = ref_results[sel_variant]
            model_label   = f"LightGBM — {sel_variant} (Reference)"

    # ══ Shared analysis block — only runs when (probs, y_true) are available ══
    if probs is not None and y_true is not None:
        ts, costs, fns, fps = threshold_sweep(probs, y_true)

        opt_idx   = int(np.argmin(costs))
        opt_t     = float(ts[opt_idx])
        opt_cost  = float(costs[opt_idx])
        half_idx  = int(np.argmin(np.abs(ts - 0.5)))
        half_cost = float(costs[half_idx])
        saving    = (half_cost - opt_cost) / half_cost * 100
        auc_val   = roc_auc_score(y_true, probs)

        # ── KPI strip ─────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        for col, val, label, sublabel, color in [
            (k1, f"${half_cost:,.0f}", "Cost at t = 0.50",   "standard threshold",              PAL["red"]),
            (k2, f"${opt_cost:,.0f}",  "Min Cost",            f"at empirical t* = {opt_t:.3f}", PAL["green"]),
            (k3, f"{saving:.1f}%",     "Cost Saved",          "vs standard t = 0.50",           PAL["yellow"]),
            (k4, f"t* = {opt_t:.3f}", "Empirical Optimal t*", f"theory: {1/(1+COST_RATIO):.3f}",PAL["blue"]),
            (k5, f"{auc_val:.4f}",    "ROC-AUC",              "on reconstructed test set",      PAL["purple"]),
        ]:
            col.markdown(f"""
            <div class="card" style="text-align:center;margin-bottom:6px">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;
                          font-weight:700;color:{color}">{val}</div>
              <div style="font-size:10px;color:#718096;margin-top:3px">{label}</div>
              <div style="font-size:9px;color:#4a5568;margin-top:1px;font-style:italic">{sublabel}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Threshold slider ───────────────────────────────────────────────
        sel_t = st.slider(
            "Decision threshold — drag to explore cost trade-off",
            0.001, 0.999, 0.500, 0.001, format="%.3f", key="cost_slider",
        )
        sel_idx  = int(np.argmin(np.abs(ts - sel_t)))
        sel_fn   = int(fns[sel_idx])
        sel_fp   = int(fps[sel_idx])
        sel_cost = float(costs[sel_idx])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("False Negatives",  f"{sel_fn:,}",
                  help="Defaulters incorrectly approved")
        m2.metric("False Positives",  f"{sel_fp:,}",
                  help="Good customers incorrectly rejected")
        m3.metric("FN Cost",          f"${sel_fn * COST_FN:,.0f}")
        m4.metric("FP Cost",          f"${sel_fp * COST_FP:,.0f}")
        m5.metric(
            "Total Cost", f"${sel_cost:,.0f}",
            delta=f"{((sel_cost - opt_cost) / opt_cost * 100):+.1f}% vs optimal",
            delta_color="inverse",
        )

        st.divider()

        # ── Row 1: Cost curve + Decomposition ─────────────────────────────
        ra, rb = st.columns([1.3, 1], gap="large")
        with ra:
            st.markdown("### Total Cost vs Threshold")
            fig, ax = sfig(figsize=(7, 3.8))
            ax.plot(ts, costs / 1_000, color=PAL["blue"], linewidth=2.2, zorder=3)
            ax.axvline(opt_t, color=PAL["green"], linestyle="--", linewidth=1.6, zorder=4,
                       label=f"Optimal  t*={opt_t:.3f}  (${opt_cost/1000:.0f}k)")
            ax.axvline(0.5,   color=PAL["gray"],  linestyle="--", linewidth=1.1, zorder=4,
                       label=f"Standard t=0.50  (${half_cost/1000:.0f}k)")
            ax.axvline(sel_t, color=PAL["yellow"], linestyle="-", linewidth=2.0, zorder=5,
                       label=f"Selected t={sel_t:.3f}  (${sel_cost/1000:.0f}k)")
            ax.fill_between(ts, costs / 1_000, costs.max() / 1_000, alpha=0.06, color=PAL["blue"])
            ax.set_xlabel("Decision Threshold")
            ax.set_ylabel("Total Cost ($k)")
            ax.legend(fontsize=8.5, framealpha=0)
            ax.grid(True, alpha=0.2)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with rb:
            st.markdown("### FN / FP Cost Split")
            fig, ax = sfig(figsize=(4.8, 3.8))
            ax.stackplot(
                ts,
                np.array(fns) * COST_FN / 1_000,
                np.array(fps) * COST_FP / 1_000,
                labels=[f"FN Cost — ${COST_FN:,.0f} each",
                        f"FP Cost — ${COST_FP:,.0f} each"],
                colors=[PAL["red"], PAL["yellow"]], alpha=0.82,
            )
            ax.axvline(sel_t, color="#fff", linewidth=1.8, linestyle="--",
                       label=f"t={sel_t:.3f}")
            ax.axvline(opt_t, color=PAL["green"], linewidth=1.2, linestyle=":",
                       label=f"Optimal t*")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Cost ($k)")
            ax.legend(fontsize=7.5, framealpha=0)
            ax.grid(True, alpha=0.15)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.divider()

        # ── Row 2: Confusion matrices ──────────────────────────────────────
        st.markdown("### Confusion Matrix Comparison")

        def draw_cm(probs, y, t, title, title_color):
            pred = (probs >= t).astype(int)
            cm   = confusion_matrix(y, pred)
            fn_  = cm[1, 0];  fp_ = cm[0, 1]
            c_   = fn_ * COST_FN + fp_ * COST_FP
            fig, ax = plt.subplots(figsize=(3.4, 3.0))
            fig.patch.set_facecolor("#0d1321")
            ax.set_facecolor("#0d1321")
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        linewidths=0.5, linecolor="#0d1321",
                        xticklabels=["Pred: No Default", "Pred: Default"],
                        yticklabels=["True: No Default", "True: Default"])
            ax.set_title(f"{title}\nt={t:.3f}  |  Cost: ${c_:,.0f}",
                         fontsize=8, color=title_color, pad=6)
            ax.tick_params(colors="#718096", labelsize=7.5)
            return fig

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.pyplot(draw_cm(probs, y_true, 0.5,   "Standard Threshold", PAL["red"]),
                      use_container_width=True)
            plt.close()
        with cc2:
            st.pyplot(draw_cm(probs, y_true, opt_t, "Optimal Threshold ⭐", PAL["green"]),
                      use_container_width=True)
            plt.close()
        with cc3:
            st.pyplot(draw_cm(probs, y_true, sel_t, "Selected Threshold", PAL["yellow"]),
                      use_container_width=True)
            plt.close()

        st.divider()

        # ── Row 3: ROC + PR curves ─────────────────────────────────────────
        rd, re = st.columns(2, gap="large")
        with rd:
            st.markdown("### ROC Curve")
            fpr_arr, tpr_arr, _ = roc_curve(y_true, probs)
            fig, ax = sfig(figsize=(5, 3.5))
            ax.plot(fpr_arr, tpr_arr, color=PAL["blue"], linewidth=2,
                    label=f"ROC-AUC = {auc_val:.4f}")
            ax.plot([0, 1], [0, 1], color=PAL["gray"], linestyle="--", linewidth=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(fontsize=9, framealpha=0)
            ax.grid(True, alpha=0.2)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with re:
            st.markdown("### Precision / Recall / F1 vs Threshold")
            prec_arr, rec_arr, pr_ts = precision_recall_curve(y_true, probs)
            prec_i = np.interp(ts, pr_ts[::-1], prec_arr[:-1][::-1])
            rec_i  = np.interp(ts, pr_ts[::-1], rec_arr[:-1][::-1])
            f1_i   = np.where(
                (prec_i + rec_i) > 0,
                2 * prec_i * rec_i / (prec_i + rec_i), 0,
            )
            fig, ax = sfig(figsize=(5, 3.5))
            ax.plot(ts, prec_i, color=PAL["blue"],   linewidth=1.8, label="Precision")
            ax.plot(ts, rec_i,  color=PAL["green"],  linewidth=1.8, label="Recall")
            ax.plot(ts, f1_i,   color=PAL["yellow"], linewidth=1.8, label="F1")
            ax.axvline(sel_t, color="#fff", linewidth=1.2, linestyle="--",
                       label=f"t={sel_t:.3f}")
            ax.axvline(opt_t, color=PAL["green"], linewidth=1.0, linestyle=":",
                       label=f"t*={opt_t:.3f}")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Score")
            ax.legend(fontsize=8, framealpha=0, ncol=3)
            ax.grid(True, alpha=0.2)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.divider()

        # ── All-variants comparison table (Reference mode) ─────────────────
        if cost_mode == "🔬  Reference Models  — 5 LightGBM Variants":
            st.markdown("### All 5 Reference Variants — Cost Summary")
            st.caption(
                "Costs computed at t = 0.50 (standard) and empirical t* (optimal). "
                "These match the notebook's reported figures."
            )
            rows = []
            for vname, res in ref_results.items():
                if res is None:
                    rows.append({"Variant": vname, "AUC": None,
                                 "Cost @ t=0.50": None, "Min Cost": None,
                                 "Optimal t*": None, "Cost Saving %": None})
                    continue
                p_, y_ = res
                ts_, cs_, _, _ = threshold_sweep(p_, y_)
                oi   = int(np.argmin(cs_))
                hi   = int(np.argmin(np.abs(ts_ - 0.5)))
                rows.append({
                    "Variant":       vname,
                    "AUC":           roc_auc_score(y_, p_),
                    "Cost @ t=0.50": cs_[hi],
                    "Min Cost":      cs_[oi],
                    "Optimal t*":    ts_[oi],
                    "Cost Saving %": (cs_[hi] - cs_[oi]) / cs_[hi] * 100,
                })
            comp_df = pd.DataFrame(rows)
            cost_c  = ["Cost @ t=0.50", "Min Cost"]
            pct_c   = ["Cost Saving %"]
            fmt     = {**{c: "${:,.0f}" for c in cost_c},
                       **{c: "{:.1f}%"  for c in pct_c},
                       "AUC": "{:.4f}", "Optimal t*": "{:.3f}"}
            st.dataframe(
                style_highlight_min(comp_df, subset=cost_c)
                    .format(fmt, na_rep="—"),
                use_container_width=True, hide_index=True,
            )

        # ── LightGBM variant CSV table (when available) ────────────────────
        try:
            ns_df = pd.read_csv(os.path.join(RESULTS, "no_smote_threshold_results.csv"))
            st.markdown("### LightGBM Variants — Notebook CSV Results")
            st.caption(
                "Raw imbalanced · Class-weighted · SMOTE · Calibrated · Custom cost-sensitive objective"
            )
            cost_cols   = [c for c in ns_df.columns if "cost" in c.lower()]
            other_float = [c for c in ns_df.select_dtypes(float).columns if c not in cost_cols]
            fmt2 = {**{c: "${:,.0f}" for c in cost_cols},
                    **{c: "{:.4f}"   for c in other_float}}
            st.dataframe(
                style_highlight_min(ns_df, subset=cost_cols)
                    .format(fmt2),
                use_container_width=True, hide_index=True,
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SINGLE APPLICANT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    scaler, feat_cols = load_assets()
    occ_dummy_cols    = [c for c in feat_cols if c.startswith("occupation_type_")]

    col_form, col_out = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-label">Model Selection</div>', unsafe_allow_html=True)
        p_model = st.selectbox("Model", list(MODEL_KEYS.keys()), key="p_model")
        p_tuned = st.radio("Variant", ["Tuned", "Baseline"],
                           horizontal=True, key="p_tuned") == "Tuned"
        p_thresh = st.slider(
            f"Decision threshold  (theoretical t* ≈ {T_STAR:.3f})",
            0.001, 0.999, 0.010, 0.001, format="%.3f", key="p_thresh",
        )

        st.markdown('<div class="section-label" style="margin-top:14px">Applicant Profile</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age        = st.number_input("Age", 18, 80, 35, key="p_age")
            gender     = st.selectbox("Gender", ["Male", "Female"], key="p_gender")
            owns_car   = st.selectbox("Owns Car", ["No", "Yes"], key="p_car")
            occupation = st.selectbox("Occupation", OCCUPATION_TYPES, index=8, key="p_occ")
        with c2:
            no_children  = st.number_input("No. of Children", 0, 10, 0, key="p_ch")
            total_family = st.number_input("Family Members (total)", 1, 15, 2, key="p_fam")
            migrant      = st.selectbox("Migrant Worker", ["No", "Yes"], key="p_mig")
            days_empl    = st.number_input("Days Employed", 0, 20_000, 2_000, key="p_days")

        st.markdown('<div class="section-label" style="margin-top:14px">Financial Profile</div>',
                    unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            net_income   = st.number_input("Net Yearly Income ($)", 0, 1_000_000,
                                           80_000, step=1_000, key="p_inc")
            yearly_debt  = st.number_input("Yearly Debt Payments ($)", 0, 500_000,
                                           15_000, step=500, key="p_debt")
            credit_score = st.slider("Credit Score", 300, 900, 700, key="p_cs")
        with c4:
            credit_used  = st.slider("Credit Limit Used (%)", 0, 100, 40, key="p_cu")
            prev_def     = st.number_input("Previous Defaults", 0, 10, 0, key="p_pd")
            def_last6    = st.selectbox("Default in Last 6 Months",
                                        ["No", "Yes"], key="p_d6")

        run = st.button("▶  Run Assessment", use_container_width=True)

    with col_out:
        st.markdown('<div class="section-label">Decision Output</div>', unsafe_allow_html=True)

        if run:
            p_key = MODEL_KEYS[p_model]
            if not model_available(p_key, p_tuned):
                st.warning(
                    f"**{p_model}** model file not found. "
                    "Random Forest is excluded due to file size. Please select another model.",
                    icon="⚠️",
                )
            else:
                try:
                    # ── Build feature row ──────────────────────────────────
                    row = {
                        "age":                     float(age),
                        "gender":                  1 if gender == "Male" else 0,
                        "owns_car":                1 if owns_car == "Yes" else 0,
                        "no_of_children":          float(no_children),
                        "net_yearly_income":       float(net_income),
                        "no_of_days_employed":     float(days_empl),
                        "migrant_worker":          1.0 if migrant == "Yes" else 0.0,
                        "yearly_debt_payments":    float(yearly_debt),
                        "credit_limit_used(%)":    float(credit_used),
                        "credit_score":            float(credit_score),
                        "prev_defaults":           float(prev_def),
                        "default_in_last_6months": 1 if def_last6 == "Yes" else 0,
                        "debt_to_income_ratio":    float(yearly_debt) / max(float(net_income), 1),
                        "children_family_ratio":   float(no_children) / max(float(total_family), 1),
                        "income_per_family_member": float(net_income) / max(float(total_family), 1),
                    }
                    occ_key = f"occupation_type_{occupation}"
                    for c in occ_dummy_cols:
                        row[c] = 1 if c == occ_key else 0

                    X_row = pd.DataFrame([row])
                    for c in feat_cols:
                        if c not in X_row.columns:
                            X_row[c] = 0
                    X_row = X_row[feat_cols]

                    Xs    = pd.DataFrame(scaler.transform(X_row), columns=feat_cols)
                    model = load_model(p_key, p_tuned)

                    if hasattr(model, "predict_proba"):
                        prob = float(model.predict_proba(Xs)[0, 1])
                    else:
                        score = float(model.decision_function(Xs.values)[0])
                        prob  = float(1 / (1 + np.exp(-score)))

                    decision = "REJECT" if prob >= p_thresh else "APPROVE"
                    pct      = prob * 100
                    color    = PAL["red"] if prob >= p_thresh else PAL["green"]

                    # ── Decision banner ────────────────────────────────────
                    if decision == "APPROVE":
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#0a2116,#0f3320);
                                    border:2px solid #38a169;border-radius:14px;
                                    padding:28px;text-align:center;">
                          <div style="font-family:'IBM Plex Mono',monospace;font-size:32px;
                                      font-weight:700;color:#68d391;">✓ APPROVE</div>
                          <div style="color:#a0aec0;margin-top:6px;font-size:13px;">
                            Default probability below threshold
                          </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#1a0a0a,#2d1515);
                                    border:2px solid #e53e3e;border-radius:14px;
                                    padding:28px;text-align:center;">
                          <div style="font-family:'IBM Plex Mono',monospace;font-size:32px;
                                      font-weight:700;color:#fc8181;">✗ REJECT</div>
                          <div style="color:#a0aec0;margin-top:6px;font-size:13px;">
                            Default probability exceeds threshold
                          </div>
                        </div>""", unsafe_allow_html=True)

                    # ── Probability gauge ──────────────────────────────────
                    st.markdown(f"""
                    <div class="card" style="margin-top:12px">
                      <div style="color:#4a5568;font-size:10px;letter-spacing:1px;margin-bottom:6px;">
                        DEFAULT PROBABILITY &nbsp;·&nbsp; {p_model}
                        {'(Tuned)' if p_tuned else '(Baseline)'}
                      </div>
                      <div style="font-size:40px;font-weight:700;color:{color};line-height:1;">
                        {pct:.1f}%
                      </div>
                      <div style="background:#1e2a3a;border-radius:3px;height:6px;margin-top:10px;">
                        <div style="background:{color};width:{min(pct,100):.1f}%;
                                    height:6px;border-radius:3px;"></div>
                      </div>
                      <div style="display:flex;justify-content:space-between;
                                  font-size:10px;color:#4a5568;margin-top:4px;">
                        <span>0%</span>
                        <span>Threshold: {p_thresh:.1%}</span>
                        <span>100%</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    # ── Cost exposure ──────────────────────────────────────
                    st.markdown(
                        '<div class="section-label" style="margin-top:14px">Cost Exposure</div>',
                        unsafe_allow_html=True,
                    )
                    ca2, cb2 = st.columns(2)
                    ca2.markdown(f"""
                    <div class="card">
                      <div style="color:#4a5568;font-size:10px;letter-spacing:1px;">IF FALSE NEGATIVE</div>
                      <div style="font-size:22px;font-weight:700;color:{PAL['red']};">${COST_FN:,.0f}</div>
                      <div style="color:#718096;font-size:10px;">Approve a defaulter</div>
                    </div>""", unsafe_allow_html=True)
                    cb2.markdown(f"""
                    <div class="card">
                      <div style="color:#4a5568;font-size:10px;letter-spacing:1px;">IF FALSE POSITIVE</div>
                      <div style="font-size:22px;font-weight:700;color:{PAL['yellow']};">${COST_FP:,.0f}</div>
                      <div style="color:#718096;font-size:10px;">Reject a good customer</div>
                    </div>""", unsafe_allow_html=True)

                    # ── Risk signals ───────────────────────────────────────
                    st.markdown(
                        '<div class="section-label" style="margin-top:14px">Risk Signals</div>',
                        unsafe_allow_html=True,
                    )
                    dti     = float(yearly_debt) / max(float(net_income), 1)
                    signals = []
                    if prev_def > 0:
                        signals.append(("⚠ Previous defaults", str(int(prev_def)), PAL["red"]))
                    if def_last6 == "Yes":
                        signals.append(("⚠ Default in last 6 months", "Yes", PAL["red"]))
                    if credit_used > 70:
                        signals.append(("↑ High credit utilization", f"{credit_used}%", PAL["yellow"]))
                    if dti > 0.4:
                        signals.append(("↑ High debt-to-income ratio", f"{dti:.1%}", PAL["yellow"]))
                    if credit_score < 600:
                        signals.append(("↓ Low credit score", str(int(credit_score)), PAL["yellow"]))
                    if not signals:
                        signals.append(("✓ No major risk signals", "", PAL["green"]))

                    for lbl, val_s, c in signals:
                        st.markdown(f"""
                        <div style="padding:9px 12px;border-radius:0 6px 6px 0;border-left:3px solid {c};
                                    margin:4px 0;background:#0d1321;font-size:13px;
                                    display:flex;justify-content:space-between;">
                          <span>{lbl}</span>
                          <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                                       color:{c};font-weight:600;">{val_s}</span>
                        </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.markdown("""
            <div style="text-align:center;padding:80px 0;color:#4a5568;">
              <div style="font-size:52px;">🧪</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                          margin-top:16px;line-height:2.4;letter-spacing:0.5px;">
                Configure the applicant profile<br>and click ▶ Run Assessment
              </div>
            </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#4a5568;font-family:'IBM Plex Mono',monospace;
            font-size:10px;line-height:2.4;">
  Credit Risk Decisioning Lab &nbsp;·&nbsp; Sebastian Vu &nbsp;·&nbsp; UTS MSc Data Analytics<br>
  7 Models &nbsp;·&nbsp; GridSearchCV &nbsp;·&nbsp; Cost-sensitive threshold &nbsp;·&nbsp;
  7.58:1 FN/FP asymmetry &nbsp;·&nbsp; ~70% cost reduction
</div>
""", unsafe_allow_html=True)
