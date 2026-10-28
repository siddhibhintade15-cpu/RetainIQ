"""
data.py
-------
Loads the dataset, the trained pipeline and the saved results ONCE
(Streamlit caching), and contains the prediction helpers used by the
Predict page.

If the model has not been trained yet, it trains it automatically the
first time the app starts (takes about a minute).
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import (BINARY, CATEGORICAL, FEATURES,  # noqa: E402
                            FRIENDLY_NAMES, NUMERIC, TARGET, original_feature)

DATA_PATH = ROOT / "data" / "app_users.csv"
MODEL_PATH = ROOT / "models" / "xgb_retention_pipeline.joblib"
RESULTS_PATH = ROOT / "models" / "results.json"
TEST_PATH = ROOT / "models" / "test_predictions.csv"


def ensure_trained():
    """Generate data and train the model if the files are missing."""
    if st.session_state.get("_model_ok"):
        return
    if not DATA_PATH.exists():
        with st.spinner("First run: generating the dataset ..."):
            subprocess.run([sys.executable, str(ROOT / "data" / "generate_data.py")],
                           check=True)
    need_training = not (MODEL_PATH.exists() and RESULTS_PATH.exists() and TEST_PATH.exists())
    if not need_training:
        # A model saved with different library versions may not load on this PC;
        # in that case simply retrain it here.
        try:
            joblib.load(MODEL_PATH)
        except Exception:
            need_training = True
    if need_training:
        with st.spinner("Training XGBoost and 7 comparison models on this computer "
                        "(about a minute, only once) ..."):
            subprocess.run([sys.executable, str(ROOT / "src" / "train.py")], check=True)
        load_pipeline.clear()
        load_results.clear()
        load_test_predictions.clear()
    st.session_state["_model_ok"] = True


@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_results() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def friendly(col: str) -> str:
    return FRIENDLY_NAMES.get(col, col.replace("_", " ").capitalize())


# ---------------- prediction helpers ----------------
def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Make an uploaded / form dataframe match the training columns."""
    df = df.copy()
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    for c in NUMERIC + BINARY:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
    return df[FEATURES]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    pipe = load_pipeline()
    return pipe.predict_proba(prepare_input(df))[:, 1]


def explain_one(row: pd.DataFrame) -> pd.DataFrame:
    """
    Per-feature contribution for ONE user using XGBoost's built-in
    TreeSHAP (pred_contribs=True).  Positive = pushes towards RETAINED,
    negative = pushes towards CHURN.  Units are log-odds.
    """
    pipe = load_pipeline()
    pre, model = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    X = pre.transform(prepare_input(row))
    names = list(pre.get_feature_names_out())
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
    bias = float(contribs[-1])
    s = pd.Series(contribs[:-1], index=names)
    grouped = s.groupby(lambda n: original_feature(n)).sum()
    out = pd.DataFrame({"feature": grouped.index, "contribution": grouped.values})
    out["label"] = out["feature"].map(friendly)
    out["value"] = out["feature"].map(lambda f: row.iloc[0][f])
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
    out.attrs["bias"] = bias
    return out


ACTIONS = {
    "days_active_week1": "Re-engagement nudge: a streak reward for opening the app on 4+ days in week 1.",
    "onboarding_completed": "Send a short 'finish setup' reminder - completing onboarding is a strong retention driver.",
    "features_used": "Show a guided tour of 2-3 features the user has not tried yet.",
    "friends_added": "Prompt the user to invite or follow friends (social lock-in).",
    "sessions_week1": "Schedule a personalised content notification to bring them back.",
    "push_opens_week1": "Improve push relevance - the user is not opening notifications.",
    "push_enabled": "Ask for notification permission at a high-value moment.",
    "crashes_week1": "Flag for QA: this user hit crashes - prioritise a fix / send an apology credit.",
    "support_tickets": "Follow up on the open support ticket within 24 hours.",
    "in_app_purchases": "Offer a first-purchase discount to build commitment.",
    "on_latest_version": "Prompt an app update - older versions retain worse.",
    "subscription_plan": "Offer a free premium trial week.",
    "signup_channel": "Paid-ads users churn faster: add them to a welcome email series.",
    "avg_session_minutes": "Surface quick-win content so sessions feel rewarding.",
    "age": "Tailor content recommendations to the user's age group.",
    "device_os": "Check device-specific performance for this OS.",
    "city_tier": "Use a lighter app mode / data-saver for slower networks.",
}


def recommendations(expl: pd.DataFrame, k=3):
    neg = expl[expl["contribution"] < 0].head(k)
    return [(r.label, ACTIONS.get(r.feature, "")) for r in neg.itertuples()]
