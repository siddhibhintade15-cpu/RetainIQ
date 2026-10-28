"""
preprocess.py
-------------
All data-cleaning steps, packed into ONE scikit-learn pipeline so that the
exact same transformations are applied during training AND when the
dashboard predicts a new user.  (Rubric item 2: Data Preprocessing.)

Step                     Why we do it
-----------------------  ---------------------------------------------------
1. Drop duplicates       duplicate rows (logging glitches) would make the
                         model over-trust some users and leak into test set.
2. Drop user_id          an ID carries no behavioural information.
3. Impute numbers        missing age / session minutes -> MEDIAN (robust to
                         outliers, unlike the mean).
4. Cap outliers (IQR)    sessions of 900 minutes are app-left-open glitches;
                         values outside Q1-3*IQR .. Q3+3*IQR (EXTREME outliers;
                         3 not 1.5 because usage data is naturally skewed
                         and we only want to remove impossible values) are clipped
                         to the fence instead of deleting the user.
5. Scale numbers         StandardScaler (mean 0, std 1). XGBoost does NOT
                         need this, but Logistic Regression and KNN (used in
                         the comparison) do, so one shared pipeline is fair.
6. Impute categories     missing city_tier -> MOST FREQUENT value.
7. One-hot encode        turns text categories (Android / iOS ...) into 0/1
                         columns because ML models only understand numbers.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "retained_day30"
ID_COL = "user_id"

CATEGORICAL = ["device_os", "city_tier", "signup_channel", "subscription_plan"]
BINARY = ["on_latest_version", "onboarding_completed", "push_enabled"]
NUMERIC = ["age", "days_active_week1", "sessions_week1", "avg_session_minutes",
           "features_used", "push_opens_week1", "friends_added",
           "in_app_purchases", "crashes_week1", "support_tickets"]
# outlier capping only makes sense for truly continuous columns
CAP_COLUMNS = ["age", "sessions_week1", "avg_session_minutes"]

FEATURES = NUMERIC + BINARY + CATEGORICAL

FRIENDLY_NAMES = {
    "age": "Age",
    "days_active_week1": "Days active in week 1",
    "sessions_week1": "Sessions in week 1",
    "avg_session_minutes": "Avg session length (min)",
    "features_used": "Distinct features used",
    "push_opens_week1": "Push notifications opened",
    "friends_added": "Friends added",
    "in_app_purchases": "In-app purchases",
    "crashes_week1": "App crashes in week 1",
    "support_tickets": "Support tickets raised",
    "on_latest_version": "On latest app version",
    "onboarding_completed": "Onboarding completed",
    "push_enabled": "Push notifications enabled",
    "device_os": "Device OS",
    "city_tier": "City tier",
    "signup_channel": "Signup channel",
    "subscription_plan": "Subscription plan",
}


class IQRCapper(BaseEstimator, TransformerMixin):
    """Clip each selected column to [Q1 - k*IQR, Q3 + k*IQR] learned on train."""

    def __init__(self, columns_idx=None, k=1.5):
        self.columns_idx = columns_idx
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        cols = range(X.shape[1]) if self.columns_idx is None else self.columns_idx
        self.bounds_ = {}
        for c in cols:
            q1, q3 = np.percentile(X[:, c], [25, 75])
            iqr = q3 - q1
            self.bounds_[c] = (q1 - self.k * iqr, q3 + self.k * iqr)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        for c, (lo, hi) in self.bounds_.items():
            X[:, c] = np.clip(X[:, c], lo, hi)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning done before the pipeline (duplicates, ID column)."""
    df = df.drop_duplicates()
    return df.drop(columns=[ID_COL], errors="ignore")


def build_preprocessor() -> ColumnTransformer:
    num_cols = NUMERIC + BINARY
    cap_idx = [num_cols.index(c) for c in CAP_COLUMNS]
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("cap_outliers", IQRCapper(columns_idx=cap_idx, k=3.0)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, CATEGORICAL),
    ], verbose_feature_names_out=False)


def original_feature(transformed_name: str) -> str:
    """'device_os_iOS' -> 'device_os' (groups one-hot columns back)."""
    for c in CATEGORICAL:
        if transformed_name.startswith(c + "_"):
            return c
    return transformed_name
