"""
train.py
--------
End-to-end training script. Run once; it saves everything the dashboard
needs into the  models/  folder.

    python src/train.py

What it does (mapped to the grading rubric):
  1. Dataset understanding ...... summary stats, missing values, class balance
  2. Preprocessing .............. duplicates, imputation, IQR capping, scaling,
                                  one-hot encoding  (see preprocess.py)
  3. Manual implementation ...... ScratchXGBClassifier (see scratch_xgboost.py)
  4. Modelling .................. XGBoost classifier
  5. Parameter tuning ........... RandomizedSearchCV, 5-fold stratified CV
  6. Evaluation ................. accuracy, precision, recall, F1, ROC-AUC,
                                  confusion matrix + one manual calculation
  7. Comparison ................. Logistic Regression, KNN, Decision Tree,
                                  Random Forest, Gradient Boosting, scratch,
                                  default XGBoost, tuned XGBoost
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import (CAP_COLUMNS, CATEGORICAL, FEATURES,  # noqa: E402
                            FRIENDLY_NAMES, ID_COL, NUMERIC, BINARY, TARGET,
                            build_preprocessor, clean_raw, original_feature)
from src.scratch_xgboost import ScratchXGBClassifier  # noqa: E402

DATA = ROOT / "data" / "app_users.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
SEED = 42


def metrics(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def thin_curve(x, y, n=120):
    """Keep ~n points of a curve so the JSON stays small."""
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return [round(float(v), 4) for v in np.asarray(x)[idx]], \
           [round(float(v), 4) for v in np.asarray(y)[idx]]


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 5)
    if isinstance(o, float):
        return None if np.isnan(o) else round(o, 5)
    return o


def main():
    t_start = time.time()
    results = {}

    # ------------------------------------------------------------------
    # 1. DATASET UNDERSTANDING
    # ------------------------------------------------------------------
    raw = pd.read_csv(DATA)
    print(f"Loaded {raw.shape[0]:,} rows x {raw.shape[1]} cols")
    n_dupes = int(raw.duplicated().sum())
    results["dataset"] = {
        "rows_raw": raw.shape[0],
        "columns": raw.shape[1],
        "n_features": len(FEATURES),
        "duplicates": n_dupes,
        "missing": raw.isna().sum().to_dict(),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "target_counts": raw[TARGET].value_counts().to_dict(),
        "describe": raw[NUMERIC].describe().round(2).to_dict(),
    }

    # ------------------------------------------------------------------
    # 2. PREPROCESSING
    # ------------------------------------------------------------------
    df = clean_raw(raw)
    results["dataset"]["rows_clean"] = df.shape[0]
    X, y = df[FEATURES], df[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train {len(X_train):,} | Test {len(X_test):,}")

    pre = build_preprocessor().fit(X_train)
    Xtr, Xte = pre.transform(X_train), pre.transform(X_test)
    feat_names = list(pre.get_feature_names_out())

    # document what preprocessing actually did (for the dashboard / report)
    num_pipe = pre.named_transformers_["num"]
    num_cols = NUMERIC + BINARY
    imp_vals = dict(zip(num_cols, num_pipe.named_steps["impute"].statistics_))
    cat_imp = dict(zip(CATEGORICAL,
                       pre.named_transformers_["cat"].named_steps["impute"].statistics_))
    caps = []
    capper = num_pipe.named_steps["cap_outliers"]
    for ci, (lo, hi) in capper.bounds_.items():
        col = num_cols[ci]
        filled = X_train[col].fillna(imp_vals[col])
        caps.append({"column": col, "lower": lo, "upper": hi,
                     "n_capped_train": int(((filled < lo) | (filled > hi)).sum()),
                     "max_before": float(X_train[col].max())})
    results["preprocessing"] = {
        "duplicates_removed": n_dupes,
        "numeric_impute_median": {c: imp_vals[c] for c in num_cols
                                  if raw[c].isna().any()},
        "categorical_impute_mode": {c: cat_imp[c] for c in CATEGORICAL
                                    if raw[c].isna().any()},
        "outlier_caps": caps,
        "encoded_feature_count": len(feat_names),
        "encoded_features": feat_names,
        "train_size": len(X_train), "test_size": len(X_test),
    }

    # ------------------------------------------------------------------
    # 3. MANUAL IMPLEMENTATION - XGBoost from scratch
    # ------------------------------------------------------------------
    print("Training scratch XGBoost ...")
    t0 = time.time()
    scratch = ScratchXGBClassifier(n_estimators=150, learning_rate=0.1,
                                   max_depth=3, reg_lambda=1.0, gamma=0.0)
    scratch.fit(Xtr, y_train)
    scratch_time = time.time() - t0
    fs = scratch.first_split_log
    fs["feature"] = feat_names[fs["feature_index"]]
    # threshold back in ORIGINAL units (undo the StandardScaler)
    thr_scaled = scratch.threshold_value(fs["feature_index"], fs["bin"])
    if fs["feature"] in num_cols:
        scaler = num_pipe.named_steps["scale"]
        j = num_cols.index(fs["feature"])
        fs["threshold_original"] = float(thr_scaled * scaler.scale_[j] + scaler.mean_[j])
    else:
        fs["threshold_original"] = thr_scaled
    fs["leaf_weight_left"] = -fs["GL"] / (fs["HL"] + fs["reg_lambda"])
    fs["leaf_weight_right"] = -fs["GR"] / (fs["HR"] + fs["reg_lambda"])
    fs["base_score_logodds"] = float(scratch.base_score)
    results["scratch"] = {"first_split": fs,
                          "loss_history": scratch.train_loss_history,
                          "params": {"n_estimators": 150, "learning_rate": 0.1,
                                     "max_depth": 3, "reg_lambda": 1.0,
                                     "gamma": 0.0, "n_bins": 32}}

    # ------------------------------------------------------------------
    # 5. PARAMETER TUNING (randomised search, 5-fold stratified CV)
    # ------------------------------------------------------------------
    print("Tuning XGBoost (RandomizedSearchCV, 30 candidates x 5 folds) ...")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    search_space = {
        "n_estimators": randint(150, 600),
        "max_depth": randint(2, 8),
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 2),
        "reg_lambda": loguniform(0.5, 10),
    }
    base_xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", random_state=SEED, n_jobs=2)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(base_xgb, search_space, n_iter=30, cv=cv,
                                scoring="roc_auc", random_state=SEED,
                                n_jobs=1, return_train_score=True, verbose=0)
    t0 = time.time()
    search.fit(Xtr, y_train)
    tune_time = time.time() - t0
    best_params = search.best_params_
    print(f"  best CV ROC-AUC = {search.best_score_:.4f}  ({tune_time:.0f}s)")

    cvr = pd.DataFrame(search.cv_results_)
    trials = []
    for _, r in cvr.iterrows():
        p = r["params"]
        trials.append({**{k: (float(v) if isinstance(v, (float, np.floating))
                              else int(v)) for k, v in p.items()},
                       "mean_cv_auc": float(r["mean_test_score"]),
                       "std_cv_auc": float(r["std_test_score"]),
                       "mean_train_auc": float(r["mean_train_score"]),
                       "rank": int(r["rank_test_score"])})
    results["tuning"] = {"method": "RandomizedSearchCV", "n_iter": 30,
                         "cv_folds": 5, "scoring": "roc_auc",
                         "best_params": best_params,
                         "best_cv_auc": search.best_score_,
                         "time_sec": tune_time, "trials": trials,
                         "scale_pos_weight_ratio": pos_weight}

    # learning curve of the tuned model: logloss vs number of trees
    lc_model = XGBClassifier(**{**best_params, "n_estimators": 800},
                             objective="binary:logistic",
                             eval_metric=["logloss", "auc"], tree_method="hist",
                             random_state=SEED, n_jobs=2)
    lc_model.fit(Xtr, y_train, eval_set=[(Xtr, y_train), (Xte, y_test)],
                 verbose=False)
    ev = lc_model.evals_result()
    results["learning_curve"] = {
        "train_logloss": ev["validation_0"]["logloss"][::5],
        "test_logloss": ev["validation_1"]["logloss"][::5],
        "test_auc": ev["validation_1"]["auc"][::5],
        "step": 5, "chosen_n_estimators": best_params["n_estimators"]}

    # ------------------------------------------------------------------
    # 4 + 7. MODELS AND COMPARISON
    # ------------------------------------------------------------------
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "KNN (k=25)": KNeighborsClassifier(n_neighbors=25),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                n_jobs=2, random_state=SEED),
        "Gradient Boosting (sklearn)": GradientBoostingClassifier(random_state=SEED),
        "XGBoost (default)": XGBClassifier(eval_metric="logloss", tree_method="hist",
                                           random_state=SEED, n_jobs=2),
        "XGBoost (tuned)": XGBClassifier(**best_params, eval_metric="logloss",
                                         tree_method="hist", random_state=SEED,
                                         n_jobs=2),
    }
    comparison, probas = {}, {}
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(Xtr, y_train)
        fit_t = time.time() - t0
        proba = model.predict_proba(Xte)[:, 1]
        probas[name] = proba
        m = metrics(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        m["roc_fpr"], m["roc_tpr"] = thin_curve(fpr, tpr)
        m["train_time_sec"] = fit_t
        m["train_auc"] = roc_auc_score(y_train, model.predict_proba(Xtr)[:, 1])
        comparison[name] = m
        print(f"  {name:<28} AUC={m['roc_auc']:.4f} F1={m['f1']:.4f} ({fit_t:.1f}s)")

    proba_s = scratch.predict_proba(Xte)[:, 1]
    m = metrics(y_test, proba_s)
    fpr, tpr, _ = roc_curve(y_test, proba_s)
    m["roc_fpr"], m["roc_tpr"] = thin_curve(fpr, tpr)
    m["train_time_sec"] = scratch_time
    m["train_auc"] = roc_auc_score(y_train, scratch.predict_proba(Xtr)[:, 1])
    comparison["XGBoost (from scratch)"] = m
    print(f"  {'XGBoost (from scratch)':<28} AUC={m['roc_auc']:.4f} F1={m['f1']:.4f}")
    results["comparison"] = comparison

    # ------------------------------------------------------------------
    # 6. EVALUATION of the final model
    # ------------------------------------------------------------------
    final_xgb = candidates["XGBoost (tuned)"]
    proba = probas["XGBoost (tuned)"]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    prec, rec, _ = precision_recall_curve(y_test, proba)
    pr_r, pr_p = thin_curve(rec[::-1], prec[::-1])
    results["final"] = {
        "name": "XGBoost (tuned)",
        "metrics": metrics(y_test, proba),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "pr_recall": pr_r, "pr_precision": pr_p,
        # manual calculation shown step by step (rubric item 6)
        "manual": {
            "accuracy": f"(TP + TN) / total = ({tp} + {tn}) / {tp+tn+fp+fn} = {(tp+tn)/(tp+tn+fp+fn):.4f}",
            "precision": f"TP / (TP + FP) = {tp} / ({tp} + {fp}) = {tp/(tp+fp):.4f}",
            "recall": f"TP / (TP + FN) = {tp} / ({tp} + {fn}) = {tp/(tp+fn):.4f}",
            "f1": f"2*P*R / (P + R) = 2*{tp/(tp+fp):.4f}*{tp/(tp+fn):.4f} / ({tp/(tp+fp):.4f} + {tp/(tp+fn):.4f}) = {2*tp/(2*tp+fp+fn):.4f}",
        },
    }

    # feature importance: gain, and grouped back to original columns
    booster = final_xgb.get_booster()
    gain = booster.get_score(importance_type="gain")
    gain_named = {feat_names[int(k[1:])]: v for k, v in gain.items()}
    total = sum(gain_named.values())
    grouped = {}
    for k, v in gain_named.items():
        o = original_feature(k)
        grouped[o] = grouped.get(o, 0) + v / total
    results["feature_importance"] = {
        "encoded": dict(sorted(((k, v / total) for k, v in gain_named.items()),
                               key=lambda t: -t[1])),
        "grouped": dict(sorted(grouped.items(), key=lambda t: -t[1])),
        "friendly": FRIENDLY_NAMES,
    }

    # test-set predictions for the dashboard (at-risk list, threshold slider)
    test_out = X_test.copy()
    test_out.insert(0, ID_COL, df.loc[X_test.index].index.map(
        lambda i: raw.loc[i, ID_COL]))
    test_out[TARGET] = y_test
    test_out["retain_probability"] = proba.round(4)
    test_out.to_csv(MODELS / "test_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    final_pipeline = Pipeline([("preprocess", pre), ("model", final_xgb)])
    joblib.dump(final_pipeline, MODELS / "xgb_retention_pipeline.joblib")
    results["meta"] = {"trained_at": time.strftime("%Y-%m-%d %H:%M"),
                       "total_time_sec": time.time() - t_start,
                       "features": FEATURES, "categorical": CATEGORICAL,
                       "numeric": NUMERIC, "binary": BINARY,
                       "cap_columns": CAP_COLUMNS, "target": TARGET}
    with open(MODELS / "results.json", "w") as f:
        json.dump(to_py(results), f, indent=1)
    print(f"Saved model + results to {MODELS}  ({time.time()-t_start:.0f}s total)")


if __name__ == "__main__":
    main()
