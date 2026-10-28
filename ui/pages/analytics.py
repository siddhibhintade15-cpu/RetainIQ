"""Model analytics - metrics, threshold explorer, curves, importance, tuning, scratch model."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import theme as T
from ui.data import friendly, load_results, load_test_predictions

PARAM_HELP = {
    "n_estimators": "Number of trees added one after another. More trees = more capacity, "
                    "but too many with a high learning rate overfits.",
    "learning_rate": "How much each new tree is allowed to correct (eta). Small values need "
                     "more trees but generalise better.",
    "max_depth": "Maximum depth of each tree = how many features can interact in one rule.",
    "min_child_weight": "Minimum sum of hessians (≈ number of confident users) a leaf must "
                        "contain - stops rules built on a handful of users.",
    "gamma": "Minimum gain needed to make a split (γ in the gain formula). Prunes weak splits.",
    "reg_lambda": "L2 penalty on leaf weights (λ). Shrinks extreme leaf values.",
    "subsample": "Fraction of users randomly sampled for each tree (row bagging).",
    "colsample_bytree": "Fraction of features randomly sampled for each tree.",
}


def confusion_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def metric_tiles(m):
    cols = st.columns(5)
    names = [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
             ("F1-score", "f1"), ("ROC-AUC", "roc_auc")]
    for i, (lab, key) in enumerate(names):
        cols[i].markdown(T.kpi_card(lab, f"{m[key]:.3f}", "test set · threshold 0.5",
                                    dark=(key == "roc_auc")), unsafe_allow_html=True)


def render():
    res = load_results()
    test = load_test_predictions()
    y = test["retained_day30"].to_numpy()
    p = test["retain_probability"].to_numpy()
    fin = res["final"]

    T.page_header("Model Analytics",
                  "How well the tuned XGBoost model performs, how it was tuned, "
                  "and how the algorithm works under the hood.")
    st.write("")
    metric_tiles(fin["metrics"])
    st.write("")

    # ---------------- threshold explorer + manual calculation ----------------
    a, b = st.columns([1.05, 1])
    with a:
        with st.container(key="card_cm"):
            T.card_title("Confusion matrix & threshold",
                         "Move the threshold: a user is predicted 'retained' if probability ≥ threshold.")
            thr = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05)
            tp, tn, fp, fn = confusion_at(y, p, thr)
            z = [[tn, fp], [fn, tp]]
            txt = [[f"TN<br><b>{tn}</b>", f"FP<br><b>{fp}</b>"],
                   [f"FN<br><b>{fn}</b>", f"TP<br><b>{tp}</b>"]]
            fig = go.Figure(go.Heatmap(
                z=z, x=["Pred: churn", "Pred: retained"], y=["Actual: churn", "Actual: retained"],
                text=txt, texttemplate="%{text}", textfont=dict(size=15),
                colorscale=[[0, T.GREEN_100], [1, T.GREEN_700]], showscale=False,
                xgap=6, ygap=6, hovertemplate="%{y} / %{x}: %{z}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", showgrid=False)
            T.show(fig, height=270, key="cm")
    with b:
        with st.container(key="card_manual"):
            tot = tp + tn + fp + fn
            acc = (tp + tn) / tot
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            T.card_title("Manual metric calculation", f"Worked out by hand at threshold {thr:.2f}")
            st.markdown(
                f'<div class="formula">Accuracy  = (TP+TN)/Total = ({tp}+{tn})/{tot} = <b>{acc:.4f}</b></div>'
                f'<div class="formula">Precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = <b>{prec:.4f}</b></div>'
                f'<div class="formula">Recall    = TP/(TP+FN) = {tp}/({tp}+{fn}) = <b>{rec:.4f}</b></div>'
                f'<div class="formula">F1 = 2·P·R/(P+R) = 2·{prec:.3f}·{rec:.3f}/({prec:.3f}+{rec:.3f}) = <b>{f1:.4f}</b></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="card-sub" style="margin-top:.6rem"><b>Business reading:</b> '
                f'Recall = share of truly loyal users we recognise; for a churn campaign we care '
                f'about the <b>{tn + fn}</b> users flagged as churners, of whom <b>{tn}</b> '
                f'({tn / max(tn + fn, 1):.0%}) really churn. Lowering the threshold flags fewer '
                f'users as churners (cheaper campaign, but misses some).</div>',
                unsafe_allow_html=True)

    # ---------------- ROC + PR ----------------
    c, d = st.columns(2)
    with c:
        with st.container(key="card_roc"):
            m = res["comparison"]["XGBoost (tuned)"]
            T.card_title("ROC curve", f"AUC = {fin['metrics']['roc_auc']:.3f} · "
                                      "probability a random retained user scores above a random churner")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color=T.MUTED, dash="dot", width=1), name="Random"))
            fig.add_trace(go.Scatter(x=m["roc_fpr"], y=m["roc_tpr"], mode="lines", fill="tozeroy",
                                     fillcolor="rgba(47,154,99,.12)",
                                     line=dict(color=T.GREEN_700, width=3), name="XGBoost"))
            fig.update_xaxes(title="False positive rate", range=[0, 1])
            fig.update_yaxes(title="True positive rate", range=[0, 1.02])
            T.show(fig, height=290, key="roc")
    with d:
        with st.container(key="card_pr"):
            T.card_title("Precision–Recall curve",
                         f"Baseline precision = retention rate ({y.mean():.0%})")
            fig = go.Figure()
            fig.add_hline(y=y.mean(), line=dict(color=T.MUTED, dash="dot", width=1))
            fig.add_trace(go.Scatter(x=fin["pr_recall"], y=fin["pr_precision"], mode="lines",
                                     line=dict(color=T.GREEN_900, width=3), fill="tozeroy",
                                     fillcolor="rgba(15,61,38,.08)", name="XGBoost"))
            fig.update_xaxes(title="Recall", range=[0, 1])
            fig.update_yaxes(title="Precision", range=[0, 1.02])
            T.show(fig, height=290, key="pr")

    # ---------------- feature importance + probability distribution ----------------
    e, f = st.columns([1.1, 1])
    with e:
        with st.container(key="card_fi"):
            T.card_title("Feature importance (gain)",
                         "Share of total loss reduction from splits on each feature "
                         "(one-hot columns grouped)")
            g = pd.Series(res["feature_importance"]["grouped"]).sort_values()
            fig = go.Figure(go.Bar(
                x=g.values, y=[friendly(i) for i in g.index], orientation="h",
                marker=dict(color=[T.GREEN_900 if v == g.max() else T.GREEN_500 if v > .05
                                   else T.GREEN_300 for v in g.values], cornerradius=6),
                text=[f"{v:.1%}" for v in g.values], textposition="outside",
                textfont=dict(size=10, color=T.MUTED),
                hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig.update_xaxes(visible=False, range=[0, g.max() * 1.18])
            T.show(fig, height=430, key="fi")
    with f:
        with st.container(key="card_dist"):
            T.card_title("Predicted probability by actual outcome",
                         "Good separation = the two humps barely overlap")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=p[y == 0], name="Actually churned", nbinsx=30,
                                       marker=dict(color="white", line=dict(color=T.GREEN_700, width=1),
                                                   pattern=dict(shape="/", fgcolor=T.GREEN_700,
                                                                solidity=0.3)), opacity=0.95))
            fig.add_trace(go.Histogram(x=p[y == 1], name="Actually retained", nbinsx=30,
                                       marker=dict(color=T.GREEN_700), opacity=0.8))
            fig.update_layout(barmode="overlay", bargap=0.05)
            fig.update_xaxes(title="Predicted retention probability")
            fig.update_yaxes(title="Users")
            T.show(fig, height=430, key="dist")

    # ---------------- tuning ----------------
    tun = res["tuning"]
    st.markdown("### Hyper-parameter tuning")
    g1, g2 = st.columns([1, 1.25])
    with g1:
        with st.container(key="card_best"):
            T.card_title("Best parameters",
                         f"{tun['method']} · {tun['n_iter']} random candidates × "
                         f"{tun['cv_folds']}-fold stratified CV · scored by ROC-AUC")
            bp = tun["best_params"]
            tbl = pd.DataFrame({"Parameter": list(bp), "Chosen": [
                f"{v:.4g}" if isinstance(v, float) else str(v) for v in bp.values()],
                "Meaning": [PARAM_HELP.get(k, "") for k in bp]})
            st.dataframe(tbl, hide_index=True, width="stretch", height=330,
                         column_config={"Meaning": st.column_config.TextColumn(width="large")})
            st.markdown(
                f'<div class="card-sub">Best mean CV ROC-AUC: <b>{tun["best_cv_auc"]:.4f}</b> · '
                f'search took {tun["time_sec"]:.0f}s. Shallow trees (depth {bp["max_depth"]}) with '
                f'strong L2 (λ={bp["reg_lambda"]:.1f}) won: the signal is mostly simple rules '
                f'and low-order interactions, so extra depth only memorises noise.</div>',
                unsafe_allow_html=True)
    with g2:
        with st.container(key="card_trials"):
            T.card_title("Every candidate tried",
                         "Each dot = one parameter combination (5-fold CV). Colour = max_depth.")
            tr = pd.DataFrame(tun["trials"])
            xparam = st.radio("X-axis", ["learning_rate", "max_depth", "n_estimators",
                                         "reg_lambda", "min_child_weight"],
                              horizontal=True, label_visibility="collapsed")
            fig = go.Figure(go.Scatter(
                x=tr[xparam], y=tr["mean_cv_auc"], mode="markers",
                error_y=dict(array=tr["std_cv_auc"], color=T.LINE, thickness=1),
                marker=dict(size=13, color=tr["max_depth"],
                            colorscale=[[0, T.GREEN_300], [1, T.GREEN_950]],
                            line=dict(color="white", width=1.5), showscale=True,
                            colorbar=dict(title="depth", thickness=10, len=0.7)),
                customdata=tr[["max_depth", "learning_rate", "n_estimators"]],
                hovertemplate="CV AUC %{y:.4f}<br>depth %{customdata[0]} · lr %{customdata[1]:.3f}"
                              " · trees %{customdata[2]}<extra></extra>"))
            best = tr.loc[tr["mean_cv_auc"].idxmax()]
            fig.add_trace(go.Scatter(x=[best[xparam]], y=[best["mean_cv_auc"]], mode="markers",
                                     marker=dict(size=22, symbol="star", color=T.AMBER,
                                                 line=dict(color="white", width=1.5)),
                                     hoverinfo="skip", showlegend=False))
            fig.update_xaxes(title=xparam, type="log" if xparam in ("learning_rate", "reg_lambda") else "linear")
            fig.update_yaxes(title="Mean CV ROC-AUC")
            fig.update_layout(showlegend=False)
            T.show(fig, height=330, key="trials")

    with st.container(key="card_lc"):
        lc = res["learning_curve"]
        T.card_title("Learning curve: when do extra trees stop helping?",
                     "Log-loss after each tree for the tuned settings, trained to 800 trees. "
                     "The gap between train and test = overfitting.")
        xs = np.arange(len(lc["train_logloss"])) * lc["step"] + 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=lc["train_logloss"], name="Train log-loss",
                                 line=dict(color=T.GREEN_300, width=3)))
        fig.add_trace(go.Scatter(x=xs, y=lc["test_logloss"], name="Test log-loss",
                                 line=dict(color=T.GREEN_900, width=3)))
        fig.add_vline(x=lc["chosen_n_estimators"], line=dict(color=T.AMBER, dash="dash"),
                      annotation_text=f"chosen: {lc['chosen_n_estimators']} trees",
                      annotation_font_color=T.MUTED)
        fig.update_xaxes(title="Number of trees")
        fig.update_yaxes(title="Log-loss (lower is better)")
        T.show(fig, height=290, key="lc")

    # ---------------- scratch implementation ----------------
    st.markdown("### XGBoost implemented from scratch")
    sc = res["scratch"]
    fs = sc["first_split"]
    h1, h2 = st.columns([1.25, 1])
    with h1:
        with st.container(key="card_split"):
            T.card_title("How the first tree chose its first split",
                         "Real numbers from src/scratch_xgboost.py on the 8,000 training users")
            lam, gam = fs["reg_lambda"], fs["gamma"]
            sL = fs["GL"] ** 2 / (fs["HL"] + lam)
            sR = fs["GR"] ** 2 / (fs["HR"] + lam)
            sP = fs["G"] ** 2 / (fs["H"] + lam)
            p0 = 1 / (1 + np.exp(-fs["base_score_logodds"]))
            st.markdown(
                f'<div class="step"><div class="num">1</div><div class="body"><div class="t">Initial guess</div>'
                f'<div class="s">Everyone starts at the retention rate p₀ = {p0:.3f} → log-odds {fs["base_score_logodds"]:.3f}. '
                f'For each user g = p − y and h = p(1−p).</div></div></div>'
                f'<div class="step"><div class="num">2</div><div class="body"><div class="t">Best split found: '
                f'{T.esc(friendly(fs["feature"]))} &lt; {fs["threshold_original"]:.1f}</div>'
                f'<div class="s">Every feature × 32 histogram buckets was tried; this one had the highest gain.</div></div></div>'
                f'<div class="formula">Left (users with {T.esc(fs["feature"])} &lt; {fs["threshold_original"]:.0f}): '
                f'G<sub>L</sub> = {fs["GL"]:.1f}, H<sub>L</sub> = {fs["HL"]:.1f}<br>'
                f'Right: G<sub>R</sub> = {fs["GR"]:.1f}, H<sub>R</sub> = {fs["HR"]:.1f}  ·  λ = {lam}, γ = {gam}</div>'
                f'<div class="formula">Gain = ½[G<sub>L</sub>²/(H<sub>L</sub>+λ) + G<sub>R</sub>²/(H<sub>R</sub>+λ) − G²/(H+λ)] − γ<br>'
                f'     = ½[{sL:.1f} + {sR:.1f} − {sP:.1f}] − {gam} = <b>{fs["gain"]:.1f}</b></div>'
                f'<div class="step"><div class="num">3</div><div class="body"><div class="t">Leaf weights w = −G/(H+λ)</div>'
                f'<div class="s">Left leaf: {fs["leaf_weight_left"]:+.3f} (lower retention) · '
                f'Right leaf: {fs["leaf_weight_right"]:+.3f} (higher retention). '
                f'Each is multiplied by the learning rate (0.1) and added to the score.</div></div></div>',
                unsafe_allow_html=True)
    with h2:
        with st.container(key="card_scratch"):
            T.card_title("Scratch vs library",
                         f"{sc['params']['n_estimators']} trees · depth {sc['params']['max_depth']} · "
                         f"lr {sc['params']['learning_rate']} · λ {sc['params']['reg_lambda']}")
            s = res["comparison"]["XGBoost (from scratch)"]
            lib = res["comparison"]["XGBoost (tuned)"]
            st.markdown(
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.3rem;font-size:.86rem">'
                f'<div></div><b>Scratch</b><b>xgboost lib</b>'
                f'<span style="color:{T.MUTED}">ROC-AUC</span><span>{s["roc_auc"]:.4f}</span><span>{lib["roc_auc"]:.4f}</span>'
                f'<span style="color:{T.MUTED}">F1</span><span>{s["f1"]:.4f}</span><span>{lib["f1"]:.4f}</span>'
                f'<span style="color:{T.MUTED}">Accuracy</span><span>{s["accuracy"]:.4f}</span><span>{lib["accuracy"]:.4f}</span>'
                f'<span style="color:{T.MUTED}">Train time</span><span>{s["train_time_sec"]:.1f}s</span><span>{lib["train_time_sec"]:.2f}s</span>'
                f'</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Scatter(y=sc["loss_history"], mode="lines",
                                       line=dict(color=T.GREEN_700, width=3), fill="tozeroy",
                                       fillcolor="rgba(47,154,99,.1)"))
            fig.update_xaxes(title="Tree number")
            fig.update_yaxes(title="Train log-loss", range=[min(sc["loss_history"]) * 0.95,
                                                           max(sc["loss_history"]) * 1.02])
            T.show(fig, height=215, key="scratch_loss")
            st.caption("Near-identical accuracy proves the hand-written gradient/hessian, gain and "
                       "leaf-weight maths is the same algorithm the library uses; the library is "
                       "faster because it is written in C++ and multi-threaded.")
