"""Model comparison - XGBoost vs other algorithms."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import theme as T
from ui.data import load_results

NOTES = {
    "Logistic Regression": ("Linear, fast, very interpretable",
                            "Assumes a straight-line (log-odds) relationship; cannot learn thresholds or interactions on its own"),
    "KNN (k=25)": ("No training, intuitive 'similar users' logic",
                   "Slow at prediction time; suffers when many features matter differently; needs scaling"),
    "Decision Tree": ("Human-readable rules, captures thresholds",
                      "A single tree is unstable and either underfits (shallow) or overfits (deep)"),
    "Random Forest": ("Averages many deep trees → robust, little tuning",
                      "Trees are built independently, so later trees don't fix earlier mistakes; large models"),
    "Gradient Boosting (sklearn)": ("Sequential error correction like XGBoost",
                                    "No second-order (hessian) step or built-in L2 on leaves; single-threaded and slower"),
    "XGBoost (default)": ("Strong out of the box, fast C++ engine",
                          "Default depth 6 with 100 trees memorises noise on this data (train AUC ≈ 0.99)"),
    "XGBoost (tuned)": ("Second-order boosting + λ/γ regularisation + row/column sampling, tuned by CV",
                        "Needs tuning; less transparent than a single tree (we use SHAP to explain it)"),
    "XGBoost (from scratch)": ("Our own NumPy implementation of the same maths",
                               "No parallelism or sparse-aware tricks → much slower than the library"),
}


def render():
    res = load_results()
    comp = res["comparison"]
    df = pd.DataFrame([{
        "Model": k, "ROC-AUC": v["roc_auc"], "F1": v["f1"], "Accuracy": v["accuracy"],
        "Precision": v["precision"], "Recall": v["recall"], "Train AUC": v["train_auc"],
        "Overfit gap": v["train_auc"] - v["roc_auc"], "Train time (s)": v["train_time_sec"]}
        for k, v in comp.items()]).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    worst = df.iloc[-1]

    T.page_header("Model Comparison",
                  "Eight models trained on the same pre-processed data and evaluated "
                  "on the same 2,000 held-out users.")
    st.write("")
    k = st.columns(4)
    name, _, variant = best["Model"].partition(" (")
    k[0].markdown(T.kpi_card("Best model", name,
                             f"{variant.rstrip(')')} · ROC-AUC {best['ROC-AUC']:.4f}".strip(" ·"),
                             dark=True), unsafe_allow_html=True)
    lr = df[df["Model"] == "Logistic Regression"].iloc[0]
    k[1].markdown(T.kpi_card("Gain over baseline", f"+{(best['ROC-AUC'] - lr['ROC-AUC']) * 100:.2f}",
                             "AUC points vs logistic regression"), unsafe_allow_html=True)
    xd = df[df["Model"] == "XGBoost (default)"].iloc[0]
    k[2].markdown(T.kpi_card("Tuning effect", f"+{(best['ROC-AUC'] - xd['ROC-AUC']) * 100:.2f}",
                             "AUC points vs default XGBoost"), unsafe_allow_html=True)
    k[3].markdown(T.kpi_card("Models compared", f"{len(df)}", f"weakest: {worst['Model']}"),
                  unsafe_allow_html=True)
    st.write("")

    with st.container(key="card_table"):
        T.card_title("Leaderboard", "Sorted by test ROC-AUC · threshold 0.5 for F1/accuracy/precision/recall")
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_config={
                "ROC-AUC": st.column_config.ProgressColumn("ROC-AUC", min_value=0.85, max_value=0.95, format="%.4f"),
                "F1": st.column_config.NumberColumn(format="%.4f"),
                "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                "Precision": st.column_config.NumberColumn(format="%.4f"),
                "Recall": st.column_config.NumberColumn(format="%.4f"),
                "Train AUC": st.column_config.NumberColumn(format="%.4f"),
                "Overfit gap": st.column_config.NumberColumn(format="%.4f"),
                "Train time (s)": st.column_config.NumberColumn(format="%.2f")})

    a, b = st.columns([1.2, 1])
    with a:
        with st.container(key="card_bars"):
            T.card_title("Metric comparison")
            metric = st.radio("Metric", ["ROC-AUC", "F1", "Accuracy", "Precision", "Recall"],
                              horizontal=True, label_visibility="collapsed")
            d = df.sort_values(metric)
            is_best = d[metric] == d[metric].max()
            xgb = d["Model"].str.contains("XGBoost")
            fig = go.Figure(go.Bar(
                x=d[metric], y=d["Model"], orientation="h",
                marker=dict(color=[T.GREEN_900 if bb else T.GREEN_500 if x else "white"
                                   for bb, x in zip(is_best, xgb)],
                            line=dict(color=T.GREEN_700, width=1.2),
                            pattern=dict(shape=["" if x else "/" for x in xgb], fgcolor=T.GREEN_700,
                                         solidity=0.3, size=7), cornerradius=8),
                text=[f"{v:.4f}" for v in d[metric]], textposition="outside",
                textfont=dict(size=10, color=T.MUTED)))
            lo = d[metric].min()
            fig.update_xaxes(range=[lo - (d[metric].max() - lo) * 0.6, d[metric].max() + (d[metric].max() - lo) * 0.35],
                             showgrid=True, gridcolor=T.LINE)
            T.show(fig, height=360, key="cmp_bars")
            st.caption("Axis is zoomed in so small differences are visible. Solid = XGBoost family, hatched = other algorithms.")
    with b:
        with st.container(key="card_rocall"):
            T.card_title("ROC curves", "All models on the same test users")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                     line=dict(color=T.MUTED, dash="dot", width=1)))
            palette = {"XGBoost (tuned)": T.GREEN_900, "XGBoost (from scratch)": T.GREEN_500,
                       "XGBoost (default)": T.GREEN_300, "Logistic Regression": T.AMBER,
                       "Random Forest": "#5C6BC0", "Gradient Boosting (sklearn)": "#00897B",
                       "Decision Tree": "#B0B7B1", "KNN (k=25)": "#E8603C"}
            for name in df["Model"]:
                v = comp[name]
                fig.add_trace(go.Scatter(x=v["roc_fpr"], y=v["roc_tpr"], mode="lines", name=name,
                                         line=dict(color=palette.get(name), width=3.5 if name == "XGBoost (tuned)" else 1.6)))
            fig.update_xaxes(title="False positive rate")
            fig.update_yaxes(title="True positive rate")
            fig.update_layout(legend=dict(orientation="v", x=0.42, y=0.04, yanchor="bottom",
                                          font=dict(size=10), bgcolor="rgba(255,255,255,.8)"))
            T.show(fig, height=400, key="cmp_roc")

    with st.container(key="card_overfit"):
        T.card_title("Overfitting check: train vs test ROC-AUC",
                     "A big gap means the model memorised the training users instead of learning general patterns")
        d = df.sort_values("Overfit gap")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d["Model"], y=d["Train AUC"], name="Train AUC",
                             marker=dict(color="white", line=dict(color=T.GREEN_700, width=1.2),
                                         pattern=dict(shape="/", fgcolor=T.GREEN_700, solidity=0.3),
                                         cornerradius=8)))
        fig.add_trace(go.Bar(x=d["Model"], y=d["ROC-AUC"], name="Test AUC",
                             marker=dict(color=T.GREEN_700, cornerradius=8)))
        fig.update_yaxes(range=[0.85, 1.0], title="ROC-AUC")
        fig.update_layout(barmode="group", bargap=0.3)
        T.show(fig, height=300, key="cmp_overfit")

    with st.container(key="card_why"):
        T.card_title("Why does tuned XGBoost come out on top?")
        rf = df[df["Model"] == "Random Forest"].iloc[0]
        st.markdown(
            f"""
- **The data has thresholds and interactions.** Retention jumps once a user is active on 4+ days,
  crashes hurt far more before onboarding is finished, and paid-ads users who stop after day 2
  almost never return. Trees learn such *if-then* rules directly; Logistic Regression
  (AUC {lr['ROC-AUC']:.4f}) can only add straight-line effects.
- **Boosting fixes mistakes sequentially.** Each new tree focuses on the users the previous trees got
  wrong (large gradients). Random Forest (AUC {rf['ROC-AUC']:.4f}) averages independent deep
  trees and overfits more (train AUC {rf['Train AUC']:.3f}).
- **Regularisation matters more than raw power.** Default XGBoost (depth 6) reached train AUC
  {xd['Train AUC']:.3f} but only {xd['ROC-AUC']:.4f} on test. Tuning chose shallow trees, a small
  learning rate and strong λ, closing the gap to {best['Overfit gap']:.3f}.
- **Honest caveat:** the margin over Logistic Regression and sklearn Gradient Boosting is small
  (a few thousandths of AUC), because most of the signal is strong and fairly simple. In a business
  setting XGBoost would still be preferred for its speed, built-in missing-value handling and
  SHAP explanations - but the simpler model is a reasonable fallback.
""")

    with st.container(key="card_notes"):
        T.card_title("Strengths & weaknesses")
        st.dataframe(pd.DataFrame([{"Model": k, "Strength": v[0], "Weakness": v[1]}
                                   for k, v in NOTES.items()]),
                     hide_index=True, width="stretch",
                     column_config={"Strength": st.column_config.TextColumn(width="large"),
                                    "Weakness": st.column_config.TextColumn(width="large")})
