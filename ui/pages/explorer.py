"""Data explorer - dataset understanding, preprocessing log and EDA charts."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.preprocess import BINARY, CATEGORICAL, NUMERIC, TARGET
from ui import theme as T
from ui.data import friendly, load_raw, load_results

DESCRIPTIONS = {
    "user_id": "Unique user identifier (dropped before modelling)",
    "age": "User's age in years",
    "device_os": "Phone operating system",
    "city_tier": "Size of the user's city (Tier 1 = metro)",
    "signup_channel": "How the user found the app",
    "subscription_plan": "Plan at the end of week 1",
    "on_latest_version": "1 if the user runs the latest app version",
    "onboarding_completed": "1 if the user finished the onboarding flow",
    "push_enabled": "1 if push notifications are allowed",
    "days_active_week1": "Number of days (0-7) the user opened the app in week 1",
    "sessions_week1": "Total app sessions in week 1",
    "avg_session_minutes": "Average session length in minutes",
    "features_used": "Number of distinct app features tried",
    "push_opens_week1": "Push notifications opened in week 1",
    "friends_added": "Friends added / followed in week 1",
    "in_app_purchases": "Purchases made in week 1",
    "crashes_week1": "App crashes experienced in week 1",
    "support_tickets": "Support tickets raised in week 1",
    "retained_day30": "TARGET - 1 if the user was still active on day 30",
}


def rate_bar(df, col, key, order=None, height=250):
    g = df.groupby(col)[TARGET].agg(["mean", "size"])
    if order:
        g = g.reindex([o for o in order if o in g.index])
    avg = df[TARGET].mean()
    fig = go.Figure(go.Bar(
        x=[str(i).replace("_", " ").title() if isinstance(i, str) else ("Yes" if i == 1 else "No")
           for i in g.index], y=g["mean"],
        marker=dict(color=[T.GREEN_700 if v >= avg else "white" for v in g["mean"]],
                    line=dict(color=T.GREEN_700, width=1.2),
                    pattern=dict(shape=["" if v >= avg else "/" for v in g["mean"]],
                                 fgcolor=T.GREEN_700, solidity=0.3, size=7),
                    cornerradius=12),
        text=[f"{v:.0%}" for v in g["mean"]], textposition="outside",
        textfont=dict(size=10, color=T.MUTED), customdata=g["size"],
        hovertemplate="%{x}<br>Retention %{y:.1%}<br>%{customdata} users<extra></extra>"))
    fig.add_hline(y=avg, line=dict(color=T.AMBER, dash="dot", width=1.5),
                  annotation_text=f"avg {avg:.0%}", annotation_font_color=T.MUTED,
                  annotation_position="top left")
    fig.update_yaxes(visible=False, range=[0, max(g["mean"].max() * 1.25, avg * 1.3)])
    T.show(fig, height=height, key=key)


def render():
    raw = load_raw()
    res = load_results()
    ds, pp = res["dataset"], res["preprocessing"]
    df = raw.drop_duplicates()

    T.page_header("Data Explorer",
                  "Understand the dataset, see exactly how it was cleaned, and explore "
                  "what separates retained users from churners.")
    st.write("")
    k = st.columns(4)
    k[0].markdown(T.kpi_card("Records", f"{ds['rows_raw']:,}", f"{ds['rows_clean']:,} after removing duplicates",
                             dark=True), unsafe_allow_html=True)
    k[1].markdown(T.kpi_card("Features", f"{ds['n_features']}",
                             f"{len(NUMERIC)} numeric · {len(BINARY)} binary · {len(CATEGORICAL)} categorical"),
                  unsafe_allow_html=True)
    miss = sum(v for v in ds["missing"].values())
    k[2].markdown(T.kpi_card("Missing values", f"{miss:,}",
                             f"in {sum(1 for v in ds['missing'].values() if v)} columns "
                             f"({miss / (ds['rows_raw'] * ds['columns']):.1%} of cells)"),
                  unsafe_allow_html=True)
    k[3].markdown(T.kpi_card("Target balance", f"{df[TARGET].mean():.0%}",
                             "retained · mildly imbalanced, stratified split used"),
                  unsafe_allow_html=True)
    st.write("")

    t1, t2, t3 = st.tabs(["Dataset & dictionary", "Preprocessing", "Exploratory analysis"])

    # ---------------- dictionary ----------------
    with t1:
        with st.container(key="card_dict"):
            T.card_title("Data dictionary",
                         "Each row is one user, described by what they did in the first 7 days "
                         "after install. Target = still active on day 30?")
            dd = pd.DataFrame({
                "Column": raw.columns,
                "Type": ["target" if c == TARGET else "id" if c == "user_id" else
                         "categorical" if c in CATEGORICAL else "binary" if c in BINARY else "numeric"
                         for c in raw.columns],
                "Pandas dtype": [str(raw[c].dtype) for c in raw.columns],
                "Missing": [int(raw[c].isna().sum()) for c in raw.columns],
                "Unique": [int(raw[c].nunique()) for c in raw.columns],
                "Description": [DESCRIPTIONS.get(c, "") for c in raw.columns]})
            st.dataframe(dd, hide_index=True, width="stretch", height=36 * (len(dd) + 1) + 4,
                         column_config={"Description": st.column_config.TextColumn(width="large")})
        with st.container(key="card_stats"):
            T.card_title("Summary statistics (numeric columns)")
            st.dataframe(raw[NUMERIC].describe().T.round(2), width="stretch")
        with st.container(key="card_sample"):
            T.card_title("Raw sample", "First 50 rows exactly as stored in data/app_users.csv")
            st.dataframe(raw.head(50), hide_index=True, width="stretch", height=300)

    # ---------------- preprocessing ----------------
    with t2:
        a, b = st.columns([1, 1.15])
        with a:
            with st.container(key="card_steps"):
                T.card_title("Pipeline steps (in order)",
                             "Fitted on the training set only, then re-used for test & live predictions")
                imp = ", ".join(f"{friendly(c)} → {v:g}" for c, v in pp["numeric_impute_median"].items())
                cimp = ", ".join(f"{friendly(c)} → {v}" for c, v in pp["categorical_impute_mode"].items())
                caps = "; ".join(f"{friendly(c['column'])} > {c['upper']:.1f} ({c['n_capped_train']} users)"
                                 for c in pp["outlier_caps"] if c["n_capped_train"])
                steps = [
                    ("Remove duplicate rows", f"{pp['duplicates_removed']} logging-glitch duplicates dropped."),
                    ("Drop identifier", "user_id carries no behaviour, so it is removed."),
                    ("Stratified train/test split", f"{pp['train_size']:,} train / {pp['test_size']:,} test, "
                                                    "same retention rate in both."),
                    ("Median imputation (numeric)", imp + ". Median is robust to the outliers below."),
                    ("Mode imputation (categorical)", cimp + "."),
                    ("IQR outlier capping (3×IQR)", caps + ". Clipped to the fence instead of deleting users."),
                    ("Standard scaling", "mean 0 / std 1 - needed by Logistic Regression & KNN; "
                                         "XGBoost is unaffected by scaling."),
                    ("One-hot encoding", f"4 categorical columns → {pp['encoded_feature_count']} model inputs in total."),
                ]
                st.markdown("".join(
                    f'<div class="step"><div class="num">{i}</div><div class="body"><div class="t">{T.esc(t)}</div>'
                    f'<div class="s">{T.esc(s)}</div></div></div>' for i, (t, s) in enumerate(steps, 1)),
                    unsafe_allow_html=True)
        with b:
            with st.container(key="card_missing"):
                T.card_title("Missing values per column (raw)")
                ms = pd.Series(ds["missing"])
                ms = ms[ms > 0].sort_values()
                fig = go.Figure(go.Bar(x=ms.values, y=[friendly(i) for i in ms.index], orientation="h",
                                       marker=dict(color=T.GREEN_500, cornerradius=6),
                                       text=[f"{v} ({v / ds['rows_raw']:.1%})" for v in ms.values],
                                       textposition="outside", textfont=dict(size=10, color=T.MUTED)))
                fig.update_xaxes(visible=False, range=[0, ms.max() * 1.35])
                T.show(fig, height=180, key="missing")
            with st.container(key="card_outlier"):
                T.card_title("Outlier capping: average session length",
                             "Before vs after clipping at the 3×IQR fence (log scale)")
                cap = next(c for c in pp["outlier_caps"] if c["column"] == "avg_session_minutes")
                before = df["avg_session_minutes"].dropna()
                after = before.clip(upper=cap["upper"])
                fig = go.Figure()
                fig.add_trace(go.Box(x=before, name="Before", marker_color=T.GREEN_300,
                                     boxpoints="outliers", line=dict(width=1.5)))
                fig.add_trace(go.Box(x=after, name="After", marker_color=T.GREEN_900,
                                     boxpoints="outliers", line=dict(width=1.5)))
                fig.update_xaxes(type="log", title="minutes (log scale)")
                fig.update_layout(showlegend=False)
                T.show(fig, height=230, key="outlier")
            with st.container(key="card_encoded"):
                T.card_title("Final model inputs", f"{pp['encoded_feature_count']} columns after encoding")
                st.markdown(" ".join(T.pill(f, "green") for f in pp["encoded_features"]),
                            unsafe_allow_html=True)

    # ---------------- EDA ----------------
    with t3:
        a, b, c = st.columns(3)
        with a:
            with st.container(key="card_eda_ch"):
                T.card_title("By signup channel", "Day-30 retention rate")
                rate_bar(df, "signup_channel", "eda_ch")
        with b:
            with st.container(key="card_eda_plan"):
                T.card_title("By subscription plan", "Day-30 retention rate")
                rate_bar(df, "subscription_plan", "eda_plan", order=["free", "trial", "premium"])
        with c:
            with st.container(key="card_eda_onb"):
                T.card_title("By onboarding completed", "Day-30 retention rate")
                rate_bar(df, "onboarding_completed", "eda_onb")

        d, e = st.columns([1.3, 1])
        with d:
            with st.container(key="card_eda_dist"):
                T.card_title("Feature distribution by outcome",
                             "Pick any numeric feature - where do retained and churned users differ?")
                feat = st.selectbox("Feature", NUMERIC, index=1, format_func=friendly,
                                    label_visibility="collapsed")
                fig = go.Figure()
                for val, name, style in [(0, "Churned", "hatch"), (1, "Retained", "solid")]:
                    x = df.loc[df[TARGET] == val, feat].dropna()
                    if feat == "avg_session_minutes":
                        x = x.clip(upper=40)
                    fig.add_trace(go.Histogram(
                        x=x, name=name, histnorm="percent", nbinsx=30, opacity=0.85,
                        marker=dict(color="white" if style == "hatch" else T.GREEN_700,
                                    line=dict(color=T.GREEN_700, width=1),
                                    pattern=dict(shape="/" if style == "hatch" else "",
                                                 fgcolor=T.GREEN_700, solidity=0.3))))
                fig.update_layout(barmode="overlay", bargap=0.06)
                fig.update_xaxes(title=friendly(feat))
                fig.update_yaxes(title="% of group")
                T.show(fig, height=300, key="eda_dist")
        with e:
            with st.container(key="card_eda_corr"):
                T.card_title("Correlation with retention",
                             "Pearson r between each numeric/binary feature and the target")
                corr = df[NUMERIC + BINARY + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
                fig = go.Figure(go.Bar(
                    x=corr.values, y=[friendly(i) for i in corr.index], orientation="h",
                    marker=dict(color=[T.GREEN_700 if v > 0 else T.RED for v in corr.values],
                                cornerradius=5),
                    hovertemplate="%{y}: r = %{x:.3f}<extra></extra>"))
                fig.add_vline(x=0, line_color=T.MUTED, line_width=1)
                T.show(fig, height=348, key="eda_corr")

        with st.container(key="card_eda_heat"):
            T.card_title("Interaction: active days × onboarding",
                         "Retention rate per cell - the same active days mean more when onboarding is done, "
                         "a non-linear pattern trees capture naturally")
            pv = df.pivot_table(index="onboarding_completed", columns="days_active_week1",
                                values=TARGET, aggfunc="mean")
            fig = go.Figure(go.Heatmap(
                z=pv.values, x=[f"{c} days" for c in pv.columns],
                y=["Not onboarded", "Onboarded"], colorscale=[[0, "#F7F8F6"], [1, T.GREEN_900]],
                text=np.vectorize(lambda v: "–" if np.isnan(v) else f"{v:.0%}")(pv.values), texttemplate="%{text}",
                xgap=4, ygap=4, showscale=False,
                hovertemplate="%{y}, %{x}: %{z:.1%}<extra></extra>"))
            T.show(fig, height=190, key="eda_heat")
        st.markdown(
            f'<div class="card-sub"><b>What the data says:</b> retention climbs steeply between 2 and 4 '
            f'active days and then flattens; referral users retain best and paid-ads users worst; '
            f'finishing onboarding is the single biggest yes/no factor; crashes hurt mostly when the '
            f'user is not yet onboarded.</div>', unsafe_allow_html=True)
