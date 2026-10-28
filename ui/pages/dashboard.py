"""Dashboard (overview) page - layout mirrors the reference design."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui import theme as T
from ui.data import friendly, load_raw, load_results, load_test_predictions
from ui.nav import go_to


def render():
    res = load_results()
    raw = load_raw()
    test = load_test_predictions()
    df = raw.drop_duplicates()
    fin = res["final"]["metrics"]

    # ---------------- header ----------------
    h1, h2 = st.columns([3, 1.4], vertical_alignment="bottom")
    with h1:
        T.page_header("Dashboard",
                      "Predict, understand and reduce mobile-app churn with XGBoost.")
    with h2:
        b1, b2 = st.columns(2)
        if b1.button("Predict User", icon=":material/add:", type="primary",
                     width="stretch"):
            go_to("predict")
        if b2.button("Explore Data", width="stretch"):
            go_to("explorer")

    # ---------------- KPI cards ----------------
    n = len(df)
    ret = int(df["retained_day30"].sum())
    churn = n - ret
    k = st.columns(4)
    k[0].markdown(T.kpi_card("Total Users", f"{n:,}", "users tracked for 30 days",
                             badge="7d", dark=True), unsafe_allow_html=True)
    k[1].markdown(T.kpi_card("Retained (Day 30)", f"{ret:,}",
                             f"{ret / n:.1%} retention rate", badge="▲"),
                  unsafe_allow_html=True)
    k[2].markdown(T.kpi_card("Churned Users", f"{churn:,}",
                             f"{churn / n:.1%} left within 30 days", badge="▼"),
                  unsafe_allow_html=True)
    k[3].markdown(T.kpi_card("Model ROC-AUC", f"{fin['roc_auc']:.3f}",
                             f"tuned XGBoost · {len(test):,} test users", badge="AI"),
                  unsafe_allow_html=True)
    st.write("")

    left, mid, right = st.columns([2.15, 1.25, 1.2])

    # ---------------- retention by active days (hatched/solid bars) ----------------
    with left:
        with st.container(key="card_activity"):
            T.card_title("Retention by Week-1 Activity",
                         "Day-30 retention rate for users active on 0–7 days in their first week")
            g = df.groupby("days_active_week1")["retained_day30"].agg(["mean", "size"])
            vals = g["mean"].to_numpy()
            top = int(np.argmax(vals))
            # solid bars for days >= 4 (the habit zone), hatched below - like the reference
            colors, patterns = [], []
            for d in g.index:
                if d == g.index[top]:
                    colors.append(T.GREEN_900); patterns.append("")
                elif d >= 4:
                    colors.append(T.GREEN_700 if d % 2 else T.GREEN_500); patterns.append("")
                else:
                    colors.append("white"); patterns.append("/")
            fig = go.Figure(go.Bar(
                x=[f"{d}d" for d in g.index], y=vals,
                marker=dict(color=colors, cornerradius=18,
                            line=dict(color=T.GREEN_700, width=1.2),
                            pattern=dict(shape=patterns, fgcolor=T.GREEN_700,
                                         size=7, solidity=0.25)),
                text=[f"{v:.0%}" for v in vals], textposition="outside",
                textfont=dict(size=10, color=T.MUTED),
                customdata=g["size"],
                hovertemplate="%{x} active<br>Retention %{y:.1%}<br>%{customdata} users<extra></extra>"))
            fig.update_yaxes(visible=False, range=[0, 1.12])
            fig.update_layout(bargap=0.22)
            T.show(fig, height=232, key="dash_activity")

        # ---------------- user watchlist ----------------
        with st.container(key="card_watch"):
            T.card_title("User Risk Watchlist",
                         "Sample of test-set users with their predicted day-30 outcome")
            picks = (test.assign(bucket=pd_cut(test["retain_probability"]))
                     .groupby("bucket", observed=True)
                     .apply(lambda d: d.sample(min(len(d), 2), random_state=3),
                            include_groups=False)
                     .reset_index(level=0).sort_values("retain_probability").head(5))
            rows = ""
            for i, r in enumerate(picks.itertuples()):
                bg, fg = T.AVATAR_COLORS[i % len(T.AVATAR_COLORS)]
                label, tone = T.risk_of(r.retain_probability)
                sub = (f"{r.signup_channel.replace('_', ' ').title()} · {r.device_os} · "
                       f"active {r.days_active_week1}/7 days · {r.sessions_week1} sessions")
                rows += (f'<div class="row-item"><div class="avatar" style="background:{bg};color:{fg}">'
                         f'{r.user_id[-2:]}</div><div class="row-main"><div class="t">{r.user_id}'
                         f' <span style="color:{T.MUTED};font-weight:500">· {r.retain_probability:.0%} stay</span></div>'
                         f'<div class="s">{T.esc(sub)}</div></div>{T.pill(label, tone)}</div>')
            st.markdown(rows, unsafe_allow_html=True)

    # ---------------- insight card + gauge ----------------
    with mid:
        with st.container(key="card_insight"):
            hab = df[df["days_active_week1"] >= 4]["retained_day30"].mean()
            low = df[df["days_active_week1"] <= 2]["retained_day30"].mean()
            top_driver = next(iter(res["feature_importance"]["grouped"]))
            T.card_title("Key Insight")
            st.markdown(
                f'<div style="font-size:1.25rem;font-weight:700;color:{T.GREEN_900};line-height:1.25;'
                f'margin:.2rem 0 .35rem">Users active 4+ days stay {hab / low:.1f}× more</div>'
                f'<div class="card-sub">{hab:.0%} vs {low:.0%} retention · top driver: '
                f'{T.esc(friendly(top_driver))}</div>', unsafe_allow_html=True)
            if st.button("Check a user", icon=":material/person_search:", type="primary",
                         width="stretch", key="insight_btn"):
                go_to("predict")

        with st.container(key="card_gauge"):
            T.card_title("Predicted Retention", "Model view of the 2,000 test users")
            p = test["retain_probability"]
            a, b, c = (p >= 0.6).sum(), ((p >= 0.4) & (p < 0.6)).sum(), (p < 0.4).sum()
            tot = a + b + c
            fig = go.Figure(go.Pie(
                values=[a, b, c, tot], hole=0.72, sort=False, direction="clockwise",
                rotation=270, textinfo="none", hoverinfo="skip",
                marker=dict(colors=[T.GREEN_700, T.GREEN_900, "white", "rgba(0,0,0,0)"],
                            line=dict(color=["white", "white", T.GREEN_700, "rgba(0,0,0,0)"], width=2),
                            pattern=dict(shape=["", "", "/", ""], fgcolor=T.GREEN_700,
                                         size=6, solidity=0.3))))
            fig.add_annotation(text=f"<b>{a / tot:.0%}</b>", x=0.5, y=0.56, showarrow=False,
                               font=dict(size=34, color=T.INK))
            fig.add_annotation(text="Likely retained", x=0.5, y=0.43, showarrow=False,
                               font=dict(size=11, color=T.MUTED))
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
            fig.update_yaxes(domain=[0, 1])
            T.show(fig, height=200, key="dash_gauge")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:.74rem;color:{T.MUTED};margin-top:-2.4rem">'
                f'<span><span class="legend-dot" style="background:{T.GREEN_700}"></span>Retain {a}</span>'
                f'<span><span class="legend-dot" style="background:{T.GREEN_900}"></span>Unsure {b}</span>'
                f'<span><span class="legend-dot" style="background:repeating-linear-gradient(45deg,{T.GREEN_700} 0 2px,white 2px 4px);border:1px solid {T.GREEN_700}"></span>Churn {c}</span></div>',
                unsafe_allow_html=True)

    # ---------------- drivers list + dark metric card ----------------
    with right:
        with st.container(key="card_drivers"):
            T.card_title("Top Drivers", right_html=T.pill("XGBoost gain", "green"))
            rows = ""
            for i, (f, v) in enumerate(list(res["feature_importance"]["grouped"].items())[:5]):
                rows += (f'<div class="row-item"><div class="icon-sq" style="background:{T.ICON_COLORS[i]}">'
                         f'{i + 1}</div><div class="row-main"><div class="t">{T.esc(friendly(f))}</div>'
                         f'<div class="s">{v:.1%} of total split gain</div></div></div>')
            st.markdown(rows, unsafe_allow_html=True)

        with st.container(key="darkcard_acc"):
            st.markdown(
                f'<div style="font-weight:600;font-size:1.02rem">Model Accuracy</div>'
                f'<div class="big-num" style="margin:.5rem 0 .7rem">{fin["accuracy"]:.1%}</div>'
                f'<div style="display:flex;gap:.4rem;flex-wrap:wrap">'
                f'<span class="chip">F1 {fin["f1"]:.3f}</span>'
                f'<span class="chip">Recall {fin["recall"]:.3f}</span></div>',
                unsafe_allow_html=True)


def pd_cut(p):
    import pandas as pd
    return pd.cut(p, [0, 0.4, 0.6, 1.0001], labels=["risk", "unsure", "safe"],
                  include_lowest=True)
