"""Predict page - single user form + batch CSV scoring, with explanations."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.preprocess import FEATURES
from ui import theme as T
from ui.data import explain_one, load_raw, predict_proba, recommendations

PRESETS = {
    "Engaged referral user": dict(
        age=24, device_os="iOS", city_tier="Tier 1", signup_channel="referral",
        subscription_plan="trial", on_latest_version=1, onboarding_completed=1,
        push_enabled=1, days_active_week1=6, sessions_week1=16,
        avg_session_minutes=11.0, features_used=7, push_opens_week1=3,
        friends_added=4, in_app_purchases=1, crashes_week1=0, support_tickets=0),
    "At-risk paid-ads user": dict(
        age=35, device_os="Android", city_tier="Tier 3", signup_channel="paid_ads",
        subscription_plan="free", on_latest_version=0, onboarding_completed=0,
        push_enabled=0, days_active_week1=1, sessions_week1=2,
        avg_session_minutes=3.5, features_used=2, push_opens_week1=0,
        friends_added=0, in_app_purchases=0, crashes_week1=3, support_tickets=1),
    "Borderline organic user": dict(
        age=29, device_os="Android", city_tier="Tier 2", signup_channel="organic",
        subscription_plan="free", on_latest_version=1, onboarding_completed=1,
        push_enabled=1, days_active_week1=3, sessions_week1=7,
        avg_session_minutes=6.0, features_used=4, push_opens_week1=1,
        friends_added=1, in_app_purchases=0, crashes_week1=1, support_tickets=0),
}


def gauge(p):
    label, tone = T.risk_of(p)
    color = {"green": T.GREEN_700, "amber": T.AMBER, "red": T.RED}[tone]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number=dict(suffix="%", font=dict(size=46, color=T.INK)),
        gauge=dict(axis=dict(range=[0, 100], tickwidth=0, tickcolor="white",
                             tickfont=dict(color=T.MUTED, size=10)),
                   bar=dict(color=color, thickness=0.32),
                   bgcolor=T.LINE, borderwidth=0,
                   steps=[dict(range=[0, 40], color="#F7E6E5"),
                          dict(range=[40, 60], color="#FBF0DE"),
                          dict(range=[60, 100], color=T.GREEN_100)],
                   threshold=dict(line=dict(color=T.INK, width=2), value=50,
                                  thickness=0.8))))
    fig.update_layout(margin=dict(l=35, r=35, t=20, b=0))
    return fig


def contrib_chart(expl):
    top = expl.head(9).iloc[::-1]
    pos = top["contribution"] >= 0
    fig = go.Figure(go.Bar(
        x=top["contribution"], y=[f"{l} = {v}" for l, v in zip(top["label"], top["value"])],
        orientation="h",
        marker=dict(color=[T.GREEN_700 if p else "white" for p in pos],
                    line=dict(color=[T.GREEN_700 if p else T.RED for p in pos], width=1.2),
                    pattern=dict(shape=["" if p else "/" for p in pos], fgcolor=T.RED,
                                 size=6, solidity=0.3),
                    cornerradius=6),
        hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>"))
    fig.add_vline(x=0, line_color=T.MUTED, line_width=1)
    fig.update_xaxes(title="← pushes towards churn   |   pushes towards retained →",
                     gridcolor=T.LINE, showgrid=True)
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    fig.update_layout(margin=dict(l=210, r=10, t=10, b=40))
    return fig


def single_user():
    raw = load_raw()
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    preset = c1.selectbox("Start from a preset", list(PRESETS), index=2)
    v = PRESETS[preset]

    with st.form("predict_form", border=False):
        with st.container(key="card_form"):
            T.card_title("User profile & first-week behaviour",
                         "Everything here is known 7 days after install - the model predicts day 30.")
            a, b, c = st.columns(3)
            with a:
                st.markdown("**Profile**")
                age = st.number_input("Age", 13, 80, int(v["age"]))
                device_os = st.selectbox("Device OS", ["Android", "iOS"],
                                         index=["Android", "iOS"].index(v["device_os"]))
                city_tier = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"],
                                         index=["Tier 1", "Tier 2", "Tier 3"].index(v["city_tier"]))
                chans = ["organic", "paid_ads", "referral", "social_media"]
                signup_channel = st.selectbox("Signup channel", chans,
                                              index=chans.index(v["signup_channel"]))
                plans = ["free", "trial", "premium"]
                subscription_plan = st.selectbox("Subscription plan", plans,
                                                 index=plans.index(v["subscription_plan"]))
            with b:
                st.markdown("**Engagement (week 1)**")
                days_active = st.slider("Days active (0-7)", 0, 7, int(v["days_active_week1"]))
                sessions = st.number_input("Sessions", 0, 60, int(v["sessions_week1"]))
                avg_min = st.number_input("Avg session length (min)", 0.0, 120.0,
                                          float(v["avg_session_minutes"]), step=0.5)
                features_used = st.slider("Distinct features used", 1, 15, int(v["features_used"]))
                friends = st.number_input("Friends added", 0, 50, int(v["friends_added"]))
            with c:
                st.markdown("**Experience & monetisation**")
                onboarding = st.toggle("Onboarding completed", bool(v["onboarding_completed"]))
                push_en = st.toggle("Push notifications enabled", bool(v["push_enabled"]))
                latest = st.toggle("On latest app version", bool(v["on_latest_version"]))
                push_opens = st.number_input("Push notifications opened", 0, 30,
                                             int(v["push_opens_week1"]))
                iap = st.number_input("In-app purchases", 0, 20, int(v["in_app_purchases"]))
                crashes = st.number_input("App crashes", 0, 20, int(v["crashes_week1"]))
                tickets = st.number_input("Support tickets", 0, 10, int(v["support_tickets"]))
            submitted = st.form_submit_button("Predict retention", icon=":material/bolt:")

    # Inside a form, widgets report their last SUBMITTED values (or the preset
    # defaults before the first submit), so the result always matches the form.
    row = pd.DataFrame([dict(
        age=age, device_os=device_os, city_tier=city_tier,
        signup_channel=signup_channel, subscription_plan=subscription_plan,
        on_latest_version=int(latest), onboarding_completed=int(onboarding),
        push_enabled=int(push_en), days_active_week1=days_active,
        sessions_week1=sessions, avg_session_minutes=avg_min,
        features_used=features_used, push_opens_week1=push_opens if push_en else 0,
        friends_added=friends, in_app_purchases=iap, crashes_week1=crashes,
        support_tickets=tickets)])

    p = float(predict_proba(row)[0])
    expl = explain_one(row)
    label, tone = T.risk_of(p)
    base = raw.drop_duplicates()["retained_day30"].mean()

    r1, r2 = st.columns([1, 1.6])
    with r1:
        with st.container(key="card_result"):
            T.card_title("Day-30 retention probability", right_html=T.pill(label, tone))
            T.show(gauge(p), height=220, key="pred_gauge")
            st.markdown(
                f'<div class="card-sub" style="text-align:center">Average user: {base:.0%} · '
                f'this user is <b>{"above" if p > base else "below"}</b> average '
                f'({(p - base) * 100:+.0f} pts)</div>', unsafe_allow_html=True)
        with st.container(key="card_actions"):
            T.card_title("Recommended actions",
                         "Targets the factors pulling this user towards churn")
            recs = recommendations(expl)
            if not recs:
                st.markdown("No negative drivers - keep this user engaged with regular "
                            "new content. ✅")
            for i, (lab, act) in enumerate(recs, 1):
                st.markdown(f'<div class="step"><div class="num">{i}</div><div class="body">'
                            f'<div class="t">{T.esc(lab)}</div><div class="s">{T.esc(act)}</div>'
                            f'</div></div>', unsafe_allow_html=True)
    with r2:
        with st.container(key="card_why"):
            T.card_title("Why this prediction?",
                         "Per-feature contribution (XGBoost TreeSHAP, log-odds). Green = helps "
                         "retention, hatched red = pushes towards churn.")
            T.show(contrib_chart(expl), height=400, key="pred_contrib")
            st.caption(f"Starting point (average log-odds): {expl.attrs['bias']:+.3f}. "
                       f"Bias + all contributions = {expl.attrs['bias'] + expl['contribution'].sum():+.3f} "
                       f"→ sigmoid → {p:.1%}.")


def batch():
    raw = load_raw()
    with st.container(key="card_batch"):
        T.card_title("Score many users at once",
                     "Upload a CSV with the same columns as the training data "
                     "(user_id optional). Missing values are handled automatically.")
        template = raw.drop(columns=["retained_day30"]).sample(25, random_state=1)
        c1, c2 = st.columns([1, 2], vertical_alignment="center")
        c1.download_button("Download CSV template", template.to_csv(index=False),
                           "retention_template.csv", "text/csv",
                           icon=":material/download:")
        up = c2.file_uploader("Upload users CSV", type=["csv"], label_visibility="collapsed")
        use_demo = st.toggle("Or use the 25-user template as a demo", value=up is None)

    data = pd.read_csv(up) if up is not None else (template if use_demo else None)
    if data is None:
        return
    try:
        probs = predict_proba(data)
    except ValueError as e:
        st.error(f"{e}. Expected columns: {', '.join(FEATURES)}")
        return
    out = data.copy()
    out["retain_probability"] = probs.round(4)
    out["prediction"] = [T.risk_of(p)[0] for p in probs]
    out = out.sort_values("retain_probability")
    cols = (["user_id"] if "user_id" in out else []) + ["retain_probability", "prediction",
                                                         "days_active_week1", "sessions_week1",
                                                         "onboarding_completed", "signup_channel",
                                                         "crashes_week1"]
    k = st.columns(3)
    k[0].markdown(T.kpi_card("Users scored", f"{len(out)}", "from uploaded file", dark=True),
                  unsafe_allow_html=True)
    k[1].markdown(T.kpi_card("Churn risk", f"{(probs < 0.4).sum()}", "probability below 40%"),
                  unsafe_allow_html=True)
    k[2].markdown(T.kpi_card("Avg. retention", f"{probs.mean():.0%}", "predicted for this group"),
                  unsafe_allow_html=True)
    st.write("")
    with st.container(key="card_batchtable"):
        T.card_title("Results (most at-risk first)")
        st.dataframe(out[cols], hide_index=True, width="stretch", height=380,
                     column_config={"retain_probability": st.column_config.ProgressColumn(
                         "Retain probability", min_value=0, max_value=1, format="percent")})
        st.download_button("Download predictions", out.to_csv(index=False),
                           "retention_predictions.csv", "text/csv", type="primary",
                           icon=":material/download:")


def render():
    T.page_header("Predict a User",
                  "Enter a user's first-week behaviour and the tuned XGBoost model "
                  "estimates whether they will still be active on day 30.")
    st.write("")
    t1, t2 = st.tabs(["Single user", "Batch upload"])
    with t1:
        single_user()
    with t2:
        batch()
