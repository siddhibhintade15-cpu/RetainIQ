"""
app.py  -  RetainIQ dashboard
-----------------------------
Streamlit front-end for "Predict Mobile App User Retention using XGBoost".

Run:   streamlit run app.py
"""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="RetainIQ · App Retention Predictor",
                   page_icon=":material/donut_large:", layout="wide",
                   initial_sidebar_state="expanded")

from ui import nav, theme as T  # noqa: E402
from ui.data import ensure_trained, load_raw, load_test_predictions  # noqa: E402
from ui.pages import about, analytics, comparison, dashboard, explorer, predict  # noqa: E402

T.inject_css()
T.register_plotly_template()
ensure_trained()

# ---------------- pages (sidebar menu) ----------------
pages = {
    "dashboard": st.Page(dashboard.render, title="Dashboard", icon=":material/space_dashboard:",
                         url_path="dashboard", default=True),
    "predict": st.Page(predict.render, title="Predict User", icon=":material/person_search:",
                       url_path="predict"),
    "analytics": st.Page(analytics.render, title="Model Analytics", icon=":material/monitoring:",
                         url_path="analytics"),
    "explorer": st.Page(explorer.render, title="Data Explorer", icon=":material/table_chart:",
                        url_path="explorer"),
    "comparison": st.Page(comparison.render, title="Comparison", icon=":material/leaderboard:",
                          url_path="comparison"),
    "about": st.Page(about.render, title="About Project", icon=":material/info:",
                     url_path="about"),
}
for name, page in pages.items():
    nav.register(name, page)

ASSETS = Path(__file__).resolve().parent / "assets"
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))

current = st.navigation({
    "Menu": [pages[k] for k in ("dashboard", "predict", "analytics", "explorer", "comparison")],
    "General": [pages["about"]],
})

with st.sidebar:
    st.markdown('<div class="side-promo"><div class="t">Score your own users</div>'
                '<div class="s">Download the dataset or upload a CSV on the Predict page.</div></div>',
                unsafe_allow_html=True)
    st.download_button("Download dataset", load_raw().to_csv(index=False), "app_users.csv",
                       "text/csv", type="primary", width="stretch", icon=":material/download:",
                       key="side_dl")

# ---------------- top bar: user search + profile ----------------
t1, t2 = st.columns([1.3, 1], vertical_alignment="center")
with t1:
    query = st.text_input("Search", placeholder="Search a test user ID, e.g. U100311",
                          label_visibility="collapsed", key="topsearch",
                          icon=":material/search:")
with t2:
    st.markdown(
        '<div class="profile"><div class="ic">&#9993;</div><div class="ic">&#128276;</div>'
        '<div class="me"><div class="av">SY</div><div><div class="n">Siddhi Bhintade</div>'
        '<div class="e">CSBS · RSCOE Pune</div></div></div></div>', unsafe_allow_html=True)

if query:
    test = load_test_predictions()
    hit = test[test["user_id"].str.upper() == query.strip().upper()]
    with st.container(key="card_search"):
        if hit.empty:
            st.markdown(f"No test-set user **{T.esc(query)}** found. Try one from the watchlist on "
                        "the Dashboard, or use the Predict page for a new user.")
        else:
            r = hit.iloc[0]
            label, tone = T.risk_of(r["retain_probability"])
            actual = "retained" if r["retained_day30"] == 1 else "churned"
            st.markdown(
                f'<div class="row-item"><div class="avatar" style="background:{T.GREEN_100};'
                f'color:{T.GREEN_900}">{r["user_id"][-2:]}</div><div class="row-main">'
                f'<div class="t">{r["user_id"]} · predicted {r["retain_probability"]:.0%} chance to stay</div>'
                f'<div class="s">{r["signup_channel"]} · {r["device_os"]} · {r["subscription_plan"]} plan · '
                f'active {r["days_active_week1"]}/7 days · {r["sessions_week1"]} sessions · '
                f'{r["crashes_week1"]} crashes · actually <b>{actual}</b></div></div>'
                f'{T.pill(label, tone)}</div>', unsafe_allow_html=True)

current.run()
