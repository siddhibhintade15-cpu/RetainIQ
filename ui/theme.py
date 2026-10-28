"""
theme.py
--------
Everything about the LOOK of the dashboard lives here:
  * CSS that turns Streamlit into the green, rounded "Donezo" style
  * a Plotly template so every chart matches
  * small HTML helpers (KPI cards, pills, list rows, page header)
"""

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------- colour tokens ----------------
GREEN_950 = "#0B2E1D"
GREEN_900 = "#0F3D26"
GREEN_700 = "#1E6B43"
GREEN_500 = "#2F9A63"
GREEN_300 = "#8ED1A9"
GREEN_100 = "#E3F4EA"
INK = "#1B1D1C"
MUTED = "#80867F"
LINE = "#ECEEEA"
BG = "#F4F5F2"
CARD = "#FFFFFF"
AMBER = "#E8A33D"
RED = "#D9534F"

FONT = "'Plus Jakarta Sans', 'Inter', 'Segoe UI', sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea, select {{
    font-family: {FONT} !important;
}}
/* keep Material icon ligatures working after the font override */
[data-testid="stIconMaterial"], .material-symbols-rounded, span[translate="no"] {{
    font-family: 'Material Symbols Rounded' !important;
}}
.stApp {{ background: {BG}; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding-top: 1.4rem !important; padding-bottom: 3rem !important;
                    max-width: 1320px; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding-top: .4rem; }}
[data-testid="stSidebarLogo"], [data-testid="stLogo"] {{ height: 2.6rem !important; max-width: 12rem; }}
[data-testid="stSidebarNav"] span, [data-testid="stNavSectionHeader"] {{
    font-weight: 500;
}}
[data-testid="stNavSectionHeader"] {{
    color: {MUTED} !important; font-size: .74rem !important;
    letter-spacing: .02em; margin-top: .8rem;
}}
[data-testid="stSidebarNavLink"] {{
    border-radius: 12px; padding: .55rem .8rem; margin: 2px 0;
    color: {MUTED};
    position: relative;
}}
[data-testid="stSidebarNavLink"] span {{ color: {MUTED}; font-size: .95rem; }}
[data-testid="stSidebarNavLink"]:hover {{ background: {GREEN_100}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: transparent; }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{
    color: {INK} !important; font-weight: 700;
}}
[data-testid="stSidebarNavLink"][aria-current="page"] [data-testid="stIconMaterial"] {{
    color: {GREEN_700} !important;
}}
[data-testid="stSidebarNavLink"][aria-current="page"]::before {{
    content: ""; position: absolute; left: -1rem; top: 18%; height: 64%;
    width: 4px; border-radius: 0 4px 4px 0; background: {GREEN_700};
}}

/* ---------- cards (any container whose key starts with "card") ---------- */
div[class*="st-key-card"] {{
    background: {CARD}; border-radius: 22px; padding: 1.1rem 1.2rem 1rem;
    border: 1px solid {LINE};
}}
div[class*="st-key-darkcard"] {{
    background: radial-gradient(120% 140% at 100% 0%, {GREEN_700} 0%, {GREEN_900} 55%, {GREEN_950} 100%);
    border-radius: 22px; padding: 1.1rem 1.2rem; color: white;
}}
div[class*="st-key-darkcard"] * {{ color: white !important; }}
div[class*="st-key-darkcard"] pre, div[class*="st-key-darkcard"] code {{ background: rgba(255,255,255,.10) !important; }}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    border-radius: 999px !important; padding: .5rem 1.25rem !important;
    font-weight: 600 !important; border: 1.5px solid {GREEN_900} !important;
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button,
.stDownloadButton > button[kind="primary"] {{
    background: linear-gradient(180deg, {GREEN_700}, {GREEN_900}) !important;
    color: white !important;
}}
.stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {{
    background: white !important; color: {GREEN_900} !important;
}}

/* ---------- inputs ---------- */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{
    border-radius: 12px !important; background: {BG} !important;
    border-color: {LINE} !important;
}}
div[class*="st-key-topsearch"] [data-baseweb="input"] {{
    border-radius: 999px !important; background: {CARD} !important;
}}
[data-testid="stSlider"] [role="slider"] {{ background: {GREEN_700}; }}

/* tabs */
[role="tablist"] {{ gap: .4rem; border: none !important; box-shadow: none !important; }}
[role="tablist"]::after, [role="tablist"]::before, [data-testid="stTab"]::after {{ display: none !important; }}
[data-testid="stTab"] {{
    border-radius: 999px !important; padding: .4rem 1.1rem !important; border-bottom: 1px solid {LINE} !important;
    background: {CARD}; border: 1px solid {LINE};
}}
[data-testid="stTab"][aria-selected="true"] {{
    background: {GREEN_900} !important; color: white !important;
}}
[data-testid="stTab"][aria-selected="true"] p {{ color: white !important; }}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display: none; }}

/* metrics, dataframes, expanders */
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}
[data-testid="stExpander"] details {{ border-radius: 16px; border-color: {LINE};
                                      background: {CARD}; }}

/* ---------- custom HTML components ---------- */
.page-head h1 {{ font-size: 2.1rem; font-weight: 700; color: {INK};
                margin: 0; padding: 0; letter-spacing: -.02em; }}
.page-head p {{ color: {MUTED}; margin: .25rem 0 .9rem; font-size: .95rem; }}
.card-title {{ font-weight: 600; font-size: 1.05rem; color: {INK};
              margin-bottom: .2rem; display:flex; justify-content:space-between;
              align-items:center; }}
.card-sub {{ color: {MUTED}; font-size: .82rem; margin-bottom: .4rem; }}

.kpi {{ border-radius: 22px; padding: 1.05rem 1.15rem 1rem; background: {CARD};
        border: 1px solid {LINE}; height: 100%; min-height: 152px; position: relative; }}
.kpi.dark {{ background: linear-gradient(140deg, {GREEN_700} 0%, {GREEN_900} 70%);
             border: none; color: white; }}
.kpi .label {{ font-weight: 600; font-size: 1.02rem; color: {INK}; }}
.kpi.dark .label {{ color: white; }}
.kpi .value {{ font-size: 2.7rem; font-weight: 600; color: {INK}; margin: .45rem 0 .35rem;
               letter-spacing: -.02em; line-height: 1.05; }}
.kpi.dark .value {{ color: white; }}
.kpi .foot {{ font-size: .76rem; color: {GREEN_700}; display:flex; align-items:center; gap:.4rem; }}
.kpi.dark .foot {{ color: {GREEN_300}; }}
.kpi .badge {{ border: 1px solid currentColor; border-radius: 6px; padding: 0 .3rem;
               font-size: .68rem; font-weight: 600; }}
.kpi .arrow {{ position: absolute; top: 1rem; right: 1rem; width: 30px; height: 30px;
               border-radius: 50%; border: 1.5px solid {INK}; display: grid;
               place-items: center; font-size: .9rem; color: {INK}; }}
.kpi.dark .arrow {{ background: white; border-color: white; color: {GREEN_900}; }}

.pill {{ display:inline-block; font-size: .7rem; font-weight: 600; border-radius: 6px;
         padding: .12rem .45rem; white-space: nowrap; }}
.pill.green {{ background: {GREEN_100}; color: {GREEN_700}; }}
.pill.amber {{ background: #FCEFD9; color: #A9661A; }}
.pill.red   {{ background: #FBE3E2; color: #B23B37; }}

.row-item {{ display:flex; align-items:center; gap:.7rem; padding:.48rem 0; }}
.row-item + .row-item {{ border-top: 1px dashed {LINE}; }}
.avatar {{ width: 36px; height: 36px; border-radius: 50%; display:grid; place-items:center;
           font-weight: 700; font-size: .78rem; flex-shrink: 0; }}
.row-main {{ flex: 1; min-width: 0; }}
.row-main .t {{ font-weight: 600; font-size: .9rem; color: {INK}; }}
.row-main .s {{ font-size: .74rem; color: {MUTED}; white-space: nowrap; overflow: hidden;
                text-overflow: ellipsis; }}
.icon-sq {{ width: 30px; height: 30px; border-radius: 9px; display:grid; place-items:center;
            color: white; font-weight: 700; font-size: .8rem; flex-shrink: 0; }}

.profile {{ display:flex; align-items:center; justify-content:flex-end; gap:.7rem; }}
.profile .ic {{ width: 40px; height: 40px; border-radius: 50%; background: {CARD};
                border: 1px solid {LINE}; display:grid; place-items:center; color: {INK}; }}
.profile .me {{ display:flex; align-items:center; gap:.6rem; }}
.profile .me .av {{ width: 42px; height: 42px; border-radius: 50%;
                    background: linear-gradient(140deg, {GREEN_300}, {GREEN_700});
                    color:white; display:grid; place-items:center; font-weight:700; }}
.profile .me .n {{ font-weight: 700; font-size: .92rem; color: {INK}; line-height: 1.1; }}
.profile .me .e {{ font-size: .76rem; color: {MUTED}; }}

.big-num {{ font-size: 2.6rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.05; }}
.chip {{ display:inline-flex; align-items:center; gap:.35rem; border-radius:999px;
         padding:.3rem .75rem; font-size:.78rem; font-weight:600;
         background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); }}
.legend-dot {{ display:inline-block; width:10px; height:10px; border-radius:50%;
               margin-right:.35rem; vertical-align: middle; }}
.formula {{ background: {BG}; border-radius: 12px; padding: .65rem .85rem;
            font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .82rem;
            color: {INK}; border: 1px solid {LINE}; margin: .3rem 0; overflow-x:auto; }}
.step {{ display:flex; gap:.8rem; padding:.55rem 0; }}
.step .num {{ width: 26px; height: 26px; border-radius: 50%; background: {GREEN_900};
              color: white; display:grid; place-items:center; font-size:.75rem;
              font-weight:700; flex-shrink:0; }}
.step .body .t {{ font-weight: 600; font-size: .92rem; }}
.step .body .s {{ color: {MUTED}; font-size: .8rem; }}
.side-promo {{ border-radius: 20px; padding: 1rem; color: white; margin-top: 1.2rem;
   background: radial-gradient(120% 120% at 100% 100%, {GREEN_500} 0%, {GREEN_900} 55%, {GREEN_950} 100%); }}
.side-promo .t {{ font-weight: 700; font-size: 1.02rem; line-height: 1.25; }}
.side-promo .s {{ font-size: .76rem; opacity: .75; margin-top: .25rem; }}
.brand {{ display:flex; align-items:center; gap:.55rem; font-weight:800; font-size:1.25rem;
          color:{INK}; padding: .2rem 0 .4rem .2rem; }}
.brand .logo {{ width: 34px; height: 34px; border-radius: 50%; border: 3px solid {GREEN_700};
                display:grid; place-items:center; }}
.brand .logo::after {{ content:""; width: 12px; height: 12px; border-radius: 50%;
                       background: {GREEN_700}; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------- Plotly template ----------------
def register_plotly_template():
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        font=dict(family=FONT, color=INK, size=12),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        colorway=[GREEN_700, GREEN_300, GREEN_900, GREEN_500, AMBER, "#9AA39C", RED],
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=LINE, tickcolor=LINE,
                   title=dict(font=dict(color=MUTED, size=11))),
        yaxis=dict(gridcolor=LINE, zeroline=False, linecolor=LINE,
                   title=dict(font=dict(color=MUTED, size=11))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(family=FONT, color=INK)),
        bargap=0.28,
    )
    pio.templates["retain"] = tmpl
    pio.templates.default = "retain"


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def show(fig, height=300, key=None):
    fig.update_layout(height=height, paper_bgcolor=CARD, plot_bgcolor=CARD,
                      margin=dict(autoexpand=True))
    fig.update_xaxes(automargin=True, title_standoff=8)
    fig.update_yaxes(automargin=True, title_standoff=8)
    st.plotly_chart(fig, config=PLOT_CONFIG, key=key, width="stretch", theme=None)


# ---------------- HTML helpers ----------------
def esc(x):
    return html.escape(str(x))


def page_header(title, subtitle):
    st.markdown(f'<div class="page-head"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>',
                unsafe_allow_html=True)


def card_title(title, sub=None, right_html=""):
    s = f'<div class="card-title"><span>{esc(title)}</span>{right_html}</div>'
    if sub:
        s += f'<div class="card-sub">{esc(sub)}</div>'
    st.markdown(s, unsafe_allow_html=True)


def kpi_card(label, value, foot, badge="", dark=False):
    cls = "kpi dark" if dark else "kpi"
    badge_html = f'<span class="badge">{esc(badge)}</span>' if badge else ""
    return (f'<div class="{cls}"><div class="arrow">&#8599;</div>'
            f'<div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
            f'<div class="foot">{badge_html}<span>{esc(foot)}</span></div></div>')


def pill(text, tone="green"):
    return f'<span class="pill {tone}">{esc(text)}</span>'


def risk_of(p):
    if p >= 0.6:
        return "Likely retained", "green"
    if p >= 0.4:
        return "Uncertain", "amber"
    return "Churn risk", "red"


AVATAR_COLORS = [("#F6D7D7", "#8A3B3B"), ("#DDEFD9", "#2E6B2A"), ("#DCE3F7", "#34468A"),
                 ("#F7E7CF", "#8A5A1E"), ("#E6DDF4", "#5A3C8A"), ("#D7EEF0", "#22666C")]
ICON_COLORS = ["#3D5AFE", "#2BB3A3", "#F2B544", "#7C4DFF", "#E8603C", GREEN_700,
               "#00897B", "#5C6BC0"]
