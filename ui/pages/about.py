"""About page - project summary, rubric mapping, how to run."""

import pandas as pd
import streamlit as st

from ui import theme as T
from ui.data import load_results

RUBRIC = [
    ("1. Dataset Understanding", 10, "Data Explorer → Dataset & dictionary", "data/generate_data.py, report §2"),
    ("2. Data Preprocessing", 10, "Data Explorer → Preprocessing", "src/preprocess.py"),
    ("3. Algorithm Implementation (manual)", 15, "Model Analytics → XGBoost from scratch", "src/scratch_xgboost.py"),
    ("4. Algorithm Application / Modeling", 15, "Model Analytics, Predict", "src/train.py (XGBClassifier)"),
    ("5. Parameter Tuning", 10, "Model Analytics → Hyper-parameter tuning", "src/train.py (RandomizedSearchCV)"),
    ("6. Model Evaluation + manual calc", 10, "Model Analytics → Manual metric calculation", "src/train.py"),
    ("7. Comparison / Analysis", 10, "Model Comparison", "src/train.py (8 models)"),
    ("8. Visualization", 10, "All pages (custom Plotly charts)", "ui/pages/*.py"),
    ("9. Interpretation & Discussion", 10, "Dashboard insight, Predict → Why, Comparison → Why", "report §7"),
    ("10. Code Documentation", 5, "-", "docstrings + comments in every file, README.md"),
    ("11. Report / Presentation", 5, "-", "Project report (Docs)"),
]


def render():
    res = load_results()
    T.page_header("About the Project",
                  "Predict Mobile App User Retention using XGBoost · ML Practical Mini-Project (A.Y. 2025-26)")
    st.write("")
    a, b = st.columns([1.3, 1])
    with a:
        with st.container(key="card_about"):
            T.card_title("Problem statement")
            st.markdown(
                "Most mobile apps lose the majority of new users within the first month. "
                "Acquiring a user is expensive, so knowing **in week 1** who is likely to leave by "
                "**day 30** lets the product team step in early (onboarding nudges, bug fixes, offers).\n\n"
                "This project trains an **XGBoost** classifier on first-week behaviour to predict "
                "`retained_day30`, explains each prediction with SHAP contributions, and compares XGBoost "
                "with seven other approaches - including our own from-scratch implementation.")
            T.card_title("Tech stack")
            st.markdown(" ".join(T.pill(t) for t in [
                "Python 3", "pandas", "NumPy", "scikit-learn", "XGBoost", "Streamlit", "Plotly", "joblib"]),
                unsafe_allow_html=True)
    with b:
        with st.container(key="darkcard_run"):
            st.markdown("**How to run**")
            st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
            st.markdown("Retrain from scratch any time:")
            st.code("python data/generate_data.py\npython src/train.py", language="bash")
            st.markdown(f"Last trained: {res['meta']['trained_at']} · "
                        f"{res['meta']['total_time_sec']:.0f}s")

    with st.container(key="card_rubric"):
        T.card_title("Where each grading criterion is covered", "100 points total")
        st.dataframe(pd.DataFrame(RUBRIC, columns=["Criterion", "Points", "In the app", "In the code"]),
                     hide_index=True, width="stretch", height=420)
