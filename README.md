# RetainIQ — Predict Mobile App User Retention using XGBoost

ML Practical Mini-Project (A.Y. 2025-26) · Project #38
Shravani Yalagatti · CSBS, JSPM's RSCOE Pune

Predicts whether a new mobile-app user will still be active on **day 30**, using only
what they did in their **first 7 days**. Includes a tuned XGBoost model, an XGBoost
implementation written from scratch, a comparison with 7 other models, and a
Streamlit dashboard.

---

## How to run (Windows, step by step)

1. Install **Python 3.10 or newer** from python.org (tick *"Add Python to PATH"*).
2. Unzip this folder and open it in **VS Code** (File → Open Folder).
3. Open a terminal in VS Code (Terminal → New Terminal) and run:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   (Mac/Linux: `source venv/bin/activate` instead of the second line.)

4. Start the dashboard:

   ```bash
   streamlit run app.py
   ```
   Your browser opens at http://localhost:8501.

The trained model is already included in `models/`. To rebuild everything from zero:

```bash
python data/generate_data.py   # creates data/app_users.csv (10,120 rows)
python src/train.py            # preprocessing, scratch model, tuning, 8 models (~1 min)
```

If you delete the `models/` folder, the app retrains automatically on first launch.

---

## Project structure

```
app-retention-xgboost/
├── app.py                    # Streamlit entry point (sidebar, top bar, page routing)
├── requirements.txt
├── .streamlit/config.toml    # green theme
├── assets/                   # logo
├── data/
│   ├── generate_data.py      # realistic synthetic dataset generator
│   └── app_users.csv
├── src/
│   ├── preprocess.py         # cleaning pipeline (impute, IQR capping, scaling, one-hot)
│   ├── scratch_xgboost.py    # XGBoost from scratch in NumPy (rubric item 3)
│   └── train.py              # training, tuning, evaluation, comparison
├── models/                   # saved pipeline, results.json, test predictions
└── ui/
    ├── theme.py              # CSS + Plotly template (dashboard look)
    ├── data.py               # cached loading, prediction + SHAP helpers
    ├── nav.py
    └── pages/                # dashboard, predict, analytics, explorer, comparison, about
```

## Results (test set, 2,000 users)

| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| **XGBoost (tuned)** | **0.9255** | **0.8187** | **0.8485** |
| Gradient Boosting (sklearn) | 0.9239 | 0.8142 | 0.8455 |
| XGBoost (from scratch) | 0.9232 | 0.8180 | 0.8485 |
| Logistic Regression | 0.9204 | 0.8153 | 0.8475 |
| Random Forest | 0.9192 | 0.8150 | 0.8470 |
| XGBoost (default) | 0.9086 | 0.7971 | 0.8315 |
| KNN (k=25) | 0.9068 | 0.7932 | 0.8305 |
| Decision Tree | 0.9017 | 0.7963 | 0.8335 |

## Rubric mapping

| Criterion | Where |
|---|---|
| Dataset understanding | Data Explorer → Dataset & dictionary |
| Preprocessing | `src/preprocess.py`, Data Explorer → Preprocessing |
| Manual implementation | `src/scratch_xgboost.py`, Model Analytics → from scratch |
| Modelling | `src/train.py`, Predict page |
| Parameter tuning | RandomizedSearchCV (30 × 5-fold), Model Analytics → tuning |
| Evaluation + manual calc | Model Analytics → confusion matrix & manual calculation |
| Comparison | Comparison page |
| Visualization | every page |
| Interpretation | Dashboard insight, Predict → "Why this prediction?", Comparison → "Why" |

**Note:** the dataset is synthetic (generated with realistic behaviour rules) because real
app-retention data is private company data.
