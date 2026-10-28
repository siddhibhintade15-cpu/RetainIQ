"""
generate_data.py
----------------
Creates a realistic (synthetic) dataset of mobile-app users.

Why synthetic?  Real retention data is private company data, so we simulate
it.  Each row is ONE user, described by what they did in their FIRST 7 DAYS
after installing the app.  The target column `retained_day30` says whether
that user was still active on day 30 (1 = retained, 0 = churned).

The rules used to create the target are realistic (more active days, finished
onboarding, push enabled -> more likely to stay; crashes, support tickets ->
more likely to leave) and include NON-LINEAR effects and INTERACTIONS, which
is exactly where tree-based models like XGBoost shine.

We also deliberately add the "messiness" of real data:
  * missing values (age, avg_session_minutes, city_tier, ...)
  * outliers       (a few users with absurd session lengths)
  * duplicate rows (logging glitches)
so that the preprocessing step has real work to do.

Run:  python data/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)          # fixed seed -> same data every run
N_USERS = 10_000
OUT_PATH = Path(__file__).resolve().parent / "app_users.csv"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(n=N_USERS):
    # ---------------- categorical / profile features ----------------
    signup_channel = RNG.choice(
        ["organic", "paid_ads", "referral", "social_media"], size=n,
        p=[0.38, 0.30, 0.14, 0.18])
    device_os = RNG.choice(["Android", "iOS"], size=n, p=[0.72, 0.28])
    city_tier = RNG.choice(["Tier 1", "Tier 2", "Tier 3"], size=n,
                           p=[0.42, 0.35, 0.23])
    subscription_plan = RNG.choice(["free", "trial", "premium"], size=n,
                                   p=[0.68, 0.22, 0.10])
    age = np.clip(RNG.normal(28, 8, n), 16, 65).round()

    # referral users are a bit more engaged from day one
    channel_boost = pd.Series(signup_channel).map(
        {"organic": 0.3, "paid_ads": -0.4, "referral": 0.7,
         "social_media": 0.0}).to_numpy()
    plan_boost = pd.Series(subscription_plan).map(
        {"free": 0.0, "trial": 0.4, "premium": 1.1}).to_numpy()

    # latent "engagement" drives most behaviour -> realistic correlations
    engagement = RNG.normal(0, 1, n) + 0.5 * channel_boost + 0.4 * plan_boost

    # ---------------- first-week behaviour features ----------------
    days_active_week1 = np.clip(
        np.round(3 + 1.6 * engagement + RNG.normal(0, 1.1, n)), 0, 7).astype(int)
    sessions_week1 = RNG.poisson(np.clip(1.5 + 2.2 * days_active_week1
                                         + engagement, 0.2, None))
    sessions_week1 = np.maximum(sessions_week1, days_active_week1)
    avg_session_minutes = np.clip(
        RNG.gamma(2.2, 3.2, n) * np.exp(0.18 * engagement), 0.5, 60).round(1)
    features_used = np.clip(
        np.round(2 + 1.4 * engagement + 0.5 * days_active_week1
                 + RNG.normal(0, 1.3, n)), 1, 15).astype(int)
    onboarding_completed = (sigmoid(0.9 + 1.1 * engagement
                                    + RNG.normal(0, 0.8, n)) > 0.5).astype(int)
    push_enabled = (RNG.random(n) < sigmoid(0.3 + 0.4 * engagement)).astype(int)
    push_opens_week1 = np.where(
        push_enabled == 1,
        RNG.poisson(np.clip(1 + 1.2 * engagement, 0.1, None)), 0)
    friends_added = RNG.poisson(np.clip(
        0.6 + 0.9 * engagement + (signup_channel == "referral") * 1.5,
        0.05, None))
    crashes_week1 = RNG.poisson(np.where(device_os == "Android", 0.9, 0.45)
                                * (1 + (city_tier == "Tier 3") * 0.6))
    support_tickets = RNG.poisson(0.15 + 0.35 * (crashes_week1 > 2))
    in_app_purchases = RNG.poisson(np.clip(
        0.1 + 0.5 * plan_boost + 0.3 * engagement, 0.01, None))
    on_latest_version = (RNG.random(n) < 0.78).astype(int)

    # ---------------- target: retained on day 30 ----------------
    # Non-linear: the jump from 2 -> 4 active days matters far more than
    # 6 -> 7 (diminishing returns), crashes only hurt above a threshold,
    # push helps only if the user actually opens notifications.
    logit = (
        -4.4
        + 1.25 * np.tanh((days_active_week1 - 3) / 1.5)
        + 0.55 * np.log1p(sessions_week1)
        + 0.06 * np.minimum(avg_session_minutes, 25)
        + 0.85 * onboarding_completed
        + 0.35 * push_enabled * np.minimum(push_opens_week1, 4) / 2
        + 0.30 * np.minimum(friends_added, 5)
        + 0.12 * features_used
        - 0.65 * np.maximum(crashes_week1 - 1, 0)
        - 0.55 * support_tickets
        + 0.55 * np.minimum(in_app_purchases, 3)
        + 0.30 * plan_boost
        + 0.25 * channel_boost
        + 0.35 * on_latest_version
        # interaction: crashes hurt NEW (un-onboarded) users much more
        - 1.6 * (crashes_week1 >= 2) * (1 - onboarding_completed)
        # interaction: paid-ads users who never come back after day 2
        - 1.5 * (signup_channel == "paid_ads") * (days_active_week1 <= 2)
        # "habit formed": active 4+ days AND 10+ sessions -> big boost
        + 1.3 * ((days_active_week1 >= 4) & (sessions_week1 >= 10))
        # social lock-in only matters once the user has 3+ friends
        + 1.0 * (friends_added >= 3) * (features_used >= 5)
        # age effect is U-shaped (students and older users churn more)
        - 0.03 * np.abs(age - 27)
        # things we cannot observe (exams, a new phone, a friend quitting...)
        + RNG.normal(0, 1.1, n)
    )
    prob = sigmoid(logit)
    retained = (RNG.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "user_id": [f"U{100000 + i}" for i in range(n)],
        "age": age,
        "device_os": device_os,
        "city_tier": city_tier,
        "signup_channel": signup_channel,
        "subscription_plan": subscription_plan,
        "on_latest_version": on_latest_version,
        "onboarding_completed": onboarding_completed,
        "push_enabled": push_enabled,
        "days_active_week1": days_active_week1,
        "sessions_week1": sessions_week1,
        "avg_session_minutes": avg_session_minutes,
        "features_used": features_used,
        "push_opens_week1": push_opens_week1,
        "friends_added": friends_added,
        "in_app_purchases": in_app_purchases,
        "crashes_week1": crashes_week1,
        "support_tickets": support_tickets,
        "retained_day30": retained,
    })

    # ---------------- add real-world messiness ----------------
    df["age"] = df["age"].astype("float")
    for col, frac in [("age", 0.06), ("avg_session_minutes", 0.04),
                      ("city_tier", 0.03), ("push_opens_week1", 0.02)]:
        idx = RNG.choice(n, size=int(frac * n), replace=False)
        df.loc[idx, col] = np.nan

    # outliers: app left open in background -> absurd session minutes
    out_idx = RNG.choice(n, size=60, replace=False)
    df.loc[out_idx, "avg_session_minutes"] = RNG.uniform(180, 900, 60).round(1)

    # duplicate rows from a logging glitch
    dup = df.sample(120, random_state=7)
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1, random_state=11).reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(data):,} rows x {data.shape[1]} columns -> {OUT_PATH}")
    print(f"Retention rate: {data['retained_day30'].mean():.1%}")
    print("Missing values per column:")
    print(data.isna().sum()[data.isna().sum() > 0])
