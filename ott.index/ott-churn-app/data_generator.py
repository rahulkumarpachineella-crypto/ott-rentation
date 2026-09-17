"""
Synthetic OTT (streaming platform) user dataset generator.
Creates realistic usage/behavioral features and a churn label whose
probability depends on those features (so a model can learn real patterns).
"""
import numpy as np
import pandas as pd


def generate_ott_dataset(n_users: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    plan_types = rng.choice(["Basic", "Standard", "Premium"], size=n_users, p=[0.35, 0.40, 0.25])
    plan_fee_map = {"Basic": 4.99, "Standard": 9.99, "Premium": 15.99}
    monthly_fee = np.array([plan_fee_map[p] for p in plan_types])

    subscription_tenure_months = rng.integers(1, 60, size=n_users)
    age = rng.integers(16, 70, size=n_users)
    num_devices = rng.integers(1, 5, size=n_users)
    uses_multiple_profiles = rng.choice([0, 1], size=n_users, p=[0.45, 0.55])

    # Engagement features
    avg_watch_hours_per_week = np.clip(rng.normal(6, 4, size=n_users), 0, 30)
    days_since_last_login = np.clip(rng.exponential(5, size=n_users), 0, 90).astype(int)
    num_genres_watched = rng.integers(1, 10, size=n_users)
    content_completion_rate = np.clip(rng.normal(0.55, 0.2, size=n_users), 0.02, 1.0)
    watch_hours_trend_pct = rng.normal(-5, 25, size=n_users)  # % change vs prior month

    # Friction / risk features
    customer_support_tickets = rng.poisson(0.6, size=n_users)
    payment_failures_last_6m = rng.poisson(0.25, size=n_users)
    downgraded_plan_last_6m = rng.choice([0, 1], size=n_users, p=[0.85, 0.15])
    price_increase_recently = rng.choice([0, 1], size=n_users, p=[0.75, 0.25])
    referred_by_friend = rng.choice([0, 1], size=n_users, p=[0.7, 0.3])

    # --- Build a latent "churn score" from weighted risk factors, then convert to probability ---
    z = (
        -1.6
        - 0.55 * (avg_watch_hours_per_week - 6) / 4
        + 0.85 * (days_since_last_login - 5) / 5
        - 0.55 * (content_completion_rate - 0.55) / 0.2
        - 0.10 * (subscription_tenure_months - 30) / 15
        + 0.55 * (customer_support_tickets)
        + 0.80 * (payment_failures_last_6m)
        + 0.70 * downgraded_plan_last_6m
        + 0.45 * price_increase_recently
        - 0.35 * referred_by_friend
        - 0.25 * uses_multiple_profiles
        - 0.45 * (watch_hours_trend_pct) / 25
        - 0.20 * (num_genres_watched - 5) / 3
        + rng.normal(0, 0.45, size=n_users)  # noise
    )
    churn_probability_true = 1 / (1 + np.exp(-z))
    churn = rng.binomial(1, churn_probability_true)

    df = pd.DataFrame({
        "user_id": [f"U{100000+i}" for i in range(n_users)],
        "age": age,
        "plan_type": plan_types,
        "monthly_fee": monthly_fee,
        "subscription_tenure_months": subscription_tenure_months,
        "num_devices": num_devices,
        "uses_multiple_profiles": uses_multiple_profiles,
        "avg_watch_hours_per_week": avg_watch_hours_per_week.round(2),
        "days_since_last_login": days_since_last_login,
        "num_genres_watched": num_genres_watched,
        "content_completion_rate": content_completion_rate.round(3),
        "watch_hours_trend_pct": watch_hours_trend_pct.round(1),
        "customer_support_tickets": customer_support_tickets,
        "payment_failures_last_6m": payment_failures_last_6m,
        "downgraded_plan_last_6m": downgraded_plan_last_6m,
        "price_increase_recently": price_increase_recently,
        "referred_by_friend": referred_by_friend,
        "churn": churn,
    })
    return df


if __name__ == "__main__":
    df = generate_ott_dataset()
    df.to_csv("sample_ott_users.csv", index=False)
    print(df["churn"].value_counts(normalize=True))
