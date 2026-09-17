"""
Churn prediction model: trains a RandomForest classifier on OTT usage data
and exposes helpers for scoring users + explaining which factors drove
each individual prediction (per-user feature attribution).
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [
    "age", "plan_type_enc", "monthly_fee", "subscription_tenure_months",
    "num_devices", "uses_multiple_profiles", "avg_watch_hours_per_week",
    "days_since_last_login", "num_genres_watched", "content_completion_rate",
    "watch_hours_trend_pct", "customer_support_tickets",
    "payment_failures_last_6m", "downgraded_plan_last_6m",
    "price_increase_recently", "referred_by_friend",
]

HUMAN_LABELS = {
    "days_since_last_login": "Inactivity (days since last login)",
    "avg_watch_hours_per_week": "Low weekly watch time",
    "content_completion_rate": "Low content completion rate",
    "customer_support_tickets": "Support ticket friction",
    "payment_failures_last_6m": "Recent payment failures",
    "downgraded_plan_last_6m": "Recent plan downgrade",
    "price_increase_recently": "Recent price increase",
    "watch_hours_trend_pct": "Declining watch-time trend",
    "referred_by_friend": "Not a friend-referred user",
    "uses_multiple_profiles": "Single-profile usage",
    "subscription_tenure_months": "Short subscription tenure",
    "num_genres_watched": "Narrow content exploration",
}


@dataclass
class TrainedModel:
    model: RandomForestClassifier
    plan_encoder: LabelEncoder
    feature_importances: pd.Series
    auc: float
    accuracy: float


def _prep_features(df: pd.DataFrame, plan_encoder: LabelEncoder = None) -> Tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy()
    if plan_encoder is None:
        plan_encoder = LabelEncoder()
        df["plan_type_enc"] = plan_encoder.fit_transform(df["plan_type"])
    else:
        df["plan_type_enc"] = plan_encoder.transform(df["plan_type"])
    return df, plan_encoder


def train_churn_model(df: pd.DataFrame) -> TrainedModel:
    df_enc, plan_encoder = _prep_features(df)
    X = df_enc[FEATURE_COLUMNS]
    y = df_enc["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=250, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    acc = accuracy_score(y_test, (proba_test >= 0.5).astype(int))

    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)

    return TrainedModel(model=clf, plan_encoder=plan_encoder, feature_importances=importances, auc=auc, accuracy=acc)


def score_users(trained: TrainedModel, df: pd.DataFrame) -> pd.DataFrame:
    """Return the input df with churn_probability and risk_segment columns appended."""
    df_enc, _ = _prep_features(df, trained.plan_encoder)
    X = df_enc[FEATURE_COLUMNS]
    proba = trained.model.predict_proba(X)[:, 1]

    out = df.copy()
    out["churn_probability"] = proba.round(4)
    out["risk_segment"] = pd.cut(
        out["churn_probability"], bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"]
    )
    return out


# Reference values considered "healthy" so we can flag a user's factor as risky
# relative to a good baseline, direction-aware per feature.
RISK_DIRECTION = {
    "days_since_last_login": ("higher_is_worse", 5),
    "avg_watch_hours_per_week": ("lower_is_worse", 6),
    "content_completion_rate": ("lower_is_worse", 0.55),
    "customer_support_tickets": ("higher_is_worse", 0.6),
    "payment_failures_last_6m": ("higher_is_worse", 0.25),
    "downgraded_plan_last_6m": ("higher_is_worse", 0),
    "price_increase_recently": ("higher_is_worse", 0),
    "watch_hours_trend_pct": ("lower_is_worse", -5),
    "referred_by_friend": ("lower_is_worse", 0.5),
    "uses_multiple_profiles": ("lower_is_worse", 0.5),
    "subscription_tenure_months": ("lower_is_worse", 30),
    "num_genres_watched": ("lower_is_worse", 5),
}


def top_risk_factors_for_user(trained: TrainedModel, user_row: pd.Series, top_n: int = 3) -> List[str]:
    """Combine global feature importance with this user's deviation from baseline
    to produce a ranked, human-readable list of the drivers behind their risk score."""
    scored = []
    for feat, importance in trained.feature_importances.items():
        base_feat = feat.replace("_enc", "")
        if base_feat not in RISK_DIRECTION:
            continue
        direction, baseline = RISK_DIRECTION[base_feat]
        val = user_row.get(base_feat)
        if val is None:
            continue
        if direction == "higher_is_worse":
            deviation = max(0.0, (val - baseline))
        else:
            deviation = max(0.0, (baseline - val))
        # normalize deviation roughly by baseline scale to keep comparable
        norm = deviation / (abs(baseline) + 1e-6) if baseline not in (0,) else deviation
        score = importance * (norm + 0.15)  # small floor so importance still counts
        scored.append((base_feat, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [HUMAN_LABELS.get(f, f) for f, s in scored[:top_n] if s > 0]
    if not top:
        top = [HUMAN_LABELS.get(scored[0][0], scored[0][0])] if scored else ["General engagement decline"]
    return top
