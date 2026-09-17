"""
Personalized retention recommendation engine.
Maps a user's specific risk drivers to concrete, actionable retention offers
rather than a single generic "give a discount" response.
"""
from typing import List, Dict
import pandas as pd


def recommend_actions(user: pd.Series, risk_segment: str) -> List[Dict[str, str]]:
    actions = []

    if risk_segment == "Low":
        actions.append({
            "action": "No intervention needed",
            "why": "User shows healthy engagement — keep sending normal personalized recommendations.",
            "channel": "In-app",
        })
        return actions

    if user.get("days_since_last_login", 0) >= 14:
        actions.append({
            "action": "Re-engagement push: 'We picked something for you' with 3 curated titles",
            "why": f"Inactive for {int(user['days_since_last_login'])} days.",
            "channel": "Push notification + Email",
        })

    if user.get("watch_hours_trend_pct", 0) <= -15:
        actions.append({
            "action": "Send a personalized 'trending in your favorite genres' watchlist",
            "why": f"Watch time dropped {abs(user['watch_hours_trend_pct']):.0f}% vs last month.",
            "channel": "In-app + Email",
        })

    if user.get("content_completion_rate", 1) < 0.35:
        actions.append({
            "action": "Shift recommendations toward shorter-form / episodic content",
            "why": "Low completion rate suggests content mismatch with attention span or interest.",
            "channel": "Recommendation engine tuning",
        })

    if user.get("payment_failures_last_6m", 0) > 0:
        actions.append({
            "action": "Offer 7-day grace period + one-click payment method update",
            "why": "Recent payment failures often precede involuntary churn.",
            "channel": "Email + SMS",
        })

    if user.get("customer_support_tickets", 0) >= 2:
        actions.append({
            "action": "Proactive outreach from support team to resolve open friction",
            "why": f"{int(user['customer_support_tickets'])} support tickets indicate unresolved frustration.",
            "channel": "Email / Call",
        })

    if user.get("downgraded_plan_last_6m", 0) == 1 or (
        user.get("plan_type") == "Premium" and user.get("avg_watch_hours_per_week", 0) < 3
    ):
        actions.append({
            "action": "Offer a 20% loyalty discount for 3 months, or suggest a better-fit lower plan",
            "why": "Signs of price sensitivity or bill-shock relative to actual usage.",
            "channel": "Email + In-app banner",
        })

    if user.get("price_increase_recently", 0) == 1 and user.get("risk_segment", risk_segment) == "High":
        actions.append({
            "action": "Targeted 'we value you' retention discount before renewal date",
            "why": "Recent price increase combined with high churn risk.",
            "channel": "Email",
        })

    if user.get("num_genres_watched", 10) <= 2:
        actions.append({
            "action": "Trigger a genre-discovery onboarding flow to broaden content exposure",
            "why": "Narrow content exploration increases risk of running out of relevant content.",
            "channel": "In-app",
        })

    if user.get("uses_multiple_profiles", 1) == 0 and user.get("num_devices", 1) <= 1:
        actions.append({
            "action": "Prompt to add a family profile / second device (increases stickiness)",
            "why": "Single-profile, single-device accounts churn more easily.",
            "channel": "In-app",
        })

    if not actions:
        actions.append({
            "action": "Send a general win-back offer with a limited-time discount",
            "why": "Elevated risk score without one dominant driver — use a broad incentive.",
            "channel": "Email",
        })

    return actions[:5]
