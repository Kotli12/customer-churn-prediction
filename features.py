"""
Feature engineering — every feature maps to a known business driver of churn.

Business rationale for each group:

PAYMENT SIGNALS
  payment_failures, failure_per_month, high_payment_risk
  → Billing friction is the #1 early warning sign. A customer who can't pay
    smoothly is already halfway out the door.

ENGAGEMENT SIGNALS
  avg_weekly_usage_hours, low_usage_flag, usage_per_dollar, usage_x_tenure
  → Low usage = low perceived value. If a customer isn't using the product,
    they have no reason to keep paying for it.

INACTIVITY SIGNALS
  last_login_days_ago, inactive_flag, recency_bucket, log_last_login
  → Disengagement precedes cancellation by 2–4 weeks on average.
    A customer who stops logging in is quietly deciding to leave.

SUPPORT SIGNALS
  support_tickets, high_support_risk, support_per_month, login_x_tickets
  → Unresolved problems erode trust. High ticket volume with continued
    inactivity is a strong double signal.

LOYALTY SIGNALS
  tenure_months, tenure_bucket, log_tenure
  → New customers (0–6 months) haven't formed habits yet — they churn fastest.
    Long-tenured customers have proven loyalty but still need monitoring.

COMPOSITE RISK SCORE
  risk_score, health_score, total_risk_flags
  → These combine individual signals into a single business-readable number.
"""

import numpy as np
import pandas as pd

PLAN_ORDER = {'Basic': 0, 'Standard': 1, 'Premium': 2}

FEATURE_COLS = [
    # Raw signals
    'avg_weekly_usage_hours', 'support_tickets', 'payment_failures',
    'tenure_months', 'last_login_days_ago', 'monthly_fee',
    # Plan
    'plan_ordinal',
    # Date context
    'signup_month', 'signup_quarter', 'signup_year',
    # Loyalty buckets
    'tenure_bucket', 'recency_bucket',
    # Binary risk flags (each = one business alarm)
    'high_payment_risk', 'high_support_risk', 'inactive_flag', 'low_usage_flag',
    'total_risk_flags',
    # Composite scores
    'risk_score', 'health_score',
    # Per-month ratios (normalise for tenure)
    'usage_per_dollar', 'support_per_month', 'failure_per_month',
    # Log transforms (reduce skew on heavy-tailed distributions)
    'log_last_login', 'log_support_tickets', 'log_payment_failures', 'log_tenure',
    # Interaction terms
    'fee_x_failures', 'usage_x_tenure', 'login_x_tickets',
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out['plan_ordinal'] = out['plan_type'].map(PLAN_ORDER).fillna(0).astype(int)

    out['signup_date'] = pd.to_datetime(out['signup_date'])
    out['signup_month'] = out['signup_date'].dt.month
    out['signup_quarter'] = out['signup_date'].dt.quarter
    out['signup_year'] = out['signup_date'].dt.year

    # Loyalty stage: how long has the customer been with us?
    out['tenure_bucket'] = pd.cut(
        out['tenure_months'], bins=[0, 6, 12, 24, 200],
        labels=[0, 1, 2, 3], include_lowest=True
    ).astype(int)

    # Recency stage: how long since they last showed up?
    out['recency_bucket'] = pd.cut(
        out['last_login_days_ago'], bins=[0, 7, 14, 30, 200],
        labels=[0, 1, 2, 3], include_lowest=True
    ).astype(int)

    # Business alarm flags — each one is a threshold a manager would recognise
    out['high_payment_risk'] = (out['payment_failures'] >= 3).astype(int)
    out['high_support_risk'] = (out['support_tickets'] >= 5).astype(int)
    out['inactive_flag']     = (out['last_login_days_ago'] >= 30).astype(int)
    out['low_usage_flag']    = (out['avg_weekly_usage_hours'] < 5).astype(int)
    out['total_risk_flags']  = (
        out['high_payment_risk'] + out['high_support_risk'] +
        out['inactive_flag'] + out['low_usage_flag']
    )

    # Risk score: weighted combination of negative signals minus positive ones
    out['risk_score'] = (
        out['payment_failures'] * 2.0 +
        out['support_tickets'] * 1.0 +
        out['last_login_days_ago'] / 10.0 -
        out['avg_weekly_usage_hours'] / 5.0
    )

    # Health score: inverse of risk — higher = safer customer
    out['health_score'] = (
        out['avg_weekly_usage_hours'] * 0.5 +
        out['tenure_months'] * 0.1 -
        out['payment_failures'] * 2.0 -
        out['support_tickets'] * 0.5
    )

    # Ratios: normalise by tenure so a new customer isn't penalised
    out['usage_per_dollar']   = out['avg_weekly_usage_hours'] / (out['monthly_fee'] / 100.0)
    out['support_per_month']  = (out['support_tickets']    / out['tenure_months'].clip(lower=1)).clip(upper=5)
    out['failure_per_month']  = (out['payment_failures']   / out['tenure_months'].clip(lower=1)).clip(upper=3)

    # Log transforms: compress the long tail (e.g. 0 days vs 60 days login gap)
    out['log_last_login']       = np.log1p(out['last_login_days_ago'])
    out['log_support_tickets']  = np.log1p(out['support_tickets'])
    out['log_payment_failures'] = np.log1p(out['payment_failures'])
    out['log_tenure']           = np.log1p(out['tenure_months'])

    # Interaction terms: some effects multiply (e.g. inactive AND many tickets)
    out['fee_x_failures']  = out['monthly_fee'] * out['payment_failures']
    out['usage_x_tenure']  = out['avg_weekly_usage_hours'] * out['tenure_months']
    out['login_x_tickets'] = out['last_login_days_ago'] * out['support_tickets']

    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and return only model input columns."""
    return engineer_features(df)[FEATURE_COLS]
