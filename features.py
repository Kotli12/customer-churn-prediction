"""Shared feature engineering — single source of truth for train and predict."""

import numpy as np
import pandas as pd

PLAN_ORDER = {'Basic': 0, 'Standard': 1, 'Premium': 2}

FEATURE_COLS = [
    # Raw
    'avg_weekly_usage_hours', 'support_tickets', 'payment_failures',
    'tenure_months', 'last_login_days_ago', 'monthly_fee',
    # Encoded plan
    'plan_ordinal',
    # Date
    'signup_month', 'signup_quarter', 'signup_year',
    # Ordinal buckets
    'tenure_bucket', 'recency_bucket',
    # Binary flags
    'high_payment_risk', 'high_support_risk', 'inactive_flag', 'low_usage_flag',
    'total_risk_flags',
    # Composite scores
    'risk_score', 'health_score',
    # Ratios
    'usage_per_dollar', 'support_per_month', 'failure_per_month',
    # Log transforms
    'log_last_login', 'log_support_tickets', 'log_payment_failures', 'log_tenure',
    # Interactions
    'fee_x_failures', 'usage_x_tenure', 'login_x_tickets',
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out['plan_ordinal'] = out['plan_type'].map(PLAN_ORDER).fillna(0).astype(int)

    out['signup_date'] = pd.to_datetime(out['signup_date'])
    out['signup_month'] = out['signup_date'].dt.month
    out['signup_quarter'] = out['signup_date'].dt.quarter
    out['signup_year'] = out['signup_date'].dt.year

    out['tenure_bucket'] = pd.cut(
        out['tenure_months'], bins=[0, 6, 12, 24, 200], labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    out['recency_bucket'] = pd.cut(
        out['last_login_days_ago'], bins=[0, 7, 14, 30, 200], labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    out['high_payment_risk'] = (out['payment_failures'] >= 3).astype(int)
    out['high_support_risk'] = (out['support_tickets'] >= 5).astype(int)
    out['inactive_flag'] = (out['last_login_days_ago'] >= 30).astype(int)
    out['low_usage_flag'] = (out['avg_weekly_usage_hours'] < 5).astype(int)
    out['total_risk_flags'] = (
        out['high_payment_risk'] + out['high_support_risk'] +
        out['inactive_flag'] + out['low_usage_flag']
    )

    out['risk_score'] = (
        out['payment_failures'] * 2.0 +
        out['support_tickets'] * 1.0 +
        out['last_login_days_ago'] / 10.0 -
        out['avg_weekly_usage_hours'] / 5.0
    )
    out['health_score'] = (
        out['avg_weekly_usage_hours'] * 0.5 +
        out['tenure_months'] * 0.1 -
        out['payment_failures'] * 2.0 -
        out['support_tickets'] * 0.5
    )

    out['usage_per_dollar'] = out['avg_weekly_usage_hours'] / (out['monthly_fee'] / 100.0)
    out['support_per_month'] = (
        out['support_tickets'] / out['tenure_months'].clip(lower=1)
    ).clip(upper=5)
    out['failure_per_month'] = (
        out['payment_failures'] / out['tenure_months'].clip(lower=1)
    ).clip(upper=3)

    out['log_last_login'] = np.log1p(out['last_login_days_ago'])
    out['log_support_tickets'] = np.log1p(out['support_tickets'])
    out['log_payment_failures'] = np.log1p(out['payment_failures'])
    out['log_tenure'] = np.log1p(out['tenure_months'])

    out['fee_x_failures'] = out['monthly_fee'] * out['payment_failures']
    out['usage_x_tenure'] = out['avg_weekly_usage_hours'] * out['tenure_months']
    out['login_x_tickets'] = out['last_login_days_ago'] * out['support_tickets']

    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the model input columns, fully engineered."""
    return engineer_features(df)[FEATURE_COLS]
