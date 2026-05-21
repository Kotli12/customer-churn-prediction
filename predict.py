"""Customer Churn Prediction — Inference"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from features import prepare_features

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / 'models'


def load_artifacts():
    with open(MODELS_DIR / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names


def predict(customer_data) -> pd.DataFrame:
    """
    Predict churn for one or more customers.

    Parameters
    ----------
    customer_data : dict or pd.DataFrame
        Required fields: user_id, signup_date, plan_type, monthly_fee,
        avg_weekly_usage_hours, support_tickets, payment_failures,
        tenure_months, last_login_days_ago

    Returns
    -------
    pd.DataFrame with: user_id, churn_prediction, churn_probability, risk_level
    """
    model, scaler, feature_names = load_artifacts()

    df = pd.DataFrame([customer_data]) if isinstance(customer_data, dict) else customer_data.copy()
    user_ids = df['user_id'].values if 'user_id' in df.columns else np.arange(len(df))

    X = prepare_features(df)[feature_names]
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]
    risk = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])

    return pd.DataFrame({
        'user_id': user_ids,
        'churn_prediction': ['Yes' if p == 1 else 'No' for p in preds],
        'churn_probability': probs.round(4),
        'risk_level': risk,
    })


def recommend_actions(result_row: dict, customer: dict) -> list[str]:
    """Return action strings for a single prediction result."""
    prob = result_row['churn_probability']
    actions = []

    if prob > 0.6:
        if customer.get('last_login_days_ago', 0) > 30:
            actions.append('URGENT: 30+ day inactivity — launch re-engagement campaign')
        if customer.get('payment_failures', 0) >= 3:
            actions.append('Offer payment plan / retry assistance')
        if customer.get('support_tickets', 0) >= 5:
            actions.append('Assign dedicated support contact')
        if customer.get('avg_weekly_usage_hours', 0) < 5:
            actions.append('Low engagement — offer onboarding check-in')
        if not actions:
            actions.append('High risk — schedule proactive outreach call')
    elif prob > 0.3:
        actions.append('Send personalized check-in / satisfaction survey')
        if customer.get('last_login_days_ago', 0) > 14:
            actions.append('Share usage tips or new feature highlights')
    else:
        actions.append('Continue standard engagement')

    return actions


if __name__ == '__main__':
    print("=== Customer Churn — Inference Demo ===\n")

    examples = [
        {
            'user_id': 9001,
            'signup_date': '2024-01-15',
            'plan_type': 'Premium',
            'monthly_fee': 699,
            'avg_weekly_usage_hours': 3.5,
            'support_tickets': 7,
            'payment_failures': 4,
            'tenure_months': 8,
            'last_login_days_ago': 45,
        },
        {
            'user_id': 9002,
            'signup_date': '2023-05-20',
            'plan_type': 'Standard',
            'monthly_fee': 399,
            'avg_weekly_usage_hours': 18.5,
            'support_tickets': 1,
            'payment_failures': 0,
            'tenure_months': 24,
            'last_login_days_ago': 2,
        },
        {
            'user_id': 9003,
            'signup_date': '2024-06-01',
            'plan_type': 'Basic',
            'monthly_fee': 199,
            'avg_weekly_usage_hours': 7.0,
            'support_tickets': 3,
            'payment_failures': 1,
            'tenure_months': 5,
            'last_login_days_ago': 18,
        },
    ]

    for ex in examples:
        result = predict(ex)
        row = result.iloc[0].to_dict()
        print(f"User {ex['user_id']} ({ex['plan_type']})")
        print(f"  Prediction : {row['churn_prediction']}  |  Probability: {row['churn_probability']:.1%}  |  Risk: {row['risk_level']}")
        actions = recommend_actions(row, ex)
        for action in actions:
            print(f"  > {action}")
        print()

    # Batch prediction on sample of training data
    print("--- Batch prediction (10 random customers) ---")
    df_all = pd.read_csv(ROOT / 'customer_subscription_churn_usage_patterns.csv')
    sample = df_all.sample(10, random_state=7)
    batch = predict(sample)
    print(batch.to_string(index=False))
    print(f"\nRisk distribution:\n{batch['risk_level'].value_counts().to_string()}")
