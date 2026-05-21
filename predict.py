"""
Customer Churn Prediction — Inference & Customer Risk Report
============================================================
Scores new customers and translates predictions into business actions.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from features import prepare_features

ROOT   = Path(__file__).parent
MODELS = ROOT / 'models'


# ── load artifacts ────────────────────────────────────────────────────────────

def load_artifacts():
    with open(MODELS / 'model.pkl',        'rb') as f: model         = pickle.load(f)
    with open(MODELS / 'scaler.pkl',       'rb') as f: scaler        = pickle.load(f)
    with open(MODELS / 'feature_names.pkl','rb') as f: feature_names = pickle.load(f)
    return model, scaler, feature_names


# ── core prediction ───────────────────────────────────────────────────────────

def predict(customer_data) -> pd.DataFrame:
    """
    Predict churn probability for one customer (dict) or many (DataFrame).

    Returns
    -------
    DataFrame with: user_id, churn_prediction, churn_probability,
                    risk_level, annual_value, revenue_at_risk
    """
    model, scaler, feature_names = load_artifacts()

    df = pd.DataFrame([customer_data]) if isinstance(customer_data, dict) else customer_data.copy()

    user_ids   = df['user_id'].values     if 'user_id'     in df.columns else np.arange(len(df))
    annual_val = df['monthly_fee'].values * 12 if 'monthly_fee' in df.columns else np.zeros(len(df))

    X       = prepare_features(df)[feature_names]
    X_s     = scaler.transform(X)
    preds   = model.predict(X_s)
    probs   = model.predict_proba(X_s)[:, 1]
    risk    = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])

    return pd.DataFrame({
        'user_id'          : user_ids,
        'churn_prediction' : ['Yes' if p == 1 else 'No' for p in preds],
        'churn_probability': probs.round(4),
        'risk_level'       : risk,
        'annual_value'     : annual_val,
        'revenue_at_risk'  : (probs * annual_val).round(2),
    })


# ── intervention logic ────────────────────────────────────────────────────────

def get_intervention(result_row: dict, customer: dict) -> dict:
    """
    Translate a churn probability into a specific, costed intervention.

    Business logic:
    - High risk  (>60%): Personal outreach + retention offer. High cost, high stakes.
    - Medium risk (30–60%): Automated nurture sequence + one check-in call.
    - Low risk    (<30%): Standard engagement. No extra spend.
    """
    prob    = result_row['churn_probability']
    risk    = result_row['risk_level']
    rev     = result_row['revenue_at_risk']

    if prob > 0.6:
        priority    = 'HIGH'
        action_type = 'Personal outreach + retention offer'
        cost        = 300   # account manager time + potential discount
        reasons     = []
        if customer.get('last_login_days_ago', 0) >= 30:
            reasons.append('Inactive 30+ days — send re-engagement campaign immediately')
        if customer.get('payment_failures', 0) >= 3:
            reasons.append('3+ payment failures — offer payment plan or billing support')
        if customer.get('support_tickets', 0) >= 5:
            reasons.append('5+ support tickets — assign dedicated support contact')
        if customer.get('avg_weekly_usage_hours', 0) < 5:
            reasons.append('Very low usage — schedule onboarding / value discovery call')
        if not reasons:
            reasons.append('Multiple moderate risk factors — proactive check-in call')

    elif prob > 0.3:
        priority    = 'MEDIUM'
        action_type = 'Automated nurture + check-in call'
        cost        = 50
        reasons     = ['Send personalised email: tips, new features, or case studies']
        if customer.get('last_login_days_ago', 0) >= 14:
            reasons.append('14+ days inactive — trigger re-engagement email sequence')

    else:
        priority    = 'LOW'
        action_type = 'Standard engagement'
        cost        = 0
        reasons     = ['Continue standard communications — no extra action needed']

    return {
        'priority'   : priority,
        'action_type': action_type,
        'est_cost'   : cost,
        'reasons'    : reasons,
        'revenue_at_risk': rev,
        'net_value_if_saved': rev - cost,
    }


# ── print a single customer report ───────────────────────────────────────────

def print_customer_report(result_row: dict, customer: dict, action: dict) -> None:
    uid  = result_row['user_id']
    prob = result_row['churn_probability']
    risk = result_row['risk_level']
    rev  = result_row['revenue_at_risk']

    risk_colors = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
    icon = risk_colors.get(str(risk), '⚪')

    print(f"  {icon}  User {uid}  |  Plan: {customer.get('plan_type','?')}  "
          f"|  Churn probability: {prob:.1%}  |  Risk: {risk}")
    print(f"     Annual value : ${customer.get('monthly_fee',0)*12:,.0f}  "
          f"|  Revenue at risk: ${rev:,.0f}")
    print(f"     Action       : [{action['priority']}] {action['action_type']}  "
          f"(est. cost: ${action['est_cost']})")
    for r in action['reasons']:
        print(f"       → {r}")
    print()


# ── batch report ─────────────────────────────────────────────────────────────

def batch_report(df: pd.DataFrame) -> None:
    """Score a full customer base and print a portfolio risk summary."""
    results = predict(df)

    print("\n" + "═" * 60)
    print("  CUSTOMER RISK PORTFOLIO SUMMARY")
    print("═" * 60)

    total_rev_at_risk = results['revenue_at_risk'].sum()
    print(f"  Customers scored : {len(results):,}")
    print(f"  Total annual revenue at risk : ${total_rev_at_risk:,.0f}")
    print()

    for level in ['High', 'Medium', 'Low']:
        seg  = results[results['risk_level'] == level]
        rev  = seg['revenue_at_risk'].sum()
        pct  = len(seg) / len(results) * 100
        print(f"  {level:6} risk : {len(seg):4} customers ({pct:.1f}%)  "
              f"|  Revenue at risk: ${rev:,.0f}")

    print()
    print("  TOP 10 CUSTOMERS BY REVENUE AT RISK")
    print("  " + "─" * 56)
    top10 = results.nlargest(10, 'revenue_at_risk')[
        ['user_id', 'churn_probability', 'risk_level', 'annual_value', 'revenue_at_risk']
    ]
    print(top10.to_string(index=False))
    print("═" * 60)


# ── demo ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "═" * 60)
    print("  CUSTOMER CHURN PREDICTION — INFERENCE DEMO")
    print("═" * 60)

    examples = [
        {
            'user_id': 9001, 'signup_date': '2024-01-15', 'plan_type': 'Premium',
            'monthly_fee': 699, 'avg_weekly_usage_hours': 3.5, 'support_tickets': 7,
            'payment_failures': 4, 'tenure_months': 8, 'last_login_days_ago': 45,
        },
        {
            'user_id': 9002, 'signup_date': '2023-05-20', 'plan_type': 'Standard',
            'monthly_fee': 399, 'avg_weekly_usage_hours': 18.5, 'support_tickets': 1,
            'payment_failures': 0, 'tenure_months': 24, 'last_login_days_ago': 2,
        },
        {
            'user_id': 9003, 'signup_date': '2024-06-01', 'plan_type': 'Basic',
            'monthly_fee': 199, 'avg_weekly_usage_hours': 7.0, 'support_tickets': 3,
            'payment_failures': 1, 'tenure_months': 5, 'last_login_days_ago': 18,
        },
    ]

    print("\n  INDIVIDUAL CUSTOMER ASSESSMENTS\n")
    for ex in examples:
        result  = predict(ex)
        row     = result.iloc[0].to_dict()
        action  = get_intervention(row, ex)
        print_customer_report(row, ex, action)

    # Batch report on the full dataset
    df_all = pd.read_csv(ROOT / 'customer_subscription_churn_usage_patterns.csv')
    batch_report(df_all)
