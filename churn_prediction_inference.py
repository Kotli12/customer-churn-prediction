"""
Churn Prediction - Inference Script
====================================
Use trained model to predict churn for new customers
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ============================================================================
# LOAD TRAINED ARTIFACTS
# ============================================================================

def load_model_artifacts():
    """Load the trained model, scaler, and feature names"""
    with open('/home/claude/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('/home/claude/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('/home/claude/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, scaler, feature_names


# ============================================================================
# FEATURE ENGINEERING (same as training)
# ============================================================================

def engineer_features_for_prediction(df):
    """Apply the same feature engineering as during training"""
    df_engineered = df.copy()
    
    # 1. Engagement Score
    df_engineered['engagement_score'] = pd.cut(
        df_engineered['avg_weekly_usage_hours'],
        bins=[0, 5, 10, 15, 100],
        labels=[1, 2, 3, 4]
    ).astype(int)
    
    # 2. Customer Health Score
    df_engineered['health_score'] = (
        10 - df_engineered['payment_failures'] - 
        (df_engineered['support_tickets'] / 2)
    )
    df_engineered['health_score'] = df_engineered['health_score'].clip(lower=0)
    
    # 3. Activity Recency Category
    df_engineered['activity_recency'] = pd.cut(
        df_engineered['last_login_days_ago'],
        bins=[0, 7, 14, 30, 100],
        labels=['Very Recent', 'Recent', 'Moderate', 'Inactive']
    )
    
    # 4. Tenure Category
    df_engineered['tenure_category'] = pd.cut(
        df_engineered['tenure_months'],
        bins=[0, 6, 12, 24, 100],
        labels=['New', 'Regular', 'Established', 'Loyal']
    )
    
    # 5. Usage Intensity
    df_engineered['usage_intensity'] = pd.cut(
        df_engineered['avg_weekly_usage_hours'],
        bins=[0, 8, 16, 100],
        labels=['Low', 'Medium', 'High']
    )
    
    # 6. Risk Flags
    df_engineered['high_payment_risk'] = (df_engineered['payment_failures'] >= 3).astype(int)
    df_engineered['high_support_risk'] = (df_engineered['support_tickets'] >= 5).astype(int)
    df_engineered['inactive_risk'] = (df_engineered['last_login_days_ago'] >= 30).astype(int)
    
    # 7. Date Features
    df_engineered['signup_date'] = pd.to_datetime(df_engineered['signup_date'])
    df_engineered['signup_month'] = df_engineered['signup_date'].dt.month
    df_engineered['signup_quarter'] = df_engineered['signup_date'].dt.quarter
    df_engineered['signup_year'] = df_engineered['signup_date'].dt.year
    
    # 8. Value Segment
    df_engineered['value_segment'] = df_engineered.apply(
        lambda row: f"{row['plan_type']}_{row['usage_intensity']}", axis=1
    )
    
    # 9. Composite Risk Score
    df_engineered['risk_score'] = (
        df_engineered['payment_failures'] * 2 +
        df_engineered['support_tickets'] +
        (df_engineered['last_login_days_ago'] / 10) -
        (df_engineered['avg_weekly_usage_hours'] / 5)
    )
    
    # 10. Usage to Fee Ratio
    df_engineered['usage_to_fee_ratio'] = (
        df_engineered['avg_weekly_usage_hours'] / 
        (df_engineered['monthly_fee'] / 100)
    )
    
    return df_engineered


def preprocess_for_prediction(df, feature_names):
    """Preprocess data to match training format"""
    # Drop columns not used in training
    features_to_drop = ['user_id', 'signup_date', 'churn']
    
    # Encode categorical features
    categorical_features = ['plan_type', 'activity_recency', 'tenure_category', 
                           'usage_intensity', 'value_segment']
    
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Drop unused columns
    for col in features_to_drop:
        if col in df_encoded.columns:
            df_encoded = df_encoded.drop(col, axis=1)
    
    # Ensure all training features are present
    for feature in feature_names:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Select only the features used in training, in the same order
    X = df_encoded[feature_names]
    
    return X


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_churn(customer_data):
    """
    Predict churn for a single customer or batch of customers
    
    Parameters:
    -----------
    customer_data : dict or DataFrame
        Customer information with required features
    
    Returns:
    --------
    DataFrame with predictions and probabilities
    """
    # Load model artifacts
    model, scaler, feature_names = load_model_artifacts()
    
    # Convert to DataFrame if dict
    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    else:
        df = customer_data.copy()
    
    # Store user_id for output
    user_ids = df['user_id'].values if 'user_id' in df.columns else range(len(df))
    
    # Engineer features
    df_engineered = engineer_features_for_prediction(df)
    
    # Preprocess
    X = preprocess_for_prediction(df_engineered, feature_names)
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Create result DataFrame
    results = pd.DataFrame({
        'user_id': user_ids,
        'churn_prediction': ['Yes' if p == 1 else 'No' for p in predictions],
        'churn_probability': probabilities,
        'risk_level': pd.cut(probabilities, 
                            bins=[0, 0.3, 0.6, 1.0],
                            labels=['Low', 'Medium', 'High'])
    })
    
    return results


def get_actionable_insights(results, customer_data):
    """Generate actionable recommendations based on predictions"""
    insights = []
    
    for idx, row in results.iterrows():
        customer = customer_data.iloc[idx] if isinstance(customer_data, pd.DataFrame) else customer_data
        
        recommendation = {
            'user_id': row['user_id'],
            'risk_level': row['risk_level'],
            'churn_probability': f"{row['churn_probability']:.2%}",
            'actions': []
        }
        
        # Generate recommendations based on customer profile
        if row['churn_probability'] > 0.6:
            recommendation['priority'] = 'HIGH'
            
            if customer.get('last_login_days_ago', 0) > 30:
                recommendation['actions'].append("URGENT: Customer inactive >30 days - Send re-engagement campaign")
            
            if customer.get('payment_failures', 0) >= 3:
                recommendation['actions'].append("Address payment issues - Offer payment plan")
            
            if customer.get('support_tickets', 0) >= 5:
                recommendation['actions'].append("High support needs - Assign dedicated support")
            
            if customer.get('avg_weekly_usage_hours', 0) < 5:
                recommendation['actions'].append("Low engagement - Offer onboarding assistance")
                
        elif row['churn_probability'] > 0.3:
            recommendation['priority'] = 'MEDIUM'
            recommendation['actions'].append("Monitor closely - Send satisfaction survey")
            
            if customer.get('last_login_days_ago', 0) > 14:
                recommendation['actions'].append("Send personalized content/offers")
        
        else:
            recommendation['priority'] = 'LOW'
            recommendation['actions'].append("Continue standard engagement")
        
        insights.append(recommendation)
    
    return insights


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CHURN PREDICTION - INFERENCE DEMO")
    print("="*70)
    
    # Example 1: High-risk customer
    print("\n" + "="*70)
    print("EXAMPLE 1: High-Risk Customer")
    print("="*70)
    
    high_risk_customer = {
        'user_id': 9999,
        'signup_date': '2024-01-15',
        'plan_type': 'Premium',
        'monthly_fee': 699,
        'avg_weekly_usage_hours': 3.5,
        'support_tickets': 7,
        'payment_failures': 4,
        'tenure_months': 8,
        'last_login_days_ago': 45,
        'churn': 'Unknown'  # This is what we're predicting
    }
    
    result = predict_churn(high_risk_customer)
    print("\nPrediction Results:")
    print(result.to_string(index=False))
    
    insights = get_actionable_insights(result, high_risk_customer)
    print("\nActionable Insights:")
    for insight in insights:
        print(f"\nUser ID: {insight['user_id']}")
        print(f"Priority: {insight['priority']}")
        print(f"Churn Probability: {insight['churn_probability']}")
        print("Recommended Actions:")
        for action in insight['actions']:
            print(f"  • {action}")
    
    # Example 2: Low-risk customer
    print("\n" + "="*70)
    print("EXAMPLE 2: Low-Risk Customer")
    print("="*70)
    
    low_risk_customer = {
        'user_id': 10000,
        'signup_date': '2023-05-20',
        'plan_type': 'Standard',
        'monthly_fee': 399,
        'avg_weekly_usage_hours': 18.5,
        'support_tickets': 2,
        'payment_failures': 0,
        'tenure_months': 20,
        'last_login_days_ago': 2,
        'churn': 'Unknown'
    }
    
    result = predict_churn(low_risk_customer)
    print("\nPrediction Results:")
    print(result.to_string(index=False))
    
    insights = get_actionable_insights(result, low_risk_customer)
    print("\nActionable Insights:")
    for insight in insights:
        print(f"\nUser ID: {insight['user_id']}")
        print(f"Priority: {insight['priority']}")
        print(f"Churn Probability: {insight['churn_probability']}")
        print("Recommended Actions:")
        for action in insight['actions']:
            print(f"  • {action}")
    
    # Example 3: Batch prediction
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Prediction on Test Set")
    print("="*70)
    
    # Load original dataset
    df_original = pd.read_csv('/mnt/user-data/uploads/customer_subscription_churn_usage_patterns.csv')
    
    # Take a sample
    sample_customers = df_original.sample(5, random_state=42)
    
    # Predict
    batch_results = predict_churn(sample_customers)
    
    print("\nBatch Prediction Results:")
    print(batch_results.to_string(index=False))
    
    # Show distribution
    print("\n" + "="*70)
    print("Risk Distribution:")
    print("="*70)
    risk_dist = batch_results['risk_level'].value_counts()
    for risk, count in risk_dist.items():
        print(f"{risk} Risk: {count} customers ({count/len(batch_results)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("INFERENCE DEMO COMPLETE!")
    print("="*70)
