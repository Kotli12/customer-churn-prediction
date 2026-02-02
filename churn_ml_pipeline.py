"""
Customer Churn Prediction - Complete ML Pipeline
=================================================
Includes: Feature Engineering, Model Training, Evaluation, and Predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)

# ============================================================================
# 1. FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create new features from existing data"""
    print("="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    df_engineered = df.copy()
    
    # 1. Engagement Score (based on usage)
    df_engineered['engagement_score'] = pd.cut(
        df_engineered['avg_weekly_usage_hours'],
        bins=[0, 5, 10, 15, 100],
        labels=[1, 2, 3, 4]
    ).astype(int)
    
    # 2. Customer Health Score (inverse of problems)
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
    
    # 8. Value Segment (combination of plan and usage)
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
    
    print(f"\n✓ Created {len(df_engineered.columns) - len(df.columns)} new features")
    print(f"✓ Total features now: {len(df_engineered.columns)}")
    
    return df_engineered


# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """Prepare data for machine learning"""
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    df_processed = df.copy()
    
    # Encode target variable
    le_target = LabelEncoder()
    df_processed['churn_encoded'] = le_target.fit_transform(df_processed['churn'])
    
    # Select features for modeling
    # Drop original date, user_id, and original churn
    features_to_drop = ['user_id', 'signup_date', 'churn']
    
    # Encode categorical features
    categorical_features = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [col for col in categorical_features if col not in features_to_drop + ['churn_encoded']]
    
    print(f"\nCategorical features to encode: {categorical_features}")
    
    df_encoded = pd.get_dummies(df_processed, columns=categorical_features, drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop(features_to_drop + ['churn_encoded'], axis=1, errors='ignore')
    y = df_encoded['churn_encoded']
    
    print(f"\n✓ Feature matrix shape: {X.shape}")
    print(f"✓ Target variable shape: {y.shape}")
    print(f"✓ Class distribution: {np.bincount(y)}")
    
    return X, y, le_target


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split data and apply scaling"""
    print("\n" + "="*70)
    print("TRAIN-TEST SPLIT & SCALING")
    print("="*70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} ({(1-test_size)*100:.0f}%)")
    print(f"Test set size: {X_test.shape[0]} ({test_size*100:.0f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✓ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def handle_imbalance(X_train, y_train):
    """Apply SMOTE to handle class imbalance - DISABLED (network required)"""
    print("\n" + "="*70)
    print("HANDLING CLASS IMBALANCE")
    print("="*70)
    
    print(f"\nOriginal class distribution: {np.bincount(y_train)}")
    print("⚠ SMOTE disabled - network required for installation")
    print("Proceeding with original class distribution")
    
    return X_train, y_train


# ============================================================================
# 3. MODEL TRAINING
# ============================================================================

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple classification models"""
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        trained_models[name] = model
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC: {roc_auc:.4f}")
    
    return results, trained_models


# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================

def evaluate_models(results, y_test):
    """Compare and visualize model performance"""
    print("\n" + "="*70)
    print("MODEL EVALUATION & COMPARISON")
    print("="*70)
    
    # Create comparison dataframe
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'] if metrics['roc_auc'] else 0
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Find best model
    best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
    print(f"\n🏆 Best Model (by F1-Score): {best_model_name}")
    
    # Visualization - Model Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Metrics Comparison
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    comparison_df.set_index('Model')[metrics_to_plot].plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. ROC Curves
    for name, metrics in results.items():
        if metrics['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
            axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})")
    
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 1].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Confusion Matrix for Best Model
    best_metrics = results[best_model_name]
    cm = confusion_matrix(y_test, best_metrics['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Actual', fontsize=12)
    axes[1, 0].set_xlabel('Predicted', fontsize=12)
    
    # 4. Feature Importance (for tree-based models)
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        feature_importance = best_metrics['model'].feature_importances_
        # Get feature names (this is a simplified version)
        top_n = 15
        indices = np.argsort(feature_importance)[-top_n:]
        
        axes[1, 1].barh(range(top_n), feature_importance[indices], color='teal')
        axes[1, 1].set_yticks(range(top_n))
        axes[1, 1].set_yticklabels([f'Feature {i}' for i in indices])
        axes[1, 1].set_xlabel('Importance', fontsize=12)
        axes[1, 1].set_title(f'Top {top_n} Feature Importances - {best_model_name}', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('/home/claude/model_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: model_evaluation.png")
    plt.close()
    
    return best_model_name, comparison_df


def detailed_classification_report(results, y_test, best_model_name):
    """Print detailed classification report for best model"""
    print("\n" + "="*70)
    print(f"DETAILED REPORT - {best_model_name}")
    print("="*70)
    
    best_metrics = results[best_model_name]
    print("\nClassification Report:")
    print(classification_report(y_test, best_metrics['y_pred'], 
                                target_names=['No Churn', 'Churn']))
    
    # Confusion Matrix Analysis
    cm = confusion_matrix(y_test, best_metrics['y_pred'])
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix Analysis:")
    print(f"  True Negatives (Correctly predicted No Churn): {tn}")
    print(f"  False Positives (Incorrectly predicted Churn): {fp}")
    print(f"  False Negatives (Missed Churn): {fn}")
    print(f"  True Positives (Correctly predicted Churn): {tp}")
    
    print(f"\n  Churn Detection Rate: {tp/(tp+fn)*100:.2f}%")
    print(f"  False Alarm Rate: {fp/(fp+tn)*100:.2f}%")


# ============================================================================
# 5. HYPERPARAMETER TUNING
# ============================================================================

def tune_best_model(X_train, y_train, X_test, y_test, model_name):
    """Perform hyperparameter tuning on the best model"""
    print("\n" + "="*70)
    print(f"HYPERPARAMETER TUNING - {model_name}")
    print("="*70)
    
    if model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        model = RandomForestClassifier(random_state=42)
    
    elif model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
        model = GradientBoostingClassifier(random_state=42)
    
    elif model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
        model = LogisticRegression(max_iter=1000, random_state=42)
    
    else:
        print(f"Tuning not configured for {model_name}")
        return None
    
    print("\nSearching for best parameters...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV F1-Score: {grid_search.best_score_:.4f}")
    
    # Evaluate tuned model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"\nTuned Model Performance on Test Set:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_pred):.4f}")
    
    return best_model


# ============================================================================
# 6. SAVE MODELS AND ARTIFACTS
# ============================================================================

def save_artifacts(model, scaler, feature_names):
    """Save trained model and preprocessing artifacts"""
    print("\n" + "="*70)
    print("SAVING MODELS & ARTIFACTS")
    print("="*70)
    
    # Save model
    with open('/home/claude/churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Model saved: churn_model.pkl")
    
    # Save scaler
    with open('/home/claude/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved: scaler.pkl")
    
    # Save feature names
    with open('/home/claude/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("✓ Feature names saved: feature_names.pkl")


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("CUSTOMER CHURN PREDICTION - ML PIPELINE")
    print("="*70)
    
    # 1. Load data
    print("\nLoading data...")
    df = pd.read_csv('/mnt/user-data/uploads/customer_subscription_churn_usage_patterns.csv')
    print(f"✓ Loaded {len(df)} records")
    
    # 2. Feature Engineering
    df_engineered = engineer_features(df)
    
    # 3. Preprocessing
    X, y, le_target = preprocess_data(df_engineered)
    
    # 4. Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # 5. Handle imbalance (optional - uncomment if needed)
    # X_train, y_train = handle_imbalance(X_train, y_train)
    
    # 6. Train models
    results, trained_models = train_models(X_train, y_train, X_test, y_test)
    
    # 7. Evaluate and compare
    best_model_name, comparison_df = evaluate_models(results, y_test)
    
    # 8. Detailed report
    detailed_classification_report(results, y_test, best_model_name)
    
    # 9. Hyperparameter tuning
    tuned_model = tune_best_model(X_train, y_train, X_test, y_test, best_model_name)
    
    # 10. Save artifacts
    final_model = tuned_model if tuned_model is not None else trained_models[best_model_name]
    save_artifacts(final_model, scaler, X.columns.tolist())
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  • model_evaluation.png")
    print("  • churn_model.pkl")
    print("  • scaler.pkl")
    print("  • feature_names.pkl")
    print("\nYou can now use the trained model for predictions!")
    print("="*70)


if __name__ == "__main__":
    main()
