"""Customer Churn Prediction — Training Pipeline"""

from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from features import FEATURE_COLS, prepare_features

ROOT = Path(__file__).parent
DATA_PATH = ROOT / 'customer_subscription_churn_usage_patterns.csv'
MODELS_DIR = ROOT / 'models'
PLOTS_DIR = ROOT / 'plots'
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = prepare_features(df)
    y = (df['churn'] == 'Yes').astype(int)
    print(f"Loaded {len(df)} records  |  churn rate: {y.mean()*100:.1f}%")
    return X, y


def build_models(pos_weight: float) -> dict:
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=0.5, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=4,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0,
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
        ),
    }


def evaluate(model, X_train, y_train, X_test, y_test, name: str) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    result = dict(
        model=model,
        cv_auc=cv_auc,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_prob),
        y_pred=y_pred,
        y_prob=y_prob,
    )
    print(
        f"  {name:<22} acc={result['accuracy']:.3f}  "
        f"auc={result['roc_auc']:.3f}  f1={result['f1']:.3f}  cv_auc={cv_auc:.3f}"
    )
    return result


def plot_evaluation(results: dict, y_test, best_name: str, feature_names: list) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Metrics bar chart
    metrics_df = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1': r['f1'],
            'ROC-AUC': r['roc_auc'],
        }
        for name, r in results.items()
    ]).set_index('Model')
    metrics_df.plot(kind='bar', ax=axes[0, 0], colormap='tab10')
    axes[0, 0].set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].tick_params(axis='x', rotation=30)
    axes[0, 0].legend(loc='lower right', fontsize=8)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. ROC curves
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        axes[0, 1].plot(fpr, tpr, label=f"{name} ({r['roc_auc']:.3f})", linewidth=1.5)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves', fontsize=13, fontweight='bold')
    axes[0, 1].legend(loc='lower right', fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # 3. Confusion matrix (best model)
    cm = confusion_matrix(y_test, results[best_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[1, 0].set_title(f'Confusion Matrix — {best_name}', fontsize=13, fontweight='bold')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xlabel('Predicted')

    # 4. Feature importance (best model)
    best_model = results[best_name]['model']
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = (
            pd.Series(best_model.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
            .head(15)
            .sort_values()
        )
        feat_imp.plot(kind='barh', ax=axes[1, 1], color='teal', edgecolor='white')
        axes[1, 1].set_title(f'Top 15 Feature Importances — {best_name}',
                             fontsize=13, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance not available',
                        ha='center', va='center', fontsize=12)

    plt.tight_layout()
    path = PLOTS_DIR / 'model_evaluation.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def save_artifacts(model, scaler: StandardScaler, feature_names: list) -> None:
    with open(MODELS_DIR / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Artifacts saved → {MODELS_DIR}/")


def main() -> None:
    print("=== Customer Churn — Training Pipeline ===\n")

    X, y = load_data()
    pos_weight = (y == 0).sum() / (y == 1).sum()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\nTraining models:")
    models = build_models(pos_weight)
    results = {
        name: evaluate(model, X_train_s, y_train, X_test_s, y_test, name)
        for name, model in models.items()
    }

    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    best = results[best_name]
    print(f"\nBest model: {best_name}  (ROC-AUC={best['roc_auc']:.4f}  F1={best['f1']:.4f})")

    print("\nDetailed report for best model:")
    print(classification_report(y_test, best['y_pred'], target_names=['No Churn', 'Churn']))

    plot_evaluation(results, y_test, best_name, FEATURE_COLS)
    save_artifacts(best['model'], scaler, FEATURE_COLS)

    print("\nDone.")


if __name__ == '__main__':
    main()
