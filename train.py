"""
Customer Churn Prediction — Model Training
==========================================
Every decision here is justified by a business reason, not just a technical one.
"""

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

ROOT   = Path(__file__).parent
DATA   = ROOT / 'customer_subscription_churn_usage_patterns.csv'
MODELS = ROOT / 'models'
PLOTS  = ROOT / 'plots'
MODELS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

RED  = '#C0392B'
GREEN= '#27AE60'
BLUE = '#2980B9'


def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def load_data():
    df       = pd.read_csv(DATA)
    X        = prepare_features(df)
    y        = (df['churn'] == 'Yes').astype(int)
    n_churned= y.sum()
    avg_fee  = df['monthly_fee'].mean()
    _section("STEP 1 — BUSINESS PROBLEM")
    print(f"  Customers in dataset : {len(y):,}")
    print(f"  Annual churn rate    : {y.mean()*100:.1f}%  (industry benchmark ≈ 5–7%)")
    print(f"  Customers lost/year  : {n_churned:,}")
    print(f"  Avg annual value     : ${avg_fee*12:,.0f} per customer")
    print(f"  Revenue at risk      : ${n_churned*avg_fee*12/1e6:.1f}M per year")
    print()
    print("  Why build a model?")
    print("  Retaining a customer costs 5–7× less than acquiring a new one.")
    print("  Catching even 40% of churners before they leave pays for itself many times over.")
    return X, y, df


def explain_features() -> None:
    _section("STEP 2 — WHY THESE FEATURES?")
    rows = [
        ("Payment failures",   "Financial friction is the #1 early warning sign"),
        ("Days since login",   "Disengagement precedes cancellation by 2–4 weeks"),
        ("Support tickets",    "Unresolved issues erode trust and loyalty"),
        ("Weekly usage (hrs)", "Low usage = low perceived value = high exit risk"),
        ("Tenure",             "New customers (0–6 mo) churn at the highest rate"),
        ("Plan type",          "Plan tier sets expectations — mismatches drive churn"),
        ("Risk score",         "Composite signal: combines all negatives into one KPI"),
        ("Usage × tenure",     "A long-term low-usage customer is a silent defector"),
    ]
    for feature, reason in rows:
        print(f"  ✓  {feature:<25}  {reason}")
    print(f"\n  Total features used: {len(FEATURE_COLS)}")


def explain_models() -> None:
    _section("STEP 3 — WHY THESE MODELS?")
    rows = [
        ("Logistic Regression", "Baseline. Simple, fast, interpretable. Sets the floor."),
        ("Random Forest",       "Ensemble of decision trees. Gives feature importance — tells us *which* factors drive churn."),
        ("Gradient Boosting",   "Builds trees sequentially, each fixing the last one's mistakes. Strong on tabular data."),
        ("XGBoost",             "Optimised gradient boosting. Handles class imbalance via scale_pos_weight."),
        ("LightGBM",            "Fastest of the ensemble methods. Good when we later retrain on more data."),
    ]
    for name, reason in rows:
        print(f"  {name}\n    → {reason}\n")


def explain_metric_choice(avg_annual_value: float) -> None:
    _section("STEP 4 — WHY WE OPTIMISE FOR RECALL, NOT ACCURACY")
    print(f"  Missing a churner (False Negative) costs : ${avg_annual_value:,.0f}/year")
    print(f"  False alarm (False Positive) costs       : $50/outreach call")
    print()
    print("  The asymmetry is clear: we should cast a wide net.")
    print("  A model with high Recall catches more churners, even at the cost of")
    print("  flagging some customers who would have stayed. That trade-off is worth it.")
    print()
    print("  Primary metric → ROC-AUC  (ranks all customers from riskiest to safest)")
    print("  Secondary metric → Recall (what % of real churners we catch)")


def train_all_models(X, y, df):
    _section("STEP 5 — MODEL TRAINING & EVALUATION")
    pos_weight = (y == 0).sum() / (y == 1).sum()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Training set: {len(X_train):,} customers")
    print(f"  Test set    : {len(X_test):,} customers  (held out — model never sees this)")
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_te_s  = scaler.transform(X_test)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced', random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=400, learning_rate=0.04, max_depth=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0),
        'LightGBM':            LGBMClassifier(n_estimators=400, learning_rate=0.04, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    print(f"\n  {'Model':<22} {'Accuracy':>9} {'Recall':>8} {'ROC-AUC':>9} {'CV-AUC':>8}")
    print(f"  {'─'*22} {'─'*9} {'─'*8} {'─'*9} {'─'*8}")
    for name, model in models.items():
        cv_auc  = cross_val_score(model, X_tr_s, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        model.fit(X_tr_s, y_train)
        y_pred  = model.predict(X_te_s)
        y_prob  = model.predict_proba(X_te_s)[:, 1]
        results[name] = dict(model=model, cv_auc=cv_auc,
            accuracy=accuracy_score(y_test, y_pred), precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred), f1=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_prob), y_pred=y_pred, y_prob=y_prob)
        r = results[name]
        print(f"  {name:<22} {r['accuracy']:>9.3f} {r['recall']:>8.3f} {r['roc_auc']:>9.3f} {cv_auc:>8.3f}")
    return results, X_test, y_test, scaler


def select_best(results, y_test, df):
    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    best      = results[best_name]
    avg_fee   = df['monthly_fee'].mean()
    _section("STEP 6 — MODEL SELECTION & BUSINESS IMPACT")
    print(f"  Best model  : {best_name}")
    print(f"  ROC-AUC     : {best['roc_auc']:.3f}  (ranks a real churner above a non-churner {best['roc_auc']*100:.0f}% of the time)")
    print(f"  Recall      : {best['recall']:.3f}  (we catch {best['recall']*100:.0f}% of customers who would have churned)")
    print(f"  Precision   : {best['precision']:.3f}  (of those we flag, {best['precision']*100:.0f}% are genuine churners)")
    print()
    n_churned     = (y_test == 1).sum() * 5
    caught        = int(n_churned * best['recall'])
    saved         = int(caught * 0.40)
    revenue_saved = saved * avg_fee * 12
    campaign_cost = caught * 50
    print("  PROJECTED ANNUAL BUSINESS IMPACT")
    print(f"  {'─'*42}")
    print(f"  Churners we can identify  : {caught:,} customers")
    print(f"  Retained at 40% save rate : {saved:,} customers")
    print(f"  Revenue protected         : ${revenue_saved:,.0f}")
    print(f"  Campaign cost (@ $50/call): ${campaign_cost:,.0f}")
    print(f"  Net gain                  : ${revenue_saved - campaign_cost:,.0f}")
    print()
    print("  DETAILED CLASSIFICATION REPORT (test set)")
    print(classification_report(y_test, best['y_pred'], target_names=['No Churn', 'Churn']))
    return best_name, best


def plot_results(results, y_test, best_name):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Evaluation — Customer Churn Prediction', fontsize=16, fontweight='bold')
    metrics_df = pd.DataFrame([
        {'Model': n, 'Accuracy': r['accuracy'], 'Precision': r['precision'],
         'Recall': r['recall'], 'F1': r['f1'], 'ROC-AUC': r['roc_auc']}
        for n, r in results.items()
    ]).set_index('Model')
    metrics_df.plot(kind='bar', ax=axes[0, 0], colormap='tab10')
    axes[0, 0].set_title('Model Performance Comparison\n(Recall = % of real churners caught)', fontweight='bold')
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].tick_params(axis='x', rotation=30)
    axes[0, 0].legend(loc='lower right', fontsize=8)
    axes[0, 0].grid(axis='y', alpha=0.3)
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        axes[0, 1].plot(fpr, tpr, linewidth=2.5 if name == best_name else 1.2, label=f"{name} ({r['roc_auc']:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 1].set_xlabel('False Positive Rate (false alarms)')
    axes[0, 1].set_ylabel('True Positive Rate (churners caught)')
    axes[0, 1].set_title('ROC Curves — Higher & left = better\n(Bold = chosen model)', fontweight='bold')
    axes[0, 1].legend(loc='lower right', fontsize=8)
    axes[0, 1].grid(alpha=0.3)
    cm = confusion_matrix(y_test, results[best_name]['y_pred'])
    tn, fp, fn, tp = cm.ravel()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'], annot_kws={'size': 14})
    axes[1, 0].set_title(f'Confusion Matrix — {best_name}\nCaught {tp} churners  |  Missed {fn}  |  False alarms {fp}', fontweight='bold')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xlabel('Predicted')
    best_model = results[best_name]['model']
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(15).sort_values()
        feat_imp.plot(kind='barh', ax=axes[1, 1], color=BLUE, edgecolor='white')
        axes[1, 1].set_title(f'Top 15 Churn Drivers — {best_name}\n(Business: these are the metrics to monitor and act on)', fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = PLOTS / 'model_evaluation.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Chart saved → {path.name}")


def save_artifacts(model, scaler, feature_names):
    _section("STEP 8 — SAVING MODEL ARTIFACTS")
    for obj, fname in [(model, 'model.pkl'), (scaler, 'scaler.pkl'), (feature_names, 'feature_names.pkl')]:
        with open(MODELS / fname, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  Saved → models/{fname}")
    print("\n  These three files are all you need to make predictions on any new customer.")


def main():
    print("\n" + "═" * 60)
    print("  CUSTOMER CHURN PREDICTION — MODEL TRAINING")
    print("═" * 60)
    X, y, df = load_data()
    explain_features()
    explain_models()
    explain_metric_choice(avg_annual_value=df['monthly_fee'].mean() * 12)
    results, X_test, y_test, scaler = train_all_models(X, y, df)
    best_name, best = select_best(results, y_test, df)
    _section("STEP 7 — VISUALISATION")
    plot_results(results, y_test, best_name)
    save_artifacts(best['model'], scaler, FEATURE_COLS)
    print("\n" + "═" * 60)
    print("  TRAINING COMPLETE")
    print("═" * 60)
    print("  Next step: run  python3 predict.py  to score new customers.")


if __name__ == '__main__':
    main()
