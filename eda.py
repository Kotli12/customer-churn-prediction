"""Customer Churn Prediction — Exploratory Data Analysis"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
DATA_PATH = ROOT / 'customer_subscription_churn_usage_patterns.csv'
PLOTS_DIR = ROOT / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def load_and_summarize() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    print(f"\nMissing values: {df.isnull().sum().sum()} total")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"\nChurn distribution:")
    dist = df['churn'].value_counts()
    pct = df['churn'].value_counts(normalize=True).mul(100)
    for v in dist.index:
        print(f"  {v}: {dist[v]} ({pct[v]:.1f}%)")
    print(f"\nNumerical summary:")
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != 'user_id']
    print(df[num_cols].describe().round(2).to_string())
    return df


def plot_churn_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    counts = df['churn'].value_counts()
    counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'], edgecolor='white')
    axes[0].set_title('Churn Count', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=0)
    for bar in axes[0].patches:
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 10, str(int(bar.get_height())),
                     ha='center', va='bottom', fontsize=11)

    axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90,
                textprops={'fontsize': 12})
    axes[1].set_title('Churn Rate', fontweight='bold')

    plan_churn = df.groupby('plan_type')['churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).sort_values(ascending=False)
    plan_churn.plot(kind='bar', ax=axes[2], color='coral', edgecolor='white')
    axes[2].set_title('Churn Rate by Plan', fontweight='bold')
    axes[2].set_ylabel('Churn Rate (%)')
    axes[2].tick_params(axis='x', rotation=30)
    overall = (df['churn'] == 'Yes').mean() * 100
    axes[2].axhline(y=overall, color='navy', linestyle='--', label=f'Overall ({overall:.1f}%)')
    axes[2].legend()

    plt.tight_layout()
    path = PLOTS_DIR / 'churn_overview.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    numerical = [
        'avg_weekly_usage_hours', 'support_tickets', 'payment_failures',
        'tenure_months', 'last_login_days_ago', 'monthly_fee',
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, col in enumerate(numerical):
        for label, color in [('No', '#2ecc71'), ('Yes', '#e74c3c')]:
            data = df[df['churn'] == label][col]
            axes[i].hist(data, bins=30, alpha=0.6, color=color,
                         label=f'Churn={label}', density=True)
        axes[i].set_title(col.replace('_', ' ').title(), fontweight='bold')
        axes[i].legend(fontsize=9)

    plt.suptitle('Feature Distributions by Churn Status', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = PLOTS_DIR / 'feature_distributions.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    df_num = df.copy()
    df_num['churn_binary'] = (df_num['churn'] == 'Yes').astype(int)
    df_num = pd.get_dummies(df_num, columns=['plan_type'], drop_first=True)
    cols = [c for c in df_num.select_dtypes(include=np.number).columns if c != 'user_id']

    corr = df_num[cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = PLOTS_DIR / 'correlation_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_usage_and_tenure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    overall_rate = (df['churn'] == 'Yes').mean() * 100

    usage_bins = pd.cut(df['avg_weekly_usage_hours'],
                        bins=[0, 5, 10, 15, 100],
                        labels=['<5h', '5-10h', '10-15h', '15h+'])
    usage_churn = df.assign(usage_bin=usage_bins).groupby('usage_bin', observed=True)['churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    )
    usage_churn.plot(kind='bar', ax=axes[0], color='#e74c3c', edgecolor='white')
    axes[0].set_title('Churn Rate by Weekly Usage', fontweight='bold')
    axes[0].set_ylabel('Churn Rate (%)')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].axhline(y=overall_rate, color='navy', linestyle='--', label=f'Overall ({overall_rate:.1f}%)')
    axes[0].legend()

    tenure_bins = pd.cut(df['tenure_months'],
                         bins=[0, 6, 12, 24, 200],
                         labels=['0-6m', '6-12m', '12-24m', '24m+'])
    tenure_churn = df.assign(tenure_bin=tenure_bins).groupby('tenure_bin', observed=True)['churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    )
    tenure_churn.plot(kind='bar', ax=axes[1], color='teal', edgecolor='white')
    axes[1].set_title('Churn Rate by Tenure', fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].axhline(y=overall_rate, color='navy', linestyle='--', label=f'Overall ({overall_rate:.1f}%)')
    axes[1].legend()

    plt.tight_layout()
    path = PLOTS_DIR / 'usage_and_tenure.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def print_churn_drivers(df: pd.DataFrame) -> None:
    print("\n--- Churn Drivers ---")
    for col in ['avg_weekly_usage_hours', 'support_tickets', 'payment_failures', 'last_login_days_ago']:
        churned = df[df['churn'] == 'Yes'][col].mean()
        retained = df[df['churn'] == 'No'][col].mean()
        diff = churned - retained
        print(f"{col:30s}  churned={churned:.2f}  retained={retained:.2f}  diff={diff:+.2f}")


if __name__ == '__main__':
    print("=== Customer Churn — EDA ===\n")
    df = load_and_summarize()
    plot_churn_overview(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_usage_and_tenure(df)
    print_churn_drivers(df)
    print(f"\nDone. Plots saved to {PLOTS_DIR}/")
