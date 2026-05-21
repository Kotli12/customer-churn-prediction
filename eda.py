"""
Customer Churn — Exploratory Data Analysis
==========================================
Each chart answers a specific business question a stakeholder would ask.
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

ROOT     = Path(__file__).parent
DATA     = ROOT / 'customer_subscription_churn_usage_patterns.csv'
PLOTS    = ROOT / 'plots'
PLOTS.mkdir(exist_ok=True)

RED    = '#C0392B'
GREEN  = '#27AE60'
BLUE   = '#2980B9'
ORANGE = '#E67E22'
GREY   = '#7F8C8D'


def _save(fig, name: str) -> None:
    path = PLOTS / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path.name}")


def plot_churn_crisis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('How Critical Is Our Churn Problem?', fontsize=16, fontweight='bold', y=1.01)

    counts   = df['churn'].value_counts()
    churn_rt = counts.get('Yes', 0) / len(df) * 100
    colors   = [GREEN if c == 'No' else RED for c in counts.index]

    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, f'{val:,}',
                     ha='center', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Churn Rate: {churn_rt:.1f}%  ← Industry benchmark is ~5–7%',
                      fontsize=11, color=RED, fontweight='bold')
    axes[0].set_ylabel('Number of Customers')
    axes[0].set_ylim(0, counts.max() * 1.15)
    axes[0].grid(axis='y', alpha=0.3)

    plan_avg_fee    = df.groupby('plan_type')['monthly_fee'].first()
    plan_churners   = df[df['churn'] == 'Yes']['plan_type'].value_counts()
    revenue_at_risk = (plan_churners * plan_avg_fee * 12).sort_values(ascending=False)

    bars2 = axes[1].bar(revenue_at_risk.index, revenue_at_risk.values / 1e6,
                        color=[RED, ORANGE, BLUE][:len(revenue_at_risk)], edgecolor='white', width=0.5)
    for bar, val in zip(bars2, revenue_at_risk.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f'${val/1e6:.1f}M',
                     ha='center', fontsize=11, fontweight='bold')
    total = revenue_at_risk.sum()
    axes[1].set_title(f'Annual Revenue Lost to Churn: ${total/1e6:.1f}M', fontsize=11, color=RED, fontweight='bold')
    axes[1].set_ylabel('Annual Revenue Lost ($M)')
    axes[1].set_ylim(0, revenue_at_risk.max() / 1e6 * 1.2)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, '01_churn_crisis.png')


def plot_who_is_leaving(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Which Customers Are We Losing?', fontsize=16, fontweight='bold', y=1.01)
    overall = (df['churn'] == 'Yes').mean() * 100
    def churn_rate(series): return (series == 'Yes').mean() * 100

    plan_churn = df.groupby('plan_type')['churn'].apply(churn_rate).sort_values(ascending=False)
    bars = axes[0].bar(plan_churn.index, plan_churn.values,
                       color=[RED if v > overall else BLUE for v in plan_churn.values], edgecolor='white', width=0.5)
    for bar, val in zip(bars, plan_churn.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{val:.1f}%',
                     ha='center', fontsize=11, fontweight='bold')
    axes[0].axhline(overall, color=GREY, linestyle='--', label=f'Overall ({overall:.1f}%)')
    axes[0].set_title('Churn Rate by Plan\n(Red = above average)', fontweight='bold')
    axes[0].set_ylabel('Churn Rate (%)')
    axes[0].set_ylim(0, plan_churn.max() * 1.2)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    tenure_labels = ['0–6 mo\n(New)', '6–12 mo\n(Growing)', '12–24 mo\n(Established)', '24+ mo\n(Loyal)']
    df2 = df.copy()
    df2['tenure_bin'] = pd.cut(df2['tenure_months'], bins=[0, 6, 12, 24, 200], labels=tenure_labels, include_lowest=True)
    tenure_churn = df2.groupby('tenure_bin', observed=True)['churn'].apply(churn_rate)
    bars = axes[1].bar(tenure_churn.index, tenure_churn.values,
                       color=[RED if v > overall else BLUE for v in tenure_churn.values], edgecolor='white', width=0.5)
    for bar, val in zip(bars, tenure_churn.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{val:.1f}%',
                     ha='center', fontsize=10, fontweight='bold')
    axes[1].axhline(overall, color=GREY, linestyle='--', label=f'Overall ({overall:.1f}%)')
    axes[1].set_title('Churn Rate by Tenure\n(New customers churn fastest)', fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)')
    axes[1].set_ylim(0, tenure_churn.max() * 1.2)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    usage_labels = ['<5h\n(Inactive)', '5–10h\n(Low)', '10–15h\n(Medium)', '15h+\n(High)']
    df2['usage_bin'] = pd.cut(df2['avg_weekly_usage_hours'], bins=[0, 5, 10, 15, 100], labels=usage_labels, include_lowest=True)
    usage_churn = df2.groupby('usage_bin', observed=True)['churn'].apply(churn_rate)
    bars = axes[2].bar(usage_churn.index, usage_churn.values,
                       color=[RED if v > overall else BLUE for v in usage_churn.values], edgecolor='white', width=0.5)
    for bar, val in zip(bars, usage_churn.values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{val:.1f}%',
                     ha='center', fontsize=10, fontweight='bold')
    axes[2].axhline(overall, color=GREY, linestyle='--', label=f'Overall ({overall:.1f}%)')
    axes[2].set_title('Churn Rate by Weekly Usage\n(Low usage = low perceived value)', fontweight='bold')
    axes[2].set_ylabel('Churn Rate (%)')
    axes[2].set_ylim(0, usage_churn.max() * 1.2)
    axes[2].legend(fontsize=9)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, '02_who_is_leaving.png')


def plot_warning_signs(df: pd.DataFrame) -> None:
    signals = {
        'Payment Failures':   'payment_failures',
        'Days Since Login':   'last_login_days_ago',
        'Support Tickets':    'support_tickets',
        'Weekly Usage (hrs)': 'avg_weekly_usage_hours',
        'Tenure (months)':    'tenure_months',
    }
    fig, axes = plt.subplots(1, len(signals), figsize=(20, 6))
    fig.suptitle('Early Warning Signs: What Separates Churners from Loyal Customers?', fontsize=15, fontweight='bold', y=1.01)
    for ax, (label, col) in zip(axes, signals.items()):
        churned  = df[df['churn'] == 'Yes'][col]
        retained = df[df['churn'] == 'No'][col]
        ax.hist(retained, bins=25, alpha=0.6, color=GREEN, label='Retained', density=True)
        ax.hist(churned,  bins=25, alpha=0.6, color=RED,   label='Churned',  density=True)
        c_mean, r_mean = churned.mean(), retained.mean()
        ax.axvline(c_mean, color=RED,   linestyle='--', linewidth=2)
        ax.axvline(r_mean, color=GREEN, linestyle='--', linewidth=2)
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.set_xlabel(f'Churned avg: {c_mean:.1f}  |  Retained avg: {r_mean:.1f}', fontsize=8)
        ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, '03_warning_signs.png')


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    df_num = df.copy()
    df_num['churn_binary'] = (df_num['churn'] == 'Yes').astype(int)
    df_num = pd.get_dummies(df_num, columns=['plan_type'], drop_first=True)
    cols = [c for c in df_num.select_dtypes(include=np.number).columns if c != 'user_id']
    corr = df_num[cols].corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8}, ax=ax)
    ax.set_title('Which Metrics Are Most Correlated with Churn?\n(Top row / column shows strength of relationship with churn_binary)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, '04_correlation_heatmap.png')


def print_business_summary(df: pd.DataFrame) -> None:
    churn_rt     = (df['churn'] == 'Yes').mean()
    n_churned    = (df['churn'] == 'Yes').sum()
    avg_fee      = df['monthly_fee'].mean()
    revenue_lost = n_churned * avg_fee * 12
    print("\n" + "═" * 60)
    print("  BUSINESS SUMMARY")
    print("═" * 60)
    print(f"  Total customers    : {len(df):,}")
    print(f"  Churn rate         : {churn_rt*100:.1f}%  (industry avg ~5–7%)")
    print(f"  Customers lost/yr  : {n_churned:,}")
    print(f"  Avg monthly fee    : ${avg_fee:.0f}")
    print(f"  Revenue lost/yr    : ${revenue_lost/1e6:.1f}M")
    print()
    print("  TOP WARNING SIGNS (avg difference: churned vs retained)")
    print("  " + "─" * 55)
    for label, col, direction in [
        ('Payment failures',   'payment_failures',      '+'),
        ('Days since login',   'last_login_days_ago',   '+'),
        ('Support tickets',    'support_tickets',        '+'),
        ('Weekly usage (hrs)', 'avg_weekly_usage_hours','−'),
    ]:
        c = df[df['churn'] == 'Yes'][col].mean()
        r = df[df['churn'] == 'No'][col].mean()
        print(f"  {label:<25}  churned={c:.1f}   retained={r:.1f}   gap={direction}{abs(c-r):.1f}")
    print("═" * 60)


if __name__ == '__main__':
    print("=== Customer Churn — Exploratory Data Analysis ===\n")
    df = pd.read_csv(DATA)
    print(f"Dataset: {len(df):,} customers, {df.shape[1]} columns, {(df['churn']=='Yes').mean()*100:.1f}% churn rate\n")
    print("Generating charts...")
    plot_churn_crisis(df)
    plot_who_is_leaving(df)
    plot_warning_signs(df)
    plot_correlation_heatmap(df)
    print_business_summary(df)
    print(f"\nAll charts saved to  →  {PLOTS}/")
