"""
Customer Churn Prediction - Exploratory Data Analysis
======================================================
Phase 1: Understanding the data and uncovering patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. DATA LOADING & INITIAL INSPECTION
# ============================================================================

def load_data(filepath):
    """Load the dataset and perform initial inspection"""
    df = pd.read_csv(filepath)
    
    print("="*70)
    print("DATASET OVERVIEW")
    print("="*70)
    print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Names:\n{df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def data_quality_check(df):
    """Comprehensive data quality assessment"""
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    print("\nMissing Values:")
    print(missing_df[missing_df['Missing_Count'] > 0])
    if missing.sum() == 0:
        print("✓ No missing values found!")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")
    if duplicates == 0:
        print("✓ No duplicate rows found!")
    
    # Duplicate user_ids
    duplicate_ids = df['user_id'].duplicated().sum()
    print(f"Duplicate User IDs: {duplicate_ids}")
    
    return missing_df


def statistical_summary(df):
    """Generate statistical summary of numerical features"""
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'user_id' in numerical_cols:
        numerical_cols.remove('user_id')
    
    print("\nNumerical Features Summary:")
    print(df[numerical_cols].describe().round(2))
    
    return numerical_cols


# ============================================================================
# 2. TARGET VARIABLE ANALYSIS
# ============================================================================

def analyze_churn_distribution(df):
    """Analyze churn rate and distribution"""
    print("\n" + "="*70)
    print("CHURN ANALYSIS")
    print("="*70)
    
    churn_counts = df['churn'].value_counts()
    churn_pct = df['churn'].value_counts(normalize=True) * 100
    
    print("\nChurn Distribution:")
    print(f"{'Category':<15} {'Count':<10} {'Percentage':<10}")
    print("-" * 35)
    for category in churn_counts.index:
        print(f"{category:<15} {churn_counts[category]:<10} {churn_pct[category]:<10.2f}%")
    
    overall_churn_rate = (df['churn'] == 'Yes').sum() / len(df) * 100
    print(f"\n📊 Overall Churn Rate: {overall_churn_rate:.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    churn_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Churn Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Churn Status', fontsize=12)
    axes[0].set_ylabel('Number of Customers', fontsize=12)
    axes[0].tick_params(axis='x', rotation=0)
    
    # Pie chart
    axes[1].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
                startangle=90, colors=['#2ecc71', '#e74c3c'], textprops={'fontsize': 12})
    axes[1].set_title('Churn Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/claude/churn_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: churn_distribution.png")
    plt.close()
    
    return overall_churn_rate


# ============================================================================
# 3. FEATURE ANALYSIS
# ============================================================================

def analyze_categorical_features(df):
    """Analyze categorical features and their relationship with churn"""
    print("\n" + "="*70)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("="*70)
    
    # Plan Type Analysis
    print("\nPlan Type Distribution:")
    plan_churn = pd.crosstab(df['plan_type'], df['churn'], normalize='index') * 100
    print(plan_churn.round(2))
    
    print("\nChurn Rate by Plan Type:")
    for plan in df['plan_type'].unique():
        plan_data = df[df['plan_type'] == plan]
        churn_rate = (plan_data['churn'] == 'Yes').sum() / len(plan_data) * 100
        print(f"  {plan}: {churn_rate:.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plan distribution
    df['plan_type'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Plan Type Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Plan Type', fontsize=12)
    axes[0].set_ylabel('Number of Customers', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Churn by plan
    plan_churn_counts = pd.crosstab(df['plan_type'], df['churn'])
    plan_churn_counts.plot(kind='bar', ax=axes[1], stacked=False, color=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Churn by Plan Type', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Plan Type', fontsize=12)
    axes[1].set_ylabel('Number of Customers', fontsize=12)
    axes[1].legend(['No Churn', 'Churn'])
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/claude/plan_type_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: plan_type_analysis.png")
    plt.close()


def analyze_numerical_features(df, numerical_cols):
    """Analyze numerical features distribution and relationship with churn"""
    print("\n" + "="*70)
    print("NUMERICAL FEATURES ANALYSIS")
    print("="*70)
    
    # Features to analyze (exclude user_id)
    features = [col for col in numerical_cols if col not in ['user_id', 'monthly_fee']]
    
    # Compare churned vs non-churned customers
    print("\nAverage Values by Churn Status:")
    print("-" * 70)
    comparison = df.groupby('churn')[features].mean()
    print(comparison.round(2))
    
    # Create visualizations
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        # Box plot
        df.boxplot(column=feature, by='churn', ax=axes[idx])
        axes[idx].set_title(f'{feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Churn Status', fontsize=10)
        axes[idx].set_ylabel(feature.replace("_", " ").title(), fontsize=10)
        plt.sca(axes[idx])
        plt.xticks(rotation=0)
    
    # Hide extra subplots
    for idx in range(len(features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/numerical_features_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: numerical_features_analysis.png")
    plt.close()


def correlation_analysis(df):
    """Analyze correlations between features"""
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    # Convert churn to binary for correlation
    df_corr = df.copy()
    df_corr['churn_binary'] = (df_corr['churn'] == 'Yes').astype(int)
    df_corr = pd.get_dummies(df_corr, columns=['plan_type'], drop_first=True)
    
    # Select numerical columns
    numerical_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
    if 'user_id' in numerical_cols:
        numerical_cols.remove('user_id')
    
    # Calculate correlation matrix
    corr_matrix = df_corr[numerical_cols].corr()
    
    # Print correlation with churn
    print("\nCorrelation with Churn:")
    churn_corr = corr_matrix['churn_binary'].sort_values(ascending=False)
    print(churn_corr.round(3))
    
    # Visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/home/claude/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: correlation_heatmap.png")
    plt.close()


# ============================================================================
# 4. ADVANCED INSIGHTS
# ============================================================================

def usage_vs_churn_analysis(df):
    """Analyze relationship between usage patterns and churn"""
    print("\n" + "="*70)
    print("USAGE PATTERNS vs CHURN")
    print("="*70)
    
    # Create usage categories
    df['usage_category'] = pd.cut(df['avg_weekly_usage_hours'], 
                                   bins=[0, 5, 10, 15, 100],
                                   labels=['Low (0-5h)', 'Medium (5-10h)', 
                                          'High (10-15h)', 'Very High (15h+)'])
    
    usage_churn = pd.crosstab(df['usage_category'], df['churn'], normalize='index') * 100
    print("\nChurn Rate by Usage Category:")
    print(usage_churn.round(2))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Usage distribution
    df['usage_category'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='coral')
    axes[0].set_title('Customer Distribution by Usage Level', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Usage Category', fontsize=12)
    axes[0].set_ylabel('Number of Customers', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Churn rate by usage
    churn_by_usage = df.groupby('usage_category')['churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).sort_index()
    churn_by_usage.plot(kind='bar', ax=axes[1], color='#e74c3c')
    axes[1].set_title('Churn Rate by Usage Level', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Usage Category', fontsize=12)
    axes[1].set_ylabel('Churn Rate (%)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=df['churn'].value_counts(normalize=True)['Yes']*100, 
                    color='black', linestyle='--', label='Overall Churn Rate')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('/home/claude/usage_vs_churn.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: usage_vs_churn.png")
    plt.close()


def tenure_analysis(df):
    """Analyze relationship between tenure and churn"""
    print("\n" + "="*70)
    print("TENURE ANALYSIS")
    print("="*70)
    
    # Create tenure categories
    df['tenure_category'] = pd.cut(df['tenure_months'], 
                                    bins=[0, 6, 12, 24, 100],
                                    labels=['New (0-6m)', 'Regular (6-12m)', 
                                           'Established (12-24m)', 'Long-term (24m+)'])
    
    tenure_churn = pd.crosstab(df['tenure_category'], df['churn'], normalize='index') * 100
    print("\nChurn Rate by Tenure:")
    print(tenure_churn.round(2))
    
    # Visualization
    plt.figure(figsize=(10, 6))
    churn_by_tenure = df.groupby('tenure_category')['churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).sort_index()
    churn_by_tenure.plot(kind='bar', color='teal')
    plt.title('Churn Rate by Customer Tenure', fontsize=14, fontweight='bold')
    plt.xlabel('Tenure Category', fontsize=12)
    plt.ylabel('Churn Rate (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.axhline(y=df['churn'].value_counts(normalize=True)['Yes']*100, 
                color='red', linestyle='--', label='Overall Churn Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/claude/tenure_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: tenure_analysis.png")
    plt.close()


# ============================================================================
# 5. KEY INSIGHTS SUMMARY
# ============================================================================

def generate_insights_report(df):
    """Generate a summary of key insights"""
    print("\n" + "="*70)
    print("KEY INSIGHTS & FINDINGS")
    print("="*70)
    
    insights = []
    
    # Overall churn rate
    churn_rate = (df['churn'] == 'Yes').sum() / len(df) * 100
    insights.append(f"1. Overall churn rate is {churn_rate:.1f}%")
    
    # Plan-based insights
    plan_churn = df.groupby('plan_type')['churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    highest_churn_plan = plan_churn.idxmax()
    insights.append(f"2. {highest_churn_plan} plan has the highest churn rate at {plan_churn[highest_churn_plan]:.1f}%")
    
    # Usage insights
    churned = df[df['churn'] == 'Yes']['avg_weekly_usage_hours'].mean()
    retained = df[df['churn'] == 'No']['avg_weekly_usage_hours'].mean()
    insights.append(f"3. Churned customers average {churned:.1f}h/week vs {retained:.1f}h/week for retained customers")
    
    # Support tickets
    churned_tickets = df[df['churn'] == 'Yes']['support_tickets'].mean()
    retained_tickets = df[df['churn'] == 'No']['support_tickets'].mean()
    insights.append(f"4. Churned customers had {churned_tickets:.1f} support tickets vs {retained_tickets:.1f} for retained")
    
    # Payment failures
    churned_failures = df[df['churn'] == 'Yes']['payment_failures'].mean()
    retained_failures = df[df['churn'] == 'No']['payment_failures'].mean()
    insights.append(f"5. Churned customers had {churned_failures:.1f} payment failures vs {retained_failures:.1f} for retained")
    
    # Last login
    churned_login = df[df['churn'] == 'Yes']['last_login_days_ago'].mean()
    retained_login = df[df['churn'] == 'No']['last_login_days_ago'].mean()
    insights.append(f"6. Churned customers' last login was {churned_login:.0f} days ago vs {retained_login:.0f} for retained")
    
    for insight in insights:
        print(f"\n{insight}")
    
    return insights


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CUSTOMER CHURN PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_data('/mnt/user-data/uploads/customer_subscription_churn_usage_patterns.csv')
    
    # Data quality check
    data_quality_check(df)
    
    # Statistical summary
    numerical_cols = statistical_summary(df)
    
    # Churn analysis
    analyze_churn_distribution(df)
    
    # Feature analysis
    analyze_categorical_features(df)
    analyze_numerical_features(df, numerical_cols)
    
    # Correlation analysis
    correlation_analysis(df)
    
    # Advanced insights
    usage_vs_churn_analysis(df)
    tenure_analysis(df)
    
    # Generate insights report
    insights = generate_insights_report(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  • churn_distribution.png")
    print("  • plan_type_analysis.png")
    print("  • numerical_features_analysis.png")
    print("  • correlation_heatmap.png")
    print("  • usage_vs_churn.png")
    print("  • tenure_analysis.png")
    print("\nNext Steps:")
    print("  1. Review the visualizations and insights")
    print("  2. Proceed to Feature Engineering (Phase 2)")
    print("  3. Build predictive models (Phase 4)")
    print("="*70)
