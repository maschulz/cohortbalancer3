"""
Advanced usage example for CohortBalancer3.

This script demonstrates:
1. Comparing different matching methods (greedy, optimal, propensity)
2. Using custom weights for distance calculation
3. Exact matching on categorical variables
4. Advanced treatment effect estimation with regression adjustment
5. Comparing and visualizing results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from cohortbalancer3 import Matcher, MatcherConfig
from cohortbalancer3.visualization import (
    plot_balance, plot_love_plot, plot_propensity_distributions, 
    plot_treatment_effects, plot_matching_summary
)

# Set random seed for reproducibility
np.random.seed(42)

def generate_realistic_data(n_samples=2000):
    """Generate realistic data for medical treatment scenario."""
    # Generate base features
    X, _ = make_classification(
        n_samples=n_samples, 
        n_features=8, 
        n_informative=5, 
        random_state=42
    )
    
    # Create DataFrame with interpretable column names
    df = pd.DataFrame()
    
    # Demographics
    df['age'] = 50 + X[:, 0] * 15  # Age between 35-65
    df['female'] = (X[:, 1] > 0).astype(int)  # Binary gender
    
    # Clinical variables
    df['bmi'] = 25 + X[:, 2] * 5  # BMI between 20-30
    df['systolic_bp'] = 120 + X[:, 3] * 20  # Systolic BP
    df['diastolic_bp'] = 80 + X[:, 4] * 10  # Diastolic BP
    df['cholesterol'] = 200 + X[:, 5] * 50  # Cholesterol
    df['glucose'] = 100 + X[:, 6] * 20  # Glucose
    
    # Categorical variables
    df['diabetes'] = (X[:, 7] > 0.5).astype(int)  # Diabetes diagnosis (binary)
    df['smoking'] = np.random.choice(['never', 'former', 'current'], size=n_samples, 
                                   p=[0.5, 0.25, 0.25])  # Smoking status
    df['region'] = np.random.choice(['north', 'south', 'east', 'west'], size=n_samples)
    
    # Create treatment assignment (not completely random - influenced by covariates)
    # Higher probability of treatment for older patients, with diabetes, high BP, etc.
    propensity = 0.3 + 0.1 * (df['age'] - 50) / 15 + 0.2 * df['diabetes'] + 0.1 * (df['systolic_bp'] - 120) / 20
    propensity = np.clip(propensity, 0.05, 0.95)  # Constrain propensity between 0.05 and 0.95
    df['treatment'] = np.random.binomial(1, propensity)
    
    # Store true propensity for evaluation
    df['true_propensity'] = propensity
    
    # Create outcomes with realistic treatment effects
    # Baseline risk based on patient characteristics
    baseline_risk = 0.1 + 0.1 * (df['age'] - 50) / 15 + 0.15 * df['diabetes'] + \
                    0.1 * (df['systolic_bp'] - 120) / 20 + 0.05 * (df['glucose'] - 100) / 20
    
    # Primary outcome: reduction in adverse events (affected by treatment)
    treatment_effect = 0.2  # 20% reduction in events
    adjusted_risk = baseline_risk * (1 - treatment_effect * df['treatment'])
    df['adverse_event'] = np.random.binomial(1, adjusted_risk)
    
    # Secondary outcomes
    # Reduction in BP (larger effect)
    baseline_sbp = df['systolic_bp']
    treatment_effect_sbp = 10  # 10 mmHg average reduction
    noise = np.random.normal(0, 5, n_samples)  # Noise term
    df['systolic_bp_followup'] = baseline_sbp - (treatment_effect_sbp * df['treatment']) + noise
    
    # Reduction in cholesterol (moderate effect)
    baseline_chol = df['cholesterol']
    treatment_effect_chol = 20  # 20 mg/dL average reduction
    noise = np.random.normal(0, 15, n_samples)  # Noise term
    df['cholesterol_followup'] = baseline_chol - (treatment_effect_chol * df['treatment']) + noise
    
    # Quality of life score (small effect)
    baseline_qol = 70 + 0.2 * (df['age'] - 50) - 5 * df['diabetes'] - 0.1 * (df['systolic_bp'] - 120)
    treatment_effect_qol = 5  # 5-point improvement
    noise = np.random.normal(0, 10, n_samples)  # Noise term
    df['quality_of_life'] = baseline_qol + (treatment_effect_qol * df['treatment']) + noise
    
    # Convert categorical variables to dummy variables
    df = pd.get_dummies(df, columns=['smoking', 'region'], drop_first=True)
    
    return df

def compare_matching_methods(data):
    """Compare different matching methods on the same data."""
    # Define shared covariates and outcomes for all configurations
    covariates = [
        'age', 'female', 'bmi', 'systolic_bp', 'diastolic_bp', 
        'cholesterol', 'glucose', 'diabetes', 
        'smoking_former', 'smoking_current',
        'region_south', 'region_east', 'region_west'
    ]
    
    outcomes = ['adverse_event', 'systolic_bp_followup', 'cholesterol_followup', 'quality_of_life']
    
    # Create configurations for different matching methods
    configs = {
        "greedy_mahalanobis": MatcherConfig(
            treatment_col="treatment",
            covariates=covariates,
            # Matching parameters
            match_method="greedy",
            distance_method="mahalanobis",
            exact_match_cols=['diabetes'],
            standardize=True,
            caliper=0.2,
            ratio=1.0,
            # No propensity
            estimate_propensity=False,
            # Balance and outcome
            calculate_balance=True,
            outcomes=outcomes,
            effect_method="regression_adjustment",
            adjustment_covariates=covariates
        ),
        
        "optimal_mahalanobis": MatcherConfig(
            treatment_col="treatment",
            covariates=covariates,
            # Matching parameters
            match_method="optimal",
            distance_method="mahalanobis",
            exact_match_cols=['diabetes'],
            standardize=True,
            ratio=1.0,
            # No propensity
            estimate_propensity=False,
            # Balance and outcome
            calculate_balance=True,
            outcomes=outcomes,
            effect_method="regression_adjustment",
            adjustment_covariates=covariates
        ),
        
        "greedy_propensity": MatcherConfig(
            treatment_col="treatment",
            covariates=covariates,
            # Matching parameters
            match_method="greedy",
            distance_method="propensity",
            standardize=True,
            caliper=0.2,
            ratio=1.0,
            # Propensity settings
            estimate_propensity=True,
            propensity_model="logistic",
            # Balance and outcome
            calculate_balance=True,
            outcomes=outcomes,
            effect_method="regression_adjustment",
            adjustment_covariates=covariates
        ),
        
        "optimal_propensity": MatcherConfig(
            treatment_col="treatment",
            covariates=covariates,
            # Matching parameters
            match_method="optimal",
            distance_method="propensity",
            standardize=True,
            ratio=1.0,
            # Propensity settings
            estimate_propensity=True,
            propensity_model="logistic",
            # Balance and outcome
            calculate_balance=True,
            outcomes=outcomes,
            effect_method="regression_adjustment",
            adjustment_covariates=covariates
        ),
        
        "weighted_greedy": MatcherConfig(
            treatment_col="treatment",
            covariates=covariates,
            # Matching parameters
            match_method="greedy",
            distance_method="euclidean",
            standardize=True,
            caliper=0.2,
            ratio=1.0,
            weights={
                'age': 2.0,
                'systolic_bp': 2.0,
                'diastolic_bp': 2.0,
                'cholesterol': 1.5,
                'glucose': 1.5,
                'diabetes': 3.0
            },
            # No propensity
            estimate_propensity=False,
            # Balance and outcome
            calculate_balance=True,
            outcomes=outcomes,
            effect_method="regression_adjustment",
            adjustment_covariates=covariates
        )
    }
    
    # Run matching for each configuration
    results = {}
    for name, config in configs.items():
        print(f"\nPerforming {name} matching...")
        matcher = Matcher(data=data, config=config)
        results[name] = matcher.match().get_results()
        
    return results

def evaluate_results(results_dict):
    """Evaluate and compare results from different matching methods."""
    # 1. Compare balance metrics
    print("\nComparing Balance Metrics:")
    balance_metrics = {}
    for name, results in results_dict.items():
        balance_index = results.balance_index
        if balance_index:
            metrics = {
                'mean_smd_before': balance_index['mean_smd_before'],
                'mean_smd_after': balance_index['mean_smd_after'],
                'balance_index': balance_index['balance_index'],
                'pct_improved': balance_index['pct_variables_improved']
            }
            balance_metrics[name] = metrics
    
    balance_df = pd.DataFrame(balance_metrics).T
    print(balance_df)
    
    # 2. Compare treatment effect estimates
    print("\nComparing Treatment Effect Estimates:")
    
    # For each outcome, collect estimates across methods
    outcomes = results_dict[list(results_dict.keys())[0]].effect_estimates['outcome'].unique()
    for outcome in outcomes:
        print(f"\nOutcome: {outcome}")
        estimates = {}
        for name, results in results_dict.items():
            effect_df = results.effect_estimates
            outcome_row = effect_df[effect_df['outcome'] == outcome].iloc[0]
            estimates[name] = {
                'effect': outcome_row['effect'],
                'ci_lower': outcome_row['ci_lower'],
                'ci_upper': outcome_row['ci_upper'],
                'p_value': outcome_row['p_value']
            }
        
        # Convert to DataFrame for easier comparison
        estimates_df = pd.DataFrame(estimates).T
        print(estimates_df)
    
    # 3. Compare matched sample sizes
    print("\nComparing Matched Sample Sizes:")
    sample_sizes = {}
    for name, results in results_dict.items():
        summary = results.get_match_summary()
        sample_sizes[name] = {
            'n_treat': summary['n_treatment_matched'],
            'n_control': summary['n_control_matched'],
            'total': summary['n_treatment_matched'] + summary['n_control_matched']
        }
    
    sizes_df = pd.DataFrame(sample_sizes).T
    print(sizes_df)
    
    return balance_df, sizes_df

def plot_comparison(results_dict):
    """Create comparison plots for different matching methods."""
    # Setup plot
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Balance comparison
    plt.subplot(2, 2, 1)
    balance_data = []
    for name, results in results_dict.items():
        before = results.balance_index['mean_smd_before']
        after = results.balance_index['mean_smd_after']
        balance_data.append({
            'Method': name,
            'Before': before,
            'After': after
        })
    
    balance_df = pd.DataFrame(balance_data)
    x = np.arange(len(balance_df))
    width = 0.35
    
    plt.bar(x - width/2, balance_df['Before'], width, label='Before Matching')
    plt.bar(x + width/2, balance_df['After'], width, label='After Matching')
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='0.1 Threshold')
    plt.xticks(x, balance_df['Method'], rotation=45, ha='right')
    plt.ylabel('Mean Standardized Difference')
    plt.title('Balance Comparison Across Methods')
    plt.legend()
    
    # 2. Primary outcome comparison
    plt.subplot(2, 2, 2)
    outcome = 'adverse_event'  # Assuming this is available in all results
    effect_data = []
    for name, results in results_dict.items():
        effect_df = results.effect_estimates
        outcome_row = effect_df[effect_df['outcome'] == outcome].iloc[0]
        effect_data.append({
            'Method': name,
            'Effect': outcome_row['effect'],
            'CI_Lower': outcome_row['ci_lower'],
            'CI_Upper': outcome_row['ci_upper']
        })
    
    effect_df = pd.DataFrame(effect_data)
    x = np.arange(len(effect_df))
    
    plt.errorbar(x, effect_df['Effect'], 
                yerr=[effect_df['Effect'] - effect_df['CI_Lower'], 
                      effect_df['CI_Upper'] - effect_df['Effect']], 
                fmt='o', capsize=5)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xticks(x, effect_df['Method'], rotation=45, ha='right')
    plt.ylabel('Treatment Effect')
    plt.title(f'Treatment Effect Comparison: {outcome}')
    
    # 3. Sample size comparison
    plt.subplot(2, 2, 3)
    size_data = []
    for name, results in results_dict.items():
        summary = results.get_match_summary()
        size_data.append({
            'Method': name,
            'Treatment': summary['n_treatment_matched'],
            'Control': summary['n_control_matched']
        })
    
    size_df = pd.DataFrame(size_data)
    x = np.arange(len(size_df))
    width = 0.35
    
    plt.bar(x - width/2, size_df['Treatment'], width, label='Treatment')
    plt.bar(x + width/2, size_df['Control'], width, label='Control')
    plt.xticks(x, size_df['Method'], rotation=45, ha='right')
    plt.ylabel('Number of Units')
    plt.title('Matched Sample Sizes')
    plt.legend()
    
    # 4. Continuous outcome comparison
    plt.subplot(2, 2, 4)
    outcome = 'systolic_bp_followup'  # Assuming this is available in all results
    effect_data = []
    for name, results in results_dict.items():
        effect_df = results.effect_estimates
        outcome_row = effect_df[effect_df['outcome'] == outcome].iloc[0]
        effect_data.append({
            'Method': name,
            'Effect': outcome_row['effect'],
            'CI_Lower': outcome_row['ci_lower'],
            'CI_Upper': outcome_row['ci_upper']
        })
    
    effect_df = pd.DataFrame(effect_data)
    x = np.arange(len(effect_df))
    
    plt.errorbar(x, effect_df['Effect'], 
                yerr=[effect_df['Effect'] - effect_df['CI_Lower'], 
                      effect_df['CI_Upper'] - effect_df['Effect']], 
                fmt='o', capsize=5)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xticks(x, effect_df['Method'], rotation=45, ha='right')
    plt.ylabel('Treatment Effect')
    plt.title(f'Treatment Effect Comparison: {outcome}')
    
    plt.tight_layout()
    plt.savefig("method_comparison.png")
    print("Comparison plots saved to 'method_comparison.png'")
    
    # Create additional composite visualizations for the best method
    best_method = "optimal_mahalanobis"  # You could determine this programmatically
    if best_method in results_dict:
        best_results = results_dict[best_method]
        
        # Create a new figure for detailed results of the best method
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Balance plot
        balance_fig = plot_balance(best_results, max_vars=10)
        plt.close(balance_fig)  # Close original figure
        
        # Propensity distributions
        if best_results.propensity_scores is not None:
            propensity_fig = plot_propensity_distributions(best_results)
            plt.close(propensity_fig)
        
        # Love plot
        love_fig = plot_love_plot(best_results)
        plt.close(love_fig)
        
        # Treatment effects
        effects_fig = plot_treatment_effects(best_results)
        plt.close(effects_fig)
        
        plt.tight_layout()
        plt.savefig(f"best_method_{best_method}.png")
        print(f"Detailed plots for best method saved to 'best_method_{best_method}.png'")

def main():
    # Generate realistic data
    print("Generating realistic medical treatment data...")
    data = generate_realistic_data(n_samples=2000)
    
    # Compare different matching methods
    print("Comparing different matching methods...")
    results_dict = compare_matching_methods(data)
    
    # Evaluate and compare results
    print("Evaluating results...")
    evaluate_results(results_dict)
    
    # Create comparison plots
    print("Creating comparison plots...")
    plot_comparison(results_dict)

if __name__ == "__main__":
    main() 