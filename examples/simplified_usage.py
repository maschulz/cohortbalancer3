"""
Simplified usage example for CohortBalancer3.

This script demonstrates how to use the simplified CohortBalancer3 API with:
1. Flattened configuration structure
2. Separate visualization functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from cohortbalancer3 import Matcher, MatcherConfig
from cohortbalancer3.visualization import (
    plot_balance, plot_propensity_distributions, 
    plot_treatment_effects, plot_matching_summary
)

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=1000, treatment_effect=1.0):
    """Generate synthetic data for demonstration."""
    # Generate features and treatment assignment
    X, treatment = make_classification(
        n_samples=n_samples, 
        n_features=10, 
        n_informative=5,
        n_redundant=2, 
        random_state=42
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(X.shape[1])])
    df['treatment'] = treatment
    
    # Add categorical variable
    df['cat1'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    df['cat2'] = np.random.choice(['X', 'Y', 'Z'], size=n_samples)
    
    # Create dummy variables for categorical features
    df = pd.get_dummies(df, columns=['cat1', 'cat2'], drop_first=True)
    
    # Generate outcomes with treatment effect
    # Baseline outcome depends on covariates
    baseline = 0.5 * df['x1'] - 0.3 * df['x2'] + 0.2 * df['x3'] + np.random.normal(0, 1, n_samples)
    
    # Generate several outcomes with different treatment effects
    df['outcome1'] = baseline + treatment_effect * treatment + np.random.normal(0, 0.5, n_samples)
    df['outcome2'] = baseline + treatment_effect * 0.5 * treatment + np.random.normal(0, 0.5, n_samples)
    df['outcome3'] = baseline + np.random.normal(0, 0.5, n_samples)  # No treatment effect
    
    return df

def main():
    # Generate sample data
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=1000, treatment_effect=1.0)
    
    # Define covariates
    covariates = [col for col in data.columns if col.startswith('x') or col.startswith('cat')]
    
    # Configure the matcher with flattened configuration
    print("Configuring matcher...")
    config = MatcherConfig(
        # Core parameters
        treatment_col="treatment",
        covariates=covariates,
        
        # Matching parameters
        match_method="greedy",
        distance_method="mahalanobis",
        ratio=2.0,
        caliper=0.2,
        standardize=True,
        random_state=42,
        
        # Propensity parameters
        estimate_propensity=True,
        propensity_model="logistic",
        common_support_trimming=True,
        trim_threshold=0.05,
        
        # Balance parameters
        calculate_balance=True,
        max_standardized_diff=0.1,
        
        # Outcome parameters
        outcomes=["outcome1", "outcome2", "outcome3"],
        effect_method="mean_difference",
        bootstrap_iterations=500
    )
    
    # Create matcher and perform matching
    print("Performing matching...")
    matcher = Matcher(data=data, config=config)
    results = matcher.match().get_results()
    
    # Print summary of matching results
    print("\nMatching Summary:")
    match_summary = results.get_match_summary()
    for key, value in match_summary.items():
        print(f"  {key}: {value}")
    
    # Print balance summary
    print("\nBalance Summary:")
    balance_df = results.get_balance_summary()
    imbalanced_vars = balance_df[balance_df['smd_after'] > 0.1]
    if len(imbalanced_vars) > 0:
        print(f"  {len(imbalanced_vars)}/{len(balance_df)} variables have SMD > 0.1 after matching:")
        for _, row in imbalanced_vars.iterrows():
            print(f"    {row['variable']}: SMD = {row['smd_after']:.3f}")
    else:
        print("  All variables are balanced (SMD ≤ 0.1)")
    
    # Print treatment effect estimates
    print("\nTreatment Effect Estimates:")
    effect_df = results.get_effect_summary()
    for _, row in effect_df.iterrows():
        p_val_str = f"p = {row['p_value']:.3f}"
        significance = " *" if row['p_value'] < 0.05 else ""
        print(f"  {row['outcome']}: Effect = {row['effect']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], {p_val_str}{significance}")
    
    # Create visualizations using the separate visualization functions
    print("\nCreating plots...")
    
    # Create figure for a dashboard of plots
    fig = plt.figure(figsize=(16, 12))
    
    # Balance plot
    plt.subplot(2, 2, 1)
    balance_fig = plot_balance(results, max_vars=10)
    plt.title("Covariate Balance")
    plt.tight_layout()
    plt.close(balance_fig)  # Close the original figure since we've incorporated it
    
    # Propensity score distributions
    plt.subplot(2, 2, 2)
    propensity_fig = plot_propensity_distributions(results)
    plt.title("Propensity Score Distributions")
    plt.tight_layout()
    plt.close(propensity_fig)
    
    # Treatment effects
    plt.subplot(2, 2, 3)
    effects_fig = plot_treatment_effects(results)
    plt.title("Treatment Effect Estimates")
    plt.tight_layout()
    plt.close(effects_fig)
    
    # Matching summary
    plt.subplot(2, 2, 4)
    summary_fig = plot_matching_summary(results)
    plt.title("Sample Sizes Before and After Matching")
    plt.tight_layout()
    plt.close(summary_fig)
    
    # Save the dashboard figure
    plt.tight_layout()
    plt.savefig("matching_dashboard.png")
    print("Plots saved to 'matching_dashboard.png'")
    
    # You can also save individual plots if needed
    plot_balance(results).savefig("balance_plot.png")
    print("Balance plot also saved separately to 'balance_plot.png'")

if __name__ == "__main__":
    main() 