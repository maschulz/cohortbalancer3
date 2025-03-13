#!/usr/bin/env python
"""
Example script demonstrating CohortBalancer3 usage with visualization outputs.

This script generates synthetic data, performs matching using various methods,
and creates visualizations and tables that are saved to the output directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from cohortbalancer3.datatypes import MatcherConfig
from cohortbalancer3.matcher import Matcher
from cohortbalancer3.visualization import (
    plot_balance, 
    plot_love_plot,
    plot_propensity_distributions, 
    plot_treatment_effects,
    plot_matched_pairs_distance,
    plot_propensity_calibration,
    plot_matching_summary,
    plot_propensity_comparison,
    plot_covariate_distributions,
    plot_matched_pairs_scatter
)

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=500):
    """Generate synthetic data for matching example."""
    # Features
    age = np.random.normal(50, 15, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    systolic_bp = np.random.normal(120, 15, n_samples)
    cholesterol = np.random.normal(200, 40, n_samples)
    
    # Binary features
    sex = np.random.binomial(1, 0.5, n_samples)  # 0=female, 1=male
    smoker = np.random.binomial(1, 0.3, n_samples)
    
    # Categorical feature (converted to dummies)
    blood_type = np.random.choice(['A', 'B', 'AB', 'O'], size=n_samples, p=[0.4, 0.1, 0.1, 0.4])
    
    # Treatment assignment (more likely for older, male, smokers with high bp)
    logit = -2 + 0.03 * age + 0.5 * sex + 0.8 * smoker + 0.02 * systolic_bp
    prob_treatment = 1 / (1 + np.exp(-logit))
    treatment = np.random.binomial(1, prob_treatment)
    
    # True propensity scores
    true_propensity = prob_treatment
    
    # Outcome with treatment effect = 5.0 (higher is worse)
    outcome = 50 + 5.0 * treatment + 0.1 * age + 5 * smoker + 0.05 * systolic_bp + np.random.normal(0, 5, n_samples)
    
    # Secondary outcome - more affected by age
    outcome2 = 20 + 3.0 * treatment + 0.2 * age + 0.1 * bmi + np.random.normal(0, 3, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'cholesterol': cholesterol,
        'smoker': smoker,
        'blood_type': blood_type,
        'treatment': treatment,
        'true_propensity': true_propensity,
        'outcome': outcome,
        'outcome2': outcome2
    })
    
    # Create dummy variables for blood type and convert to int
    blood_type_dummies = pd.get_dummies(df['blood_type'], prefix='blood')
    # Convert boolean columns to integers (0/1)
    blood_type_dummies = blood_type_dummies.astype(int)
    df = pd.concat([df, blood_type_dummies], axis=1)
    
    return df

def perform_matching(data, method="greedy", use_propensity=True):
    """Perform matching using specified method."""
    
    # Define covariates for matching
    covariates = [
        'age', 'sex', 'bmi', 'systolic_bp', 'cholesterol', 'smoker', 
        'blood_A', 'blood_B', 'blood_AB', 'blood_O'
    ]
    
    # Create matching configuration
    config = MatcherConfig(
        treatment_col='treatment',
        covariates=covariates,
        outcomes=['outcome', 'outcome2'],
        match_method=method,  # 'greedy' or 'optimal'
        distance_method='propensity' if use_propensity else 'mahalanobis',
        standardize=True,
        caliper='auto',
        exact_match_cols=['sex'],  # Exact matching on sex
        estimate_propensity=use_propensity,
        propensity_col=None if use_propensity else 'true_propensity',
        random_state=42,
        calculate_balance=True,
        ratio=1.0  # 1:1 matching
    )
    
    # Initialize and perform matching
    matcher = Matcher(data=data, config=config)
    matcher.match()
    
    # Get results
    results = matcher.get_results()
    
    return results

def save_visualizations(results, prefix=""):
    """Create and save visualizations."""
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        prefix = f"{prefix}_"
    
    # 1. Balance plot
    fig_balance = plot_balance(results, max_vars=15, figsize=(12, 8))
    fig_balance.savefig(os.path.join(OUTPUT_DIR, f"{prefix}balance_plot_{timestamp}.png"), dpi=300)
    plt.close(fig_balance)
    
    # 2. Love plot (alternative balance visualization)
    fig_love = plot_love_plot(results, figsize=(10, 14))
    fig_love.savefig(os.path.join(OUTPUT_DIR, f"{prefix}love_plot_{timestamp}.png"), dpi=300)
    plt.close(fig_love)
    
    # 3. Propensity score distributions (if available)
    if results.propensity_scores is not None:
        fig_propensity = plot_propensity_distributions(results)
        fig_propensity.savefig(os.path.join(OUTPUT_DIR, f"{prefix}propensity_distributions_{timestamp}.png"), dpi=300)
        plt.close(fig_propensity)
        
        # 4. Propensity calibration plot
        fig_calibration = plot_propensity_calibration(results)
        fig_calibration.savefig(os.path.join(OUTPUT_DIR, f"{prefix}propensity_calibration_{timestamp}.png"), dpi=300)
        plt.close(fig_calibration)
        
        # NEW: Propensity score comparison (before vs after matching)
        fig_ps_comp = plot_propensity_comparison(results, figsize=(12, 6))
        fig_ps_comp.savefig(os.path.join(OUTPUT_DIR, f"{prefix}propensity_comparison_{timestamp}.png"), dpi=300)
        plt.close(fig_ps_comp)
        print(f"- Created propensity score comparison (before vs. after matching)")
    
    # 5. Treatment effects
    fig_effects = plot_treatment_effects(results)
    fig_effects.savefig(os.path.join(OUTPUT_DIR, f"{prefix}treatment_effects_{timestamp}.png"), dpi=300)
    plt.close(fig_effects)
    
    # 6. Matched pairs distance
    fig_distances = plot_matched_pairs_distance(results)
    fig_distances.savefig(os.path.join(OUTPUT_DIR, f"{prefix}matched_distances_{timestamp}.png"), dpi=300)
    plt.close(fig_distances)
    
    # 7. Matching summary
    fig_summary = plot_matching_summary(results)
    fig_summary.savefig(os.path.join(OUTPUT_DIR, f"{prefix}matching_summary_{timestamp}.png"), dpi=300)
    plt.close(fig_summary)
    
    # NEW: 8. Covariate distributions before and after matching
    fig_cov_dist = plot_covariate_distributions(results, max_vars=8, figsize=(14, 16))
    fig_cov_dist.savefig(os.path.join(OUTPUT_DIR, f"{prefix}covariate_distributions_{timestamp}.png"), dpi=300)
    plt.close(fig_cov_dist)
    print(f"- Created covariate distributions before vs. after matching")
    
    # NEW: 9. Matched pairs scatter plot
    # Choose the two covariates with highest standardized mean difference before matching
    if results.balance_statistics is not None:
        sorted_vars = results.balance_statistics.sort_values('smd_before', ascending=False)
        # Get top 2 covariates that are in the data (not treatment or outcome)
        covariates = results.config.covariates
        top_vars = [var for var in sorted_vars['variable'] if var in covariates]
        
        if len(top_vars) >= 2:
            x_var, y_var = top_vars[0], top_vars[1]
            fig_scatter = plot_matched_pairs_scatter(results, x_var=x_var, y_var=y_var, figsize=(10, 10))
            fig_scatter.savefig(os.path.join(OUTPUT_DIR, f"{prefix}matched_pairs_scatter_{timestamp}.png"), dpi=300)
            plt.close(fig_scatter)
            print(f"- Created matched pairs scatter plot for {x_var} vs {y_var}")
    
    # Print success message
    print(f"Visualizations saved to {OUTPUT_DIR}")

def save_tables(results, prefix=""):
    """Save tables to CSV files."""
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        prefix = f"{prefix}_"
    
    # 1. Matched data
    results.matched_data.to_csv(
        os.path.join(OUTPUT_DIR, f"{prefix}matched_data_{timestamp}.csv"), 
        index=True
    )
    
    # 2. Balance statistics
    if results.balance_statistics is not None:
        results.balance_statistics.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}balance_statistics_{timestamp}.csv"), 
            index=False
        )
    
    # 3. Effect estimates
    if results.effect_estimates is not None:
        results.effect_estimates.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}effect_estimates_{timestamp}.csv"), 
            index=False
        )
    
    # 4. Match pairs (convert to DataFrame first)
    match_pairs_df = results.get_match_pairs()
    match_pairs_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{prefix}match_pairs_{timestamp}.csv"), 
        index=False
    )
    
    # 5. Rubin statistics (if available)
    if results.rubin_statistics is not None:
        pd.DataFrame([results.rubin_statistics]).to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}rubin_statistics_{timestamp}.csv"), 
            index=False
        )
    
    # 6. Balance index (if available)
    if results.balance_index is not None:
        pd.DataFrame([results.balance_index]).to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}balance_index_{timestamp}.csv"), 
            index=False
        )
    
    # Print success message
    print(f"Tables saved to {OUTPUT_DIR}")

def generate_html_report(results, method_name, filename="matching_report.html"):
    """Generate a simple HTML report summarizing the matching results."""
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get basic statistics
    n_total = len(results.original_data)
    n_matched = len(results.matched_data)
    n_treat_orig = (results.original_data['treatment'] == 1).sum()
    n_control_orig = (results.original_data['treatment'] == 0).sum()
    n_treat_matched = (results.matched_data['treatment'] == 1).sum()
    n_control_matched = (results.matched_data['treatment'] == 0).sum()
    
    # Extract treatment effects
    effect_html = ""
    if results.effect_estimates is not None:
        effect_html = results.effect_estimates.to_html(
            float_format="%.4f", 
            classes="table table-striped"
        )
    
    # Prepare balance statistics summary
    balance_html = ""
    if results.balance_statistics is not None:
        # Get top 10 covariates with highest SMD before matching
        top_covariates = results.balance_statistics.sort_values(
            'smd_before', ascending=False
        ).head(10)
        
        balance_html = top_covariates.to_html(
            float_format="%.4f",
            classes="table table-striped"
        )
    
    # Generate file paths for images
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_paths = {
        'balance': f"balance_plot_{timestamp_str}.png",
        'love': f"love_plot_{timestamp_str}.png",
        'propensity': f"propensity_distributions_{timestamp_str}.png",
        'treatment': f"treatment_effects_{timestamp_str}.png",
        'distances': f"matched_distances_{timestamp_str}.png",
        'summary': f"matching_summary_{timestamp_str}.png",
        # New visualization paths
        'propensity_comparison': f"propensity_comparison_{timestamp_str}.png",
        'covariate_distributions': f"covariate_distributions_{timestamp_str}.png",
        'matched_pairs_scatter': f"matched_pairs_scatter_{timestamp_str}.png",
    }
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Matching Results: {method_name}</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {{ padding: 20px; }}
            .img-container img {{ max-width: 100%; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Matching Results: {method_name}</h1>
            <p class="text-muted">Generated on {timestamp}</p>
            
            <h2>Summary Statistics</h2>
            <div class="row">
                <div class="col-md-6">
                    <table class="table table-bordered">
                        <tr><th>Total samples</th><td>{n_total}</td></tr>
                        <tr><th>Matched samples</th><td>{n_matched}</td></tr>
                        <tr><th>Treatment (original)</th><td>{n_treat_orig}</td></tr>
                        <tr><th>Control (original)</th><td>{n_control_orig}</td></tr>
                        <tr><th>Treatment (matched)</th><td>{n_treat_matched}</td></tr>
                        <tr><th>Control (matched)</th><td>{n_control_matched}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <div class="img-container">
                        <img src="{image_paths['summary']}" alt="Matching Summary">
                    </div>
                </div>
            </div>
            
            <h2>Balance Overview</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="img-container">
                        <img src="{image_paths['balance']}" alt="Balance Plot">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="img-container">
                        <img src="{image_paths['love']}" alt="Love Plot">
                    </div>
                </div>
            </div>
            
            <h2>Top Covariates by Standardized Mean Difference (Before Matching)</h2>
            {balance_html}
            
            <h2>Propensity Score Distributions</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="img-container">
                        <img src="{image_paths['propensity']}" alt="Propensity Distributions">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="img-container">
                        <img src="{image_paths['propensity_comparison']}" alt="Propensity Comparison">
                        <p>Comparison of propensity score distributions before matching (left) and after matching (right).
                        This visualization shows how matching improves the overlap between treatment and control groups.</p>
                    </div>
                </div>
            </div>
            
            <h2>Treatment Effects</h2>
            {effect_html}
            <div class="row">
                <div class="col-md-12">
                    <div class="img-container">
                        <img src="{image_paths['treatment']}" alt="Treatment Effects">
                    </div>
                </div>
            </div>
            
            <h2>Matching Quality</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="img-container">
                        <img src="{image_paths['distances']}" alt="Match Distances">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="img-container">
                        <img src="{image_paths['matched_pairs_scatter']}" alt="Matched Pairs Scatter">
                        <p>Scatter plot showing matched pairs in the space of two covariates with the highest
                        standardized mean difference before matching. Blue points represent treatment units, 
                        orange points represent control units, and connecting lines show the matches. Longer lines
                        indicate pairs that are more distant in these dimensions.</p>
                    </div>
                </div>
            </div>
            
            <h2>Covariate Distributions</h2>
            <div class="row">
                <div class="col-md-12">
                    <div class="img-container">
                        <img src="{image_paths['covariate_distributions']}" alt="Covariate Distributions">
                        <p>This multi-panel plot shows distributions of individual covariates before matching (left) and
                        after matching (right). Each row represents one covariate, with standardized mean difference (SMD)
                        values displayed to quantify the improvement in balance after matching.</p>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_path}")
    
    return output_path

def main():
    """Main function to run the example."""
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=1000)
    
    # Save original data
    data.to_csv(os.path.join(OUTPUT_DIR, "synthetic_data.csv"), index=False)
    print(f"Synthetic data saved to {os.path.join(OUTPUT_DIR, 'synthetic_data.csv')}")
    
    print("\n1. Running greedy matching with propensity scores...")
    results_greedy = perform_matching(data, method="greedy", use_propensity=True)
    save_visualizations(results_greedy, prefix="greedy_propensity")
    save_tables(results_greedy, prefix="greedy_propensity")
    generate_html_report(results_greedy, "Greedy Matching with Propensity Scores", 
                         "greedy_propensity_report.html")
    
    print("\n2. Running optimal matching with Mahalanobis distance...")
    results_optimal = perform_matching(data, method="optimal", use_propensity=False)
    save_visualizations(results_optimal, prefix="optimal_mahalanobis")
    save_tables(results_optimal, prefix="optimal_mahalanobis")
    generate_html_report(results_optimal, "Optimal Matching with Mahalanobis Distance",
                         "optimal_mahalanobis_report.html")
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    main() 