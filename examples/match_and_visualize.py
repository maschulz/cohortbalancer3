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
    
    # Create dictionary to store image filenames for the HTML report
    image_paths = {}
    
    # 1. Balance plot
    fig_balance = plot_balance(results, max_vars=15, figsize=(12, 8))
    balance_filename = f"{prefix}balance_plot_{timestamp}.png"
    fig_balance.savefig(os.path.join(OUTPUT_DIR, balance_filename), dpi=300)
    plt.close(fig_balance)
    image_paths['balance'] = balance_filename
    
    # REMOVED: Love plot (redundant with balance plot)
    
    # REMOVED: Single propensity density plot (redundant with comparison plot)
    
    # Propensity scores visualization (if available)
    if results.propensity_scores is not None:
        # 2. Propensity calibration plot
        fig_calibration = plot_propensity_calibration(results)
        calibration_filename = f"{prefix}propensity_calibration_{timestamp}.png"
        fig_calibration.savefig(os.path.join(OUTPUT_DIR, calibration_filename), dpi=300)
        plt.close(fig_calibration)
        image_paths['calibration'] = calibration_filename
        
        # 3. Propensity score comparison (before vs after matching)
        fig_ps_comp = plot_propensity_comparison(results, figsize=(12, 6))
        ps_comp_filename = f"{prefix}propensity_comparison_{timestamp}.png"
        fig_ps_comp.savefig(os.path.join(OUTPUT_DIR, ps_comp_filename), dpi=300)
        plt.close(fig_ps_comp)
        print(f"- Created propensity score comparison (before vs. after matching)")
        image_paths['propensity_comparison'] = ps_comp_filename
    
    # 4. Treatment effects
    fig_effects = plot_treatment_effects(results)
    effects_filename = f"{prefix}treatment_effects_{timestamp}.png"
    fig_effects.savefig(os.path.join(OUTPUT_DIR, effects_filename), dpi=300)
    plt.close(fig_effects)
    image_paths['treatment'] = effects_filename
    
    # 5. Matched pairs distance
    fig_distances = plot_matched_pairs_distance(results)
    distances_filename = f"{prefix}matched_distances_{timestamp}.png"
    fig_distances.savefig(os.path.join(OUTPUT_DIR, distances_filename), dpi=300)
    plt.close(fig_distances)
    image_paths['distances'] = distances_filename
    
    # 6. Matching summary
    fig_summary = plot_matching_summary(results)
    summary_filename = f"{prefix}matching_summary_{timestamp}.png"
    fig_summary.savefig(os.path.join(OUTPUT_DIR, summary_filename), dpi=300)
    plt.close(fig_summary)
    image_paths['summary'] = summary_filename
    
    # 7. Covariate distributions before and after matching
    fig_cov_dist = plot_covariate_distributions(results, max_vars=8, figsize=(14, 16))
    cov_dist_filename = f"{prefix}covariate_distributions_{timestamp}.png"
    fig_cov_dist.savefig(os.path.join(OUTPUT_DIR, cov_dist_filename), dpi=300)
    plt.close(fig_cov_dist)
    print(f"- Created covariate distributions before vs. after matching")
    image_paths['covariate_distributions'] = cov_dist_filename
    
    # 8. Matched pairs scatter plot
    # Choose the two covariates with highest standardized mean difference before matching
    if results.balance_statistics is not None:
        sorted_vars = results.balance_statistics.sort_values('smd_before', ascending=False)
        # Get top 2 covariates that are in the data (not treatment or outcome)
        covariates = results.config.covariates
        top_vars = [var for var in sorted_vars['variable'] if var in covariates]
        
        if len(top_vars) >= 2:
            x_var, y_var = top_vars[0], top_vars[1]
            fig_scatter = plot_matched_pairs_scatter(results, x_var=x_var, y_var=y_var, figsize=(10, 10))
            scatter_filename = f"{prefix}matched_pairs_scatter_{timestamp}.png"
            fig_scatter.savefig(os.path.join(OUTPUT_DIR, scatter_filename), dpi=300)
            plt.close(fig_scatter)
            print(f"- Created matched pairs scatter plot for {x_var} vs {y_var}")
            image_paths['matched_pairs_scatter'] = scatter_filename
    
    # Print success message
    print(f"Visualizations saved to {OUTPUT_DIR}")
    
    return image_paths

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

def generate_html_report(results, method_name, image_paths, filename="matching_report.html"):
    """Generate a professional HTML report summarizing the matching results.
    
    Args:
        results: MatchResults object containing matching results
        method_name: Name of the matching method used
        image_paths: Dictionary of image paths from save_visualizations function
        filename: Output filename for the HTML report
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get basic statistics
    n_total = len(results.original_data)
    n_matched = len(results.matched_data)
    n_treat_orig = (results.original_data['treatment'] == 1).sum()
    n_control_orig = (results.original_data['treatment'] == 0).sum()
    n_treat_matched = (results.matched_data['treatment'] == 1).sum()
    n_control_matched = (results.matched_data['treatment'] == 0).sum()
    matching_ratio = n_control_matched / n_treat_matched if n_treat_matched > 0 else "N/A"
    
    # Extract treatment effects
    effect_html = ""
    effect_table_html = ""
    if results.effect_estimates is not None:
        effect_table_html = results.effect_estimates.to_html(
            float_format="%.4f", 
            classes="table table-bordered table-striped table-sm"
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
            classes="table table-bordered table-striped table-sm"
        )
        
        # Compute overall balance metrics
        mean_smd_before = results.balance_statistics['smd_before'].mean()
        mean_smd_after = results.balance_statistics['smd_after'].mean()
        max_smd_before = results.balance_statistics['smd_before'].max()
        max_smd_after = results.balance_statistics['smd_after'].max()
        prop_balanced_before = (results.balance_statistics['smd_before'] < 0.1).mean() * 100
        prop_balanced_after = (results.balance_statistics['smd_after'] < 0.1).mean() * 100
    
    # Get configuration details
    config = results.config
    distance_method = config.distance_method
    caliper = config.caliper
    exact_match_cols = config.exact_match_cols
    ratio = config.ratio
    
    # Create HTML content with academic-style CSS
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Causal Analysis: {method_name} Matching Results</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600&family=Lato:wght@300;400;700&display=swap">
        <style>
            body {{ 
                padding: 40px 0; 
                font-family: 'Crimson Pro', 'Times New Roman', serif;
                line-height: 1.6;
                color: #212529;
                background-color: #fcfcfc;
            }}
            .container {{
                max-width: 1200px;
                background-color: white;
                padding: 2rem 3rem;
                box-shadow: 0 0 15px rgba(0,0,0,0.05);
                border-radius: 5px;
            }}
            h1, h2, h3, h4, h5, h6 {{
                font-family: 'Lato', Arial, sans-serif;
                font-weight: 700;
                color: #2c3e50;
                margin-top: 1.5em;
                margin-bottom: 0.7em;
            }}
            h1 {{ 
                font-size: 2.2rem; 
                text-align: center;
                margin-bottom: 0.3em;
                border-bottom: 1px solid #eee;
                padding-bottom: 15px;
            }}
            h2 {{ 
                font-size: 1.7rem; 
                border-bottom: 1px solid #eee;
                padding-bottom: 8px;
                margin-top: 2.5rem;
            }}
            h3 {{ font-size: 1.4rem; }}
            p {{ margin-bottom: 1rem; }}
            .text-muted {{ 
                font-family: 'Lato', Arial, sans-serif;
                text-align: center; 
                margin-bottom: 2rem;
                font-size: 0.95rem;
            }}
            .img-container {{ 
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 20px; 
                height: 100%;
                background-color: #fafafa;
                border: 1px solid #eee;
                border-radius: 5px;
                padding: 15px;
            }}
            .img-container img {{ 
                max-width: 100%; 
                max-height: 400px;
                object-fit: contain;
            }}
            .figure-col {{
                display: flex;
                flex-direction: column;
                margin-bottom: 30px;
            }}
            .caption {{
                margin-top: 10px;
                font-size: 0.9em;
                color: #555;
                text-align: center;
                padding: 0 10px;
                font-style: italic;
            }}
            .section-description {{
                background-color: #f7fbff;
                border-left: 4px solid #4a89dc;
                padding: 15px;
                margin-bottom: 25px;
                font-size: 0.95rem;
                border-radius: 3px;
            }}
            .table {{
                font-size: 0.9rem;
                margin-top: 1rem;
                margin-bottom: 2rem;
            }}
            .table th {{
                background-color: #f5f7f9;
                border-top: 2px solid #ddd;
            }}
            .stats-card {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 20px;
                box-shadow: 0 0 5px rgba(0,0,0,0.03);
                margin-bottom: 20px;
            }}
            .stats-number {{
                font-size: 2rem;
                font-weight: bold;
                color: #3a7bd5;
                font-family: 'Lato', Arial, sans-serif;
            }}
            .stats-label {{
                font-size: 0.9rem;
                color: #666;
                font-family: 'Lato', Arial, sans-serif;
            }}
            .executive-summary {{
                background-color: #f9f9f9;
                border: 1px solid #eaeaea;
                border-radius: 5px;
                padding: 20px 25px;
                margin: 30px 0;
            }}
            .divider {{
                height: 1px;
                background-color: #eee;
                margin: 40px 0;
            }}
            @media print {{
                body {{ background-color: white; }}
                .container {{
                    box-shadow: none;
                    max-width: 100%;
                    padding: 0;
                }}
                .img-container {{ break-inside: avoid; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Propensity Score Matching Analysis</h1>
            <p class="text-muted">Method: {method_name} | Generated on {timestamp}</p>
            
            <div class="executive-summary">
                <h3 style="margin-top:0">Executive Summary</h3>
                <p>This report presents the results of a matching analysis using the <strong>{method_name}</strong> approach. 
                The matching procedure reduced the original sample of <strong>{n_total}</strong> subjects to <strong>{n_matched}</strong> matched subjects, 
                achieving a matching ratio of <strong>{matching_ratio:.2f}:1</strong> (control:treatment).</p>
                
                <p>The analysis used <strong>{distance_method}</strong> distance for matching
                {f" with a caliper of {caliper}" if caliper != 'auto' else " with an automatically selected caliper"}.
                {f" Exact matching was enforced on: {', '.join(exact_match_cols)}." if exact_match_cols else ""}</p>
                
                <p>After matching, {prop_balanced_after:.1f}% of covariates had standardized mean differences below 0.1 
                (compared to {prop_balanced_before:.1f}% before matching), indicating 
                {"excellent" if prop_balanced_after > 90 else "good" if prop_balanced_after > 75 else "moderate" if prop_balanced_after > 50 else "poor"} 
                overall balance achievement.</p>
            </div>
            
            <h2>1. Sample Characteristics</h2>
            <div class="section-description">
                This section provides an overview of the sample sizes before and after matching,
                illustrating how many treatment and control units were included in the analysis
                and the resulting matching ratio.
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="stats-card text-center">
                                <div class="stats-number">{n_treat_orig}</div>
                                <div class="stats-label">Original Treatment Units</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="stats-card text-center">
                                <div class="stats-number">{n_control_orig}</div>
                                <div class="stats-label">Original Control Units</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="stats-card text-center">
                                <div class="stats-number">{n_treat_matched}</div>
                                <div class="stats-label">Matched Treatment Units</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="stats-card text-center">
                                <div class="stats-number">{n_control_matched}</div>
                                <div class="stats-label">Matched Control Units</div>
                            </div>
                        </div>
                        <div class="col-md-12">
                            <div class="stats-card text-center">
                                <div class="stats-number">{matching_ratio:.2f}:1</div>
                                <div class="stats-label">Matching Ratio (Control:Treatment)</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{image_paths.get('summary', '')}" alt="Matching Summary">
                    </div>
                    <div class="caption">
                        <strong>Figure 1:</strong> Sample sizes before and after matching. The bars represent counts of treatment and control units
                        in the original and matched datasets.
                    </div>
                </div>
            </div>
            
            <h2>2. Covariate Balance Assessment</h2>
            <div class="section-description">
                Covariate balance is crucial for valid causal inference. This section evaluates how well
                the matching procedure balanced the distribution of covariates between treatment and control groups.
                Standardized mean differences (SMD) below 0.1 indicate good balance.
            </div>
            
            <div class="row">
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{image_paths.get('balance', '')}" alt="Balance Plot">
                    </div>
                    <div class="caption">
                        <strong>Figure 2:</strong> Standardized mean differences (SMD) for covariates before (blue) and after (orange) matching.
                        The red horizontal line indicates the conventional 0.1 threshold for acceptable balance.
                    </div>
                </div>
    """
    
    # Only include second column if we have pairs scatter plot
    if 'matched_pairs_scatter' in image_paths:
        html_content += f"""
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{image_paths.get('matched_pairs_scatter', '')}" alt="Matched Pairs Scatter">
                    </div>
                    <div class="caption">
                        <strong>Figure 3:</strong> Scatter plot showing matched pairs in the space of two covariates with the highest
                        initial imbalance. Blue points represent treatment units, orange points represent control units,
                        and connecting lines show the matches.
                    </div>
                </div>
        """
    else:
        # If only one plot in this row, still use 6 columns to maintain consistent size
        html_content += """
                <div class="col-md-6">
                </div>
        """
    
    html_content += f"""
            </div>
            
            <h3>Balance Metrics Summary</h3>
            <div class="row">
                <div class="col-md-6">
                    <table class="table table-sm table-bordered">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Before Matching</th>
                                <th>After Matching</th>
                                <th>Improvement</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Mean SMD</td>
                                <td>{mean_smd_before:.4f}</td>
                                <td>{mean_smd_after:.4f}</td>
                                <td>{(1 - mean_smd_after/mean_smd_before)*100:.1f}%</td>
                            </tr>
                            <tr>
                                <td>Maximum SMD</td>
                                <td>{max_smd_before:.4f}</td>
                                <td>{max_smd_after:.4f}</td>
                                <td>{(1 - max_smd_after/max_smd_before)*100:.1f}%</td>
                            </tr>
                            <tr>
                                <td>% Covariates with SMD < 0.1</td>
                                <td>{prop_balanced_before:.1f}%</td>
                                <td>{prop_balanced_after:.1f}%</td>
                                <td>{(prop_balanced_after - prop_balanced_before):.1f} pp</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <h3>Top 10 Covariates by Initial Imbalance</h3>
            {balance_html}
            
    """
    
    # Add propensity sections if available
    if 'propensity_comparison' in image_paths:
        html_content += f"""
            <div class="divider"></div>
            <h2>3. Propensity Score Analysis</h2>
            <div class="section-description">
                Propensity scores estimate the probability of treatment assignment based on observed covariates.
                Comparing propensity distributions before and after matching helps evaluate whether the matching
                procedure successfully created comparable groups.
            </div>
            
            <div class="row">
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{image_paths.get('propensity_comparison', '')}" alt="Propensity Comparison">
                    </div>
                    <div class="caption">
                        <strong>Figure 4:</strong> Comparison of propensity score distributions before matching (left) and after matching (right).
                        Better overlap after matching indicates improved overall balance across covariates.
                    </div>
                </div>
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{image_paths.get('calibration', '')}" alt="Propensity Calibration">
                    </div>
                    <div class="caption">
                        <strong>Figure 5:</strong> Calibration plot showing how well propensity scores align with observed treatment rates.
                        Points closer to the diagonal line indicate better calibration. Point size represents bin count.
                    </div>
                </div>
            </div>
        """
    
    # Add matching quality section
    html_content += f"""
            <div class="divider"></div>
            <h2>4. Matching Quality</h2>
            <div class="section-description">
                This section evaluates the quality of matches by examining the distribution of distances
                between matched pairs. Lower distances indicate better quality matches, while outliers
                may suggest problematic matches that could affect treatment effect estimates.
            </div>
            
            <div class="row">
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{image_paths.get('distances', '')}" alt="Match Distances">
                    </div>
                    <div class="caption">
                        <strong>Figure {6 if 'propensity_comparison' in image_paths else 4}:</strong> Distribution of distances between matched pairs.
                        The red vertical line indicates the median distance. Clusters of matches with large distances may warrant further investigation.
                    </div>
                </div>
    """
    
    # Add second column or empty space to maintain consistent size
    if 'matched_pairs_scatter' in image_paths and 'matched_pairs_scatter' not in html_content:
        html_content += f"""
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{image_paths.get('matched_pairs_scatter', '')}" alt="Matched Pairs Scatter">
                    </div>
                    <div class="caption">
                        <strong>Figure {7 if 'propensity_comparison' in image_paths else 5}:</strong> Scatter plot showing matched pairs in the space of two covariates with the highest
                        initial imbalance, illustrating the quality of matches in a multidimensional space.
                    </div>
                </div>
        """
    else:
        # Keep column structure even with only one plot
        html_content += """
                <div class="col-md-6">
                </div>
        """
        
    html_content += """
            </div>
    """
    
    # Treatment effects section
    if 'treatment' in image_paths:
        html_content += f"""
            <div class="divider"></div>
            <h2>5. Treatment Effect Estimates</h2>
            <div class="section-description">
                After achieving balance through matching, this section presents the estimated causal effects
                of treatment on the outcome(s) of interest. These estimates represent the average treatment effect
                on the treated (ATT) under the assumption of no unmeasured confounding.
            </div>
            
            <div class="row">
                <div class="col-md-12 figure-col">
                    <div class="img-container">
                        <img src="{image_paths.get('treatment', '')}" alt="Treatment Effects">
                    </div>
                    <div class="caption">
                        <strong>Figure {8 if 'propensity_comparison' in image_paths else 6}:</strong> Estimated treatment effects with confidence intervals.
                        Effects crossing the zero line (dashed red vertical line) are not statistically significant at α=0.05.
                        Asterisks indicate statistically significant effects.
                    </div>
                </div>
            </div>
            
            <h3>Detailed Treatment Effect Estimates</h3>
            {effect_table_html}
        """
    
    # Add covariate distributions section if available
    if 'covariate_distributions' in image_paths:
        html_content += f"""
            <div class="divider"></div>
            <h2>{6 if 'treatment' in image_paths else 5}. Detailed Covariate Distributions</h2>
            <div class="section-description">
                This section provides a more detailed view of how individual covariates were balanced through matching.
                Each row shows the distribution of a specific covariate before and after matching, allowing for
                visual assessment of balance improvement.
            </div>
            
            <div class="row">
                <div class="col-md-12 figure-col">
                    <div class="img-container">
                        <img src="{image_paths.get('covariate_distributions', '')}" alt="Covariate Distributions">
                    </div>
                    <div class="caption">
                        <strong>Figure {9 if 'propensity_comparison' in image_paths and 'treatment' in image_paths else 
                                   7 if 'propensity_comparison' in image_paths or 'treatment' in image_paths else 
                                   5}:</strong> 
                        Distributions of individual covariates before matching (left) and after matching (right).
                        Each row represents one covariate, with SMD values quantifying the improvement in balance.
                        For binary variables, bar charts show the proportion of '1' values in each group.
                    </div>
                </div>
            </div>
        """
    
    # Add a methodology section
    html_content += f"""
            <div class="divider"></div>
            <h2>Methodology Notes</h2>
            <div class="section-description">
                This section provides details on the matching methodology used in this analysis.
            </div>
            
            <h3>Matching Configuration</h3>
            <table class="table table-sm table-bordered">
                <tr><th>Method</th><td>{method_name}</td></tr>
                <tr><th>Distance Metric</th><td>{distance_method}</td></tr>
                <tr><th>Caliper</th><td>{caliper}</td></tr>
                <tr><th>Matching Ratio</th><td>{ratio}:1</td></tr>
                <tr><th>Exact Matching Columns</th><td>{', '.join(exact_match_cols) if exact_match_cols else 'None'}</td></tr>
            </table>
            
            <h3>Interpretation Guidelines</h3>
            <p>When interpreting the results in this report, consider the following:</p>
            <ul>
                <li><strong>Balance assessment:</strong> Standardized mean differences (SMD) below 0.1 indicate good balance between treatment and control groups. Most covariates should be below this threshold after matching.</li>
                <li><strong>Propensity overlap:</strong> Good overlap in propensity score distributions after matching suggests the matched groups are comparable.</li>
                <li><strong>Treatment effects:</strong> Effects are interpreted as the average treatment effect on the treated (ATT) — the average effect for subjects who received treatment.</li>
                <li><strong>Statistical significance:</strong> Effects with confidence intervals that do not cross zero are statistically significant at the α=0.05 level.</li>
            </ul>
            
            <div class="text-center mt-5">
                <p class="text-muted">
                    <small>Generated using CohortBalancer3 | {timestamp}</small>
                </p>
            </div>
    """
    
    # Close the HTML
    html_content += """
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
    image_paths_greedy = save_visualizations(results_greedy, prefix="greedy_propensity")
    save_tables(results_greedy, prefix="greedy_propensity")
    generate_html_report(results_greedy, "Greedy Matching with Propensity Scores", 
                         image_paths_greedy, "greedy_propensity_report.html")
    
    print("\n2. Running optimal matching with Mahalanobis distance...")
    results_optimal = perform_matching(data, method="optimal", use_propensity=False)
    image_paths_optimal = save_visualizations(results_optimal, prefix="optimal_mahalanobis")
    save_tables(results_optimal, prefix="optimal_mahalanobis")
    generate_html_report(results_optimal, "Optimal Matching with Mahalanobis Distance",
                         image_paths_optimal, "optimal_mahalanobis_report.html")
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    main() 