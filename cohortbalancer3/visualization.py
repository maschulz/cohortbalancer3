"""
Visualization functions for CohortBalancer3.

This module provides functions for visualizing matching results, balance statistics,
propensity score distributions, and treatment effects using a unified MatchResults object.
"""

from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import MatchResults for type hints
if TYPE_CHECKING:
    from cohortbalancer3.datatypes import MatchResults

# Set style defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")


def plot_balance(results: 'MatchResults', max_vars: int = 20, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """Plot covariate balance before and after matching.
    
    Args:
        results: MatchResults object containing matching results
        max_vars: Maximum number of variables to show (ordered by SMD before matching)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract balance statistics from results
    balance_statistics = results.balance_statistics
    if balance_statistics is None:
        raise ValueError("Balance statistics are not available in the results.")
    
    # Sort by standardized mean difference before matching
    sorted_df = balance_statistics.sort_values('smd_before', ascending=False)
    
    # Limit to top N variables
    plot_df = sorted_df.head(max_vars)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    x = np.arange(len(plot_df))
    width = 0.35
    
    ax.bar(x - width/2, plot_df['smd_before'], width, label='Before Matching', alpha=0.7)
    ax.bar(x + width/2, plot_df['smd_after'], width, label='After Matching', alpha=0.7)
    
    # Add reference lines
    ax.axhline(y=0.1, color='r', linestyle='-', alpha=0.3, label='0.1 Threshold')
    ax.axhline(y=0.2, color='orange', linestyle='-', alpha=0.3, label='0.2 Threshold')
    
    # Customize plot
    ax.set_ylabel('Standardized Mean Difference')
    ax.set_title('Covariate Balance Before and After Matching')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['variable'], rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    return fig


def plot_love_plot(results: 'MatchResults', threshold: float = 0.1, figsize: Tuple[int, int] = (10, 12)) -> plt.Figure:
    """Create a Love plot showing standardized mean differences.
    
    Args:
        results: MatchResults object containing matching results
        threshold: Threshold line for acceptable balance
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract balance statistics from results
    balance_statistics = results.balance_statistics
    if balance_statistics is None:
        raise ValueError("Balance statistics are not available in the results.")
    
    # Sort by variable name
    sorted_df = balance_statistics.sort_values('variable')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    y = np.arange(len(sorted_df))
    
    ax.scatter(sorted_df['smd_before'], y, label='Before Matching', 
               marker='o', s=50, alpha=0.7)
    ax.scatter(sorted_df['smd_after'], y, label='After Matching', 
               marker='x', s=50, alpha=0.7)
    
    # Add connecting lines
    for i, (before, after) in enumerate(zip(sorted_df['smd_before'], sorted_df['smd_after'])):
        ax.plot([before, after], [i, i], 'k-', alpha=0.3)
    
    # Add reference line
    ax.axvline(x=threshold, color='r', linestyle='--', alpha=0.5, 
               label=f'{threshold} Threshold')
    
    # Customize plot
    ax.set_xlabel('Standardized Mean Difference')
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_df['variable'])
    ax.set_title('Love Plot: Standardized Mean Differences')
    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_propensity_distributions(results: 'MatchResults', bins: int = 30, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot propensity score distributions for treatment and control groups.
    
    Args:
        results: MatchResults object containing matching results
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract propensity scores and treatment mask from results
    propensity_scores = results.propensity_scores
    if propensity_scores is None:
        raise ValueError("Propensity scores are not available in the results.")
    
    # Get treatment column and create mask
    treatment_col = results.config.treatment_col
    treatment_mask = results.original_data[treatment_col] == 1
    
    # Extract propensity scores for each group
    treatment_ps = propensity_scores[treatment_mask]
    control_ps = propensity_scores[~treatment_mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histograms
    ax.hist(treatment_ps, bins=bins, alpha=0.5, label='Treatment', 
            density=True, color='blue')
    ax.hist(control_ps, bins=bins, alpha=0.5, label='Control', 
            density=True, color='orange')
    
    # Add density curves
    if len(treatment_ps) > 1:
        kde_treatment = sns.kdeplot(treatment_ps, ax=ax, color='blue', label='_nolegend_')
    if len(control_ps) > 1:
        kde_control = sns.kdeplot(control_ps, ax=ax, color='orange', label='_nolegend_')
    
    # Customize plot
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title('Propensity Score Distributions')
    ax.legend()
    
    fig.tight_layout()
    return fig


def plot_propensity_calibration(results: 'MatchResults', bins: int = 10, figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """Plot calibration curve for propensity scores.
    
    Args:
        results: MatchResults object containing matching results
        bins: Number of bins for calibration
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract propensity scores and treatment mask from results
    propensity_scores = results.propensity_scores
    if propensity_scores is None:
        raise ValueError("Propensity scores are not available in the results.")
    
    # Get treatment column and create mask
    treatment_col = results.config.treatment_col
    treatment_mask = results.original_data[treatment_col] == 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bins and calculate mean propensity and observed treatment rates
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    observed_rates = []
    mean_predicted = []
    counts = []
    
    for i in range(bins):
        lower, upper = bin_edges[i], bin_edges[i+1]
        mask = (propensity_scores >= lower) & (propensity_scores < upper)
        
        if np.sum(mask) > 0:
            observed_rate = np.mean(treatment_mask[mask])
            mean_pred = np.mean(propensity_scores[mask])
            
            observed_rates.append(observed_rate)
            mean_predicted.append(mean_pred)
            counts.append(np.sum(mask))
    
    # Plot calibration curve
    ax.scatter(mean_predicted, observed_rates, s=[c/10 for c in counts], alpha=0.7, 
               label='Calibration points')
    
    # Add reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Customize plot
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Observed Treatment Rate')
    ax.set_title('Propensity Score Calibration Plot')
    ax.legend(loc='lower right')
    
    fig.tight_layout()
    return fig


def plot_treatment_effects(results: 'MatchResults', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot treatment effects with confidence intervals.
    
    Args:
        results: MatchResults object containing matching results
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract effect estimates from results
    effect_estimates = results.effect_estimates
    if effect_estimates is None:
        raise ValueError("Treatment effect estimates are not available in the results.")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    outcomes = effect_estimates['outcome'].tolist()
    effects = effect_estimates['effect'].values
    
    # Check if CI is available
    has_ci = 'ci_lower' in effect_estimates.columns and 'ci_upper' in effect_estimates.columns
    
    if has_ci:
        ci_lower = effect_estimates['ci_lower'].values
        ci_upper = effect_estimates['ci_upper'].values
        ci_width = ci_upper - effects
        ci_height = effects - ci_lower
        yerr = np.vstack((ci_height, ci_width))
    else:
        yerr = None
    
    # Plot data
    y = np.arange(len(outcomes))
    ax.errorbar(effects, y, xerr=yerr if has_ci else None, fmt='o', capsize=5)
    
    # Add zero reference line
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    # Highlight significant effects
    if 'p_value' in effect_estimates.columns:
        significant = effect_estimates['p_value'] < 0.05
        if significant.any():
            ax.scatter(effects[significant], y[significant], marker='*', s=100, 
                      color='red', label='p < 0.05', zorder=10)
            ax.legend()
    
    # Customize plot
    ax.set_yticks(y)
    ax.set_yticklabels(outcomes)
    ax.set_xlabel('Treatment Effect')
    ax.set_title('Treatment Effect Estimates with Confidence Intervals')
    
    fig.tight_layout()
    return fig


def plot_matched_pairs_distance(results: 'MatchResults', bins: int = 30, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot histogram of distances between matched pairs.
    
    Args:
        results: MatchResults object containing matching results
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract match distances
    if not hasattr(results, 'match_distances'):
        # For backwards compatibility, compute distances
        distance_matrix = results.distance_matrix
        match_pairs = results.match_pairs
        
        match_distances = []
        for t_pos, c_pos_list in match_pairs.items():
            for c_pos in c_pos_list:
                match_distances.append(distance_matrix[t_pos, c_pos])
    else:
        match_distances = results.match_distances
    
    if not match_distances:
        raise ValueError("Match distances are not available in the results.")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(match_distances, bins=bins, alpha=0.7)
    
    # Add median line
    median_distance = np.median(match_distances)
    ax.axvline(x=median_distance, color='r', linestyle='--', 
               label=f'Median: {median_distance:.4f}')
    
    # Customize plot
    ax.set_xlabel('Distance')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Distances Between Matched Pairs')
    ax.legend()
    
    fig.tight_layout()
    return fig


def plot_matching_summary(results: 'MatchResults', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot summary of matching results.
    
    Args:
        results: MatchResults object containing matching results
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Get matching summary
    treatment_col = results.config.treatment_col
    original_data = results.original_data
    matched_data = results.matched_data
    
    # Calculate counts
    original_counts = {
        'n_treatment_orig': (original_data[treatment_col] == 1).sum(),
        'n_control_orig': (original_data[treatment_col] == 0).sum()
    }
    
    matched_counts = {
        'n_treatment_matched': (matched_data[treatment_col] == 1).sum(),
        'n_control_matched': (matched_data[treatment_col] == 0).sum()
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    labels = ['Treatment', 'Control']
    original = [original_counts['n_treatment_orig'], original_counts['n_control_orig']]
    matched = [matched_counts['n_treatment_matched'], matched_counts['n_control_matched']]
    
    # Plot data
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, original, width, label='Original', alpha=0.7)
    ax.bar(x + width/2, matched, width, label='Matched', alpha=0.7)
    
    # Add counts as text
    for i, count in enumerate(original):
        ax.text(i - width/2, count + 0.05 * max(original), str(count), 
                ha='center', va='bottom')
    
    for i, count in enumerate(matched):
        ax.text(i + width/2, count + 0.05 * max(original), str(count), 
                ha='center', va='bottom')
    
    # Customize plot
    ax.set_ylabel('Count')
    ax.set_title('Sample Sizes Before and After Matching')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add matching ratio
    if matched_counts['n_treatment_matched'] > 0:
        ratio = matched_counts['n_control_matched'] / matched_counts['n_treatment_matched']
        ratio_text = f"Matching Ratio: {ratio:.2f}:1"
        ax.text(0.5, 0.05, ratio_text, ha='center', va='bottom', 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    return fig 