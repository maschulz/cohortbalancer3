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
import networkx as nx
from matplotlib.lines import Line2D

# Import logging
from cohortbalancer3.utils.logging import get_logger

# Create a logger for this module
logger = get_logger(__name__)

# Import MatchResults for type hints
if TYPE_CHECKING:
    from cohortbalancer3.datatypes import MatchResults

# Set style defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Make sure to add plot_match_groups to the list of exports at the top of the file
__all__ = [
    "plot_balance",
    "plot_love_plot",
    "plot_propensity_distributions",
    "plot_propensity_calibration",
    "plot_treatment_effects",
    "plot_matched_pairs_distance",
    "plot_matching_summary",
    "plot_propensity_comparison",
    "plot_covariate_distributions",
    "plot_matched_pairs_scatter",
    "plot_match_groups"
]


def plot_balance(results: 'MatchResults', max_vars: int = 20, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """Plot covariate balance before and after matching.
    
    Creates a bar chart showing standardized mean differences (SMD) before and after matching
    for covariates. Includes reference lines at 0.1 and 0.2 thresholds. Variables are sorted
    by pre-matching SMD with the largest imbalances at the top.
    
    Args:
        results: MatchResults object containing matching results
        max_vars: Maximum number of variables to plot (default: 20)
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib figure object
    """
    logger.debug(f"Creating balance plot with up to {max_vars} variables")
    
    # Extract balance statistics
    balance_stats = results.balance_statistics
    if balance_stats is None:
        logger.warning("No balance statistics available in MatchResults object")
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No balance statistics available", 
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Focus on SMD columns and get variable names
    smd_cols = ['smd_before', 'smd_after']
    if 'smd_after' not in balance_stats.columns:
        smd_cols = ['smd_before']
        logger.debug("Only pre-matching SMD values found")
    
    # Sort by pre-matching SMD and take top variables
    sorted_stats = balance_stats.sort_values('smd_before', ascending=False)
    
    # Use standard column for variable names
    var_col = 'variable'
    
    # Take top N variables
    top_stats = sorted_stats.head(max_vars)
    logger.debug(f"Plotting balance for {len(top_stats)} variables")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # For multiple series (before and after)
    if len(smd_cols) > 1:
        # Get a color-blind friendly palette from seaborn
        palette = sns.color_palette("colorblind")
        
        # Plot bars
        bar_width = 0.4
        positions1 = np.arange(len(top_stats))
        positions2 = positions1 + bar_width
        
        # Pre-matching bars
        ax.barh(positions1, top_stats['smd_before'], height=bar_width, 
                label='Before matching', color=palette[0], alpha=0.7)
        
        # Post-matching bars
        ax.barh(positions2, top_stats['smd_after'], height=bar_width, 
                label='After matching', color=palette[1], alpha=0.7)
        
        # Set y-tick positions and labels
        ax.set_yticks(positions1 + bar_width/2)
        ax.set_yticklabels(top_stats[var_col])
    else:
        # Single series (only before matching)
        ax.barh(top_stats[var_col], top_stats['smd_before'], color='steelblue', alpha=0.7)
    
    # Add reference lines
    ax.axvline(0.1, color='darkred', linestyle='--', alpha=0.7, 
               label='0.1 threshold')
    ax.axvline(0.2, color='darkred', linestyle=':', alpha=0.7, 
               label='0.2 threshold')
    
    # Set labels and title
    ax.set_xlabel('Standardized Mean Difference')
    ax.set_title('Covariate Balance')
    
    # Add legend if we have both before and after
    if len(smd_cols) > 1:
        ax.legend(loc='best')
    
    # Adjust layout to make room for variable names
    plt.tight_layout()
    logger.debug("Balance plot created successfully")
    
    return fig


def plot_love_plot(results: 'MatchResults', threshold: float = 0.1, figsize: Tuple[int, int] = (10, 12)) -> plt.Figure:
    """Create a Love plot showing standardized mean differences.
    
    Displays a dot plot where each row represents a covariate, with points showing the 
    standardized mean difference before and after matching. Lines connect the before/after
    points for each variable. A vertical reference line indicates the balance threshold.
    Variables are alphabetically sorted.
    
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
    
    Creates overlapping histograms with density curves showing the distribution of 
    propensity scores for treatment (blue) and control (orange) groups. The plot visually
    demonstrates the degree of overlap between the groups' propensity distributions.
    
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
    
    Creates a calibration plot showing how well the predicted propensity scores align with 
    observed treatment rates. Points represent binned propensity scores, with their size 
    proportional to bin count. A diagonal dashed line represents perfect calibration. 
    Well-calibrated models have points close to this line.
    
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
    
    Creates a forest plot displaying estimated treatment effects for different outcomes with
    confidence intervals. Points represent effect estimates, with horizontal error bars for
    confidence intervals. A vertical line at zero represents no effect. Statistically
    significant effects (p < 0.05) are highlighted with a star symbol.
    
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
    
    Creates a histogram showing the distribution of distances between matched treatment-control
    pairs. A vertical dashed red line indicates the median distance. This visualization helps
    assess the quality of matches and identify potential outliers with large distances.
    
    Args:
        results: MatchResults object containing matching results
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract match distances directly
    if not hasattr(results, 'match_distances') or not results.match_distances:
        # If distances are not available or empty, raise an error or return an empty plot
        logger.warning("Match distances are not available or empty in the results. Cannot plot distance distribution.")
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Match distances not available",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig

    match_distances = results.match_distances

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
    
    Creates a grouped bar chart comparing the sample sizes of treatment and control groups
    before and after matching. Bars display counts with numeric labels. A text annotation
    shows the matching ratio (control:treatment) after matching. This visualization provides
    a quick overview of how matching affected sample sizes.
    
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


def plot_propensity_comparison(results: 'MatchResults', bins: int = 30, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """Compare propensity score distributions before and after matching.
    
    Creates a side-by-side comparison of propensity score distributions before matching (left)
    and after matching (right). Each subplot shows overlapping histograms with density curves
    for treatment and control groups. This visualization helps assess how matching improves
    the overlap between propensity distributions.
    
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
    
    # Get treatment column and create masks
    treatment_col = results.config.treatment_col
    original_data = results.original_data
    matched_data = results.matched_data
    
    # Original data masks
    original_treatment_mask = original_data[treatment_col] == 1
    original_treatment_ps = propensity_scores[original_treatment_mask]
    original_control_ps = propensity_scores[~original_treatment_mask]
    
    # Create a mapping from original index to propensity score
    ps_map = dict(zip(original_data.index, propensity_scores))
    
    # Extract propensity scores for matched data using the mapping
    matched_treatment_ps = [ps_map[idx] for idx in matched_data[matched_data[treatment_col] == 1].index]
    matched_control_ps = [ps_map[idx] for idx in matched_data[matched_data[treatment_col] == 0].index]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot before matching (left subplot)
    ax1.hist(original_treatment_ps, bins=bins, alpha=0.5, label='Treatment', 
             density=True, color='blue')
    ax1.hist(original_control_ps, bins=bins, alpha=0.5, label='Control', 
             density=True, color='orange')
    
    # Add density curves for before matching
    if len(original_treatment_ps) > 1:
        sns.kdeplot(original_treatment_ps, ax=ax1, color='blue', label='_nolegend_')
    if len(original_control_ps) > 1:
        sns.kdeplot(original_control_ps, ax=ax1, color='orange', label='_nolegend_')
    
    # Plot after matching (right subplot)
    ax2.hist(matched_treatment_ps, bins=bins, alpha=0.5, label='Treatment', 
             density=True, color='blue')
    ax2.hist(matched_control_ps, bins=bins, alpha=0.5, label='Control', 
             density=True, color='orange')
    
    # Add density curves for after matching
    if len(matched_treatment_ps) > 1:
        sns.kdeplot(matched_treatment_ps, ax=ax2, color='blue', label='_nolegend_')
    if len(matched_control_ps) > 1:
        sns.kdeplot(matched_control_ps, ax=ax2, color='orange', label='_nolegend_')
    
    # Customize subplots
    ax1.set_xlabel('Propensity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Before Matching')
    ax1.legend()
    
    ax2.set_xlabel('Propensity Score')
    ax2.set_title('After Matching')
    ax2.legend()
    
    fig.suptitle('Propensity Score Distributions Before and After Matching', fontsize=14)
    fig.tight_layout()
    return fig


def plot_covariate_distributions(results: 'MatchResults', 
                                max_vars: int = 10, 
                                bins: int = 20, 
                                figsize: Tuple[int, int] = (15, 15)) -> plt.Figure:
    """Plot covariate distributions before and after matching.
    
    Creates a multi-panel plot with rows representing different covariates. Each row contains
    two subplots: distributions before matching (left) and after matching (right). Each subplot
    shows overlapping histograms for treatment and control groups. This visualization helps
    assess how matching improves the balance of individual covariates.
    
    Binary variables (those with only values 0 and 1) are displayed as proportion bar charts
    rather than histograms, making them easier to interpret.
    
    Args:
        results: MatchResults object containing matching results
        max_vars: Maximum number of variables to show (ordered by SMD before matching)
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract balance statistics and data from results
    balance_statistics = results.balance_statistics
    if balance_statistics is None:
        raise ValueError("Balance statistics are not available in the results.")
    
    # Sort by standardized mean difference before matching and get top variables
    sorted_df = balance_statistics.sort_values('smd_before', ascending=False)
    top_vars = sorted_df.head(max_vars)['variable'].tolist()
    
    # Get data
    original_data = results.original_data
    matched_data = results.matched_data
    treatment_col = results.config.treatment_col
    
    # Create figure with subplots
    n_vars = len(top_vars)
    fig, axes = plt.subplots(n_vars, 2, figsize=figsize)
    
    # Adjust for single variable case
    if n_vars == 1:
        axes = np.array([axes])
    
    # Plot each variable
    for i, var in enumerate(top_vars):
        # Skip if variable not in data or is the treatment variable
        if var not in original_data.columns or var == treatment_col:
            continue
        
        # Check if variable is binary (only contains 0 and 1 values)
        is_binary = set(original_data[var].dropna().unique()).issubset({0, 1})
        
        # Original data
        orig_treat = original_data[original_data[treatment_col] == 1][var].dropna()
        orig_ctrl = original_data[original_data[treatment_col] == 0][var].dropna()
        
        # Matched data
        match_treat = matched_data[matched_data[treatment_col] == 1][var].dropna()
        match_ctrl = matched_data[matched_data[treatment_col] == 0][var].dropna()
        
        # Different visualization approaches based on variable type
        if is_binary:
            # For binary variables, show proportion of 1s
            # Before matching
            orig_treat_prop = orig_treat.mean()
            orig_ctrl_prop = orig_ctrl.mean()
            
            # After matching
            match_treat_prop = match_treat.mean()
            match_ctrl_prop = match_ctrl.mean()
            
            # Calculate y-axis range for consistency between before/after plots
            # Get min and max proportions across all groups, with a small buffer
            all_props = [orig_treat_prop, orig_ctrl_prop, match_treat_prop, match_ctrl_prop]
            min_prop = max(0, min(all_props) * 0.9)  # Ensure we don't go below 0
            max_prop = min(1, max(all_props) * 1.1)  # Ensure we don't go above 1
            
            # Add some padding if the range is very narrow
            if max_prop - min_prop < 0.1:
                pad = (0.1 - (max_prop - min_prop)) / 2
                min_prop = max(0, min_prop - pad)
                max_prop = min(1, max_prop + pad)
            
            # Plot proportions as bar charts
            # Before matching plot
            bar_width = 0.35
            axes[i, 0].bar([0.5-bar_width/2], [orig_treat_prop], bar_width, label='Treatment', color='blue', alpha=0.7)
            axes[i, 0].bar([0.5+bar_width/2], [orig_ctrl_prop], bar_width, label='Control', color='orange', alpha=0.7)
            axes[i, 0].set_xticks([0.5])
            axes[i, 0].set_xticklabels(['Proportion "1"'])
            axes[i, 0].set_ylim([min_prop, max_prop])
            axes[i, 0].set_ylabel('Proportion')
            
            # After matching plot
            axes[i, 1].bar([0.5-bar_width/2], [match_treat_prop], bar_width, label='Treatment', color='blue', alpha=0.7)
            axes[i, 1].bar([0.5+bar_width/2], [match_ctrl_prop], bar_width, label='Control', color='orange', alpha=0.7)
            axes[i, 1].set_xticks([0.5])
            axes[i, 1].set_xticklabels(['Proportion "1"'])
            axes[i, 1].set_ylim([min_prop, max_prop])
            axes[i, 1].set_ylabel('Proportion')
            
            # Add text with actual proportion values
            axes[i, 0].text(0.5-bar_width/2, orig_treat_prop, f"{orig_treat_prop:.3f}", 
                          ha='center', va='bottom', fontsize=9)
            axes[i, 0].text(0.5+bar_width/2, orig_ctrl_prop, f"{orig_ctrl_prop:.3f}", 
                          ha='center', va='bottom', fontsize=9)
            axes[i, 1].text(0.5-bar_width/2, match_treat_prop, f"{match_treat_prop:.3f}", 
                          ha='center', va='bottom', fontsize=9)
            axes[i, 1].text(0.5+bar_width/2, match_ctrl_prop, f"{match_ctrl_prop:.3f}", 
                          ha='center', va='bottom', fontsize=9)
            
        else:
            # For continuous variables, use histograms as before
            # Determine bin range for consistency between plots
            all_values = pd.concat([orig_treat, orig_ctrl, match_treat, match_ctrl])
            min_val, max_val = all_values.min(), all_values.max()
            hist_bins = np.linspace(min_val, max_val, bins)
            
            # Plot before matching
            axes[i, 0].hist(orig_treat, bins=hist_bins, alpha=0.5, label='Treatment', 
                           density=True, color='blue')
            axes[i, 0].hist(orig_ctrl, bins=hist_bins, alpha=0.5, label='Control', 
                           density=True, color='orange')
            
            # Plot after matching
            axes[i, 1].hist(match_treat, bins=hist_bins, alpha=0.5, label='Treatment', 
                           density=True, color='blue')
            axes[i, 1].hist(match_ctrl, bins=hist_bins, alpha=0.5, label='Control', 
                           density=True, color='orange')
            
            # Add density curves if enough data points
            if len(orig_treat) > 1:
                sns.kdeplot(orig_treat, ax=axes[i, 0], color='blue', label='_nolegend_')
            if len(orig_ctrl) > 1:
                sns.kdeplot(orig_ctrl, ax=axes[i, 0], color='orange', label='_nolegend_')
            if len(match_treat) > 1:
                sns.kdeplot(match_treat, ax=axes[i, 1], color='blue', label='_nolegend_')
            if len(match_ctrl) > 1:
                sns.kdeplot(match_ctrl, ax=axes[i, 1], color='orange', label='_nolegend_')
        
        # Customize subplots
        axes[i, 0].set_title(f'{var} (Before)')
        axes[i, 1].set_title(f'{var} (After)')
        
        # Add SMD values as annotations
        smd_before = sorted_df[sorted_df['variable'] == var]['smd_before'].values[0]
        smd_after = sorted_df[sorted_df['variable'] == var]['smd_after'].values[0]
        
        axes[i, 0].annotate(f'SMD: {smd_before:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                          va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        axes[i, 1].annotate(f'SMD: {smd_after:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                          va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Add legend (for both binary and continuous cases)
        if i == 0:
            axes[i, 0].legend()
            axes[i, 1].legend()
    
    # Adjust layout
    fig.suptitle('Covariate Distributions Before and After Matching', fontsize=14)
    fig.tight_layout()
    return fig


def plot_matched_pairs_scatter(results: 'MatchResults',
                              x_var: str,
                              y_var: str,
                              figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    """Plot matched pairs as a scatter plot in two dimensions with connecting lines.
    
    Creates a scatter plot where each matched pair is visualized in a two-dimensional space
    defined by the specified covariates. Treatment units are shown in blue, control units in
    orange, with lines connecting matched pairs. This visualization helps assess the spatial
    distribution of matches and identify potentially problematic matches with large distances
    in the selected dimensions.
    
    Args:
        results: MatchResults object containing matching results
        x_var: Name of the covariate to use for x-axis
        y_var: Name of the covariate to use for y-axis
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract data from results
    original_data = results.original_data
    # Use the ID-based pairs list
    match_id_pairs = results.pairs
    treatment_col = results.config.treatment_col # Keep this line if needed, though not directly used now

    # Check if variables exist in data
    if x_var not in original_data.columns or y_var not in original_data.columns:
        raise ValueError(f"Specified variables not found in data: {x_var}, {y_var}")

    # Check if match pairs exist
    if not match_id_pairs:
        logger.warning("No matched pairs found. Cannot create scatter plot.")
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No matched pairs found",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get data for scatter plot using IDs
    plotted_treat_ids = set()
    plotted_control_ids = set()

    for t_id, c_id in match_id_pairs:
        # Check if IDs exist in the original data index
        if t_id not in original_data.index or c_id not in original_data.index:
            logger.warning(f"Skipping pair ({t_id}, {c_id}): ID not found in original data.")
            continue

        # Get coordinates using IDs
        t_x = original_data.loc[t_id, x_var]
        t_y = original_data.loc[t_id, y_var]
        c_x = original_data.loc[c_id, x_var]
        c_y = original_data.loc[c_id, y_var]

        # Plot treatment unit (only once per ID)
        if t_id not in plotted_treat_ids:
            ax.scatter(t_x, t_y, color='blue', s=50, alpha=0.7, zorder=2)
            plotted_treat_ids.add(t_id)

        # Plot control unit (only once per ID if no replacement, else plot all instances implicitly)
        # Note: This simplification assumes that if a control is reused,
        # plotting it multiple times at the same coordinates is acceptable.
        # If unique plotting per *pair* instance is needed, more complex tracking is required.
        if not results.config.replace:
             if c_id not in plotted_control_ids:
                ax.scatter(c_x, c_y, color='orange', s=50, alpha=0.7, zorder=2)
                plotted_control_ids.add(c_id)
        else:
            # With replacement, plot control points for every pair they appear in
            ax.scatter(c_x, c_y, color='orange', s=50, alpha=0.7, zorder=2)


        # Draw line connecting the pair
        ax.plot([t_x, c_x], [t_y, c_y], 'k-', alpha=0.3, zorder=1)

    # Add legend elements manually for clarity
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Treatment',
               markerfacecolor='blue', markersize=10, alpha=0.7),
        Line2D([0], [0], marker='o', color='w', label='Control',
               markerfacecolor='orange', markersize=10, alpha=0.7),
        Line2D([0], [0], color='k', lw=1, label='Matched Pair', alpha=0.3)
    ]

    # Customize plot
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f'Matched Pairs in {x_var} vs {y_var} Space')
    ax.legend(handles=legend_elements)

    fig.tight_layout()
    return fig


def plot_match_groups(results: 'MatchResults', figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Visualize the matching groups structure, particularly useful for many-to-one matching.
    
    This plot shows how multiple control units are connected to each treatment unit in
    many-to-one matching scenarios. It represents the network structure of the matches,
    with treatment units on the left and control units on the right.
    
    Args:
        results: MatchResults object containing the match groups
        figsize: Size of the figure (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get match groups
    match_groups = results.match_groups
    
    if not match_groups:
        ax.text(0.5, 0.5, "No matches found", 
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Extract treatment and control counts
    treatment_ids = list(match_groups.keys())
    n_treatment = len(treatment_ids)
    
    # Get all unique control IDs
    all_control_ids = set()
    for control_ids in match_groups.values():
        all_control_ids.update(control_ids)
    n_control = len(all_control_ids)
    
    # Create a mapping of control IDs to positions
    control_id_to_pos = {c_id: i for i, c_id in enumerate(sorted(all_control_ids))}
    
    # Create a bipartite graph
    G = nx.DiGraph()
    
    # Add treatment nodes (left side)
    for i, t_id in enumerate(treatment_ids):
        G.add_node(f"T{i}", bipartite=0, id=t_id, pos=(0, i))
    
    # Add control nodes (right side)
    for c_id, pos in control_id_to_pos.items():
        G.add_node(f"C{pos}", bipartite=1, id=c_id, pos=(1, pos))
    
    # Add edges between treatment and control nodes
    for i, (t_id, c_ids) in enumerate(match_groups.items()):
        for c_id in c_ids:
            c_pos = control_id_to_pos[c_id]
            G.add_edge(f"T{i}", f"C{c_pos}")
    
    # Get node positions
    # Scale the y positions to fit the plot
    t_scale = max(1, (n_treatment - 1)) 
    c_scale = max(1, (n_control - 1))
    
    pos = {}
    for node in G.nodes():
        if node.startswith('T'):
            idx = int(node[1:])
            pos[node] = (0, (n_treatment - idx - 1) / max(1, t_scale))
        else:  # Control nodes
            idx = int(node[1:])
            pos[node] = (1, (n_control - idx - 1) / max(1, c_scale))
    
    # Get treatment and control counts from matched data
    treatment_col = results.config.treatment_col
    matched_data = results.matched_data
    n_matched_treat = (matched_data[treatment_col] == 1).sum()
    n_matched_control = (matched_data[treatment_col] == 0).sum()
    
    # Draw the graph
    treatment_nodes = [n for n in G.nodes() if n.startswith('T')]
    control_nodes = [n for n in G.nodes() if n.startswith('C')]
    
    # Draw treatment nodes (left side)
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=treatment_nodes, 
                           node_color='tab:red',
                           node_size=300, 
                           alpha=0.8,
                           ax=ax)
    
    # Draw control nodes (right side)
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=control_nodes, 
                           node_color='tab:blue',
                           node_size=300, 
                           alpha=0.8,
                           ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)
    
    # Add labels (show IDs)
    labels = {}
    for node in G.nodes():
        if node.startswith('T'):
            i = int(node[1:])
            labels[node] = f"T{i}"
        else:
            i = int(node[1:])
            labels[node] = f"C{i}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Set plot title and axis labels
    plt.title(f"Match Groups Structure\n"
              f"{n_matched_treat} Treatment Units, {n_matched_control} Control Units\n"
              f"Ratio = {n_matched_control/max(1, n_matched_treat):.2f}:1", 
              fontsize=14)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', markersize=10, label='Treatment'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=10, label='Control')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    # Remove axis ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.set_xlim(-0.1, 1.1)
    plt.tight_layout()
    
    return fig 