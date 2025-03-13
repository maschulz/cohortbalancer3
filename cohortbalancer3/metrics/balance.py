"""
Balance assessment metrics for CohortBalancer2.

This module provides functions for assessing balance between treatment and control groups
before and after matching, including standardized mean differences, variance ratios, and
other balance metrics.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from cohortbalancer3.validation import validate_data, validate_numeric_columns, validate_treatment_column


def standardized_mean_difference(
    data: pd.DataFrame,
    var_name: str,
    treatment_col: str,
    matched_indices: Optional[pd.Index] = None
) -> float:
    """Calculate standardized mean difference for a single variable.
    
    Args:
        data: DataFrame containing the data
        var_name: Name of the variable
        treatment_col: Name of the treatment indicator column
        matched_indices: Optional index for subsetting matched data
        
    Returns:
        Standardized mean difference
    """
    # Subset data if matched_indices provided
    if matched_indices is not None:
        data = data.loc[matched_indices]

    # Get treatment and control groups
    treat_vals = data.loc[data[treatment_col] == 1, var_name]
    control_vals = data.loc[data[treatment_col] == 0, var_name]

    # Calculate means
    treat_mean = treat_vals.mean()
    control_mean = control_vals.mean()

    # Calculate standard deviations
    treat_var = treat_vals.var(ddof=1)  # Use N-1 for sample variance
    control_var = control_vals.var(ddof=1)

    # Calculate pooled standard deviation
    # Formula: sqrt((s1^2 + s2^2) / 2)
    pooled_std = np.sqrt((treat_var + control_var) / 2)

    # Handle zero standard deviation
    if pooled_std == 0:
        # If means are equal, SMD is 0
        if treat_mean == control_mean:
            return 0.0
        # If means differ but std is 0, return a large value
        else:
            return np.inf if treat_mean > control_mean else -np.inf

    # Calculate standardized mean difference
    smd = (treat_mean - control_mean) / pooled_std

    return abs(smd)  # Return absolute value


def variance_ratio(
    data: pd.DataFrame,
    var_name: str,
    treatment_col: str,
    matched_indices: Optional[pd.Index] = None
) -> float:
    """Calculate variance ratio for a single variable.
    
    Args:
        data: DataFrame containing the data
        var_name: Name of the variable
        treatment_col: Name of the treatment indicator column
        matched_indices: Optional index for subsetting matched data
        
    Returns:
        Variance ratio
    """
    # Subset data if matched_indices provided
    if matched_indices is not None:
        data = data.loc[matched_indices]

    # Get treatment and control groups
    treat_vals = data.loc[data[treatment_col] == 1, var_name]
    control_vals = data.loc[data[treatment_col] == 0, var_name]

    # Calculate variances
    treat_var = treat_vals.var(ddof=1)  # Use N-1 for sample variance
    control_var = control_vals.var(ddof=1)

    # Handle zero variance
    if treat_var == 0 and control_var == 0:
        return 1.0  # Equal variances (both zero)
    elif treat_var == 0 or control_var == 0:
        return np.inf  # One group has zero variance

    # Calculate variance ratio (larger variance in numerator)
    if treat_var >= control_var:
        return treat_var / control_var
    else:
        return control_var / treat_var


def calculate_balance_stats(
    data: pd.DataFrame,
    matched_data: pd.DataFrame,
    covariates: List[str],
    treatment_col: str
) -> pd.DataFrame:
    """Calculate balance statistics before and after matching.
    
    Args:
        data: Original DataFrame containing all units
        matched_data: DataFrame containing matched units
        covariates: List of covariate column names
        treatment_col: Name of the treatment indicator column
        
    Returns:
        DataFrame with balance statistics
    """
    # Validate input data
    validate_data(
        data=data,
        treatment_col=treatment_col,
        covariates=covariates
    )
    
    validate_data(
        data=matched_data,
        treatment_col=treatment_col,
        covariates=covariates
    )
    
    # Initialize results
    results = []

    # Calculate statistics for each covariate
    for cov in covariates:
        # Before matching
        smd_before = standardized_mean_difference(
            data=data,
            var_name=cov,
            treatment_col=treatment_col
        )

        var_ratio_before = variance_ratio(
            data=data,
            var_name=cov,
            treatment_col=treatment_col
        )

        # After matching
        smd_after = standardized_mean_difference(
            data=matched_data,
            var_name=cov,
            treatment_col=treatment_col
        )

        var_ratio_after = variance_ratio(
            data=matched_data,
            var_name=cov,
            treatment_col=treatment_col
        )

        # Get means and sample sizes
        treat_mean_before = data.loc[data[treatment_col] == 1, cov].mean()
        control_mean_before = data.loc[data[treatment_col] == 0, cov].mean()
        treat_mean_after = matched_data.loc[matched_data[treatment_col] == 1, cov].mean()
        control_mean_after = matched_data.loc[matched_data[treatment_col] == 0, cov].mean()

        n_treat_before = (data[treatment_col] == 1).sum()
        n_control_before = (data[treatment_col] == 0).sum()
        n_treat_after = (matched_data[treatment_col] == 1).sum()
        n_control_after = (matched_data[treatment_col] == 0).sum()

        # Run statistical test (t-test for numeric data)
        t_stat_before, p_value_before = stats.ttest_ind(
            data.loc[data[treatment_col] == 1, cov],
            data.loc[data[treatment_col] == 0, cov],
            equal_var=False  # Welch's t-test (doesn't assume equal variances)
        )

        # t-test after matching
        t_stat_after, p_value_after = stats.ttest_ind(
            matched_data.loc[matched_data[treatment_col] == 1, cov],
            matched_data.loc[matched_data[treatment_col] == 0, cov],
            equal_var=False  # Welch's t-test
        )

        # Add to results
        results.append({
            'variable': cov,
            'type': 'numeric',  # All variables are assumed to be numeric
            'treat_mean_before': treat_mean_before,
            'control_mean_before': control_mean_before,
            'treat_mean_after': treat_mean_after,
            'control_mean_after': control_mean_after,
            'treat_n_before': n_treat_before,
            'control_n_before': n_control_before,
            'treat_n_after': n_treat_after,
            'control_n_after': n_control_after,
            'smd_before': smd_before,
            'smd_after': smd_after,
            'var_ratio_before': var_ratio_before,
            'var_ratio_after': var_ratio_after,
            'p_value_before': p_value_before,
            'p_value_after': p_value_after,
            'smd_reduction': smd_before - smd_after,
            'smd_reduction_percent': 100 * (smd_before - smd_after) / smd_before if smd_before != 0 else np.nan
        })

    # Create DataFrame from results
    balance_df = pd.DataFrame(results)

    return balance_df


def calculate_rubin_rules(balance_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate Rubin's rules for assessing balance.
    
    Rubin suggested that for balanced matching:
    1. Standardized mean differences should be < 0.25
    2. Variance ratios should be between 0.5 and 2 (for continuous variables only)
    
    Args:
        balance_df: DataFrame with balance statistics
        
    Returns:
        Dictionary with Rubin's rules results
    """
    df = balance_df.copy()

    # For SMD rule, use all variables (both numeric and categorical)
    n_smd_small = (df['smd_after'] < 0.25).sum()
    pct_smd_small = 100 * n_smd_small / len(df) if len(df) > 0 else np.nan

    # For variance ratio rule, use only numeric variables
    numeric_df = df[df['type'] == 'numeric'].copy()
    n_var_ratio_good = ((numeric_df['var_ratio_after'] >= 0.5) &
                        (numeric_df['var_ratio_after'] <= 2)).sum()
    pct_var_ratio_good = 100 * n_var_ratio_good / len(numeric_df) if len(numeric_df) > 0 else np.nan

    # For combined rule, use only numeric variables
    n_both_good = ((numeric_df['smd_after'] < 0.25) &
                  (numeric_df['var_ratio_after'] >= 0.5) &
                  (numeric_df['var_ratio_after'] <= 2)).sum()
    pct_both_good = 100 * n_both_good / len(numeric_df) if len(numeric_df) > 0 else np.nan

    return {
        'n_variables_total': len(df),
        'n_variables_numeric': len(numeric_df),
        'n_variables_categorical': len(df) - len(numeric_df),
        'n_smd_small': n_smd_small,
        'pct_smd_small': pct_smd_small,
        'n_var_ratio_good': n_var_ratio_good,
        'pct_var_ratio_good': pct_var_ratio_good,
        'n_both_good': n_both_good,
        'pct_both_good': pct_both_good
    }


def calculate_balance_index(balance_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate a balance index to summarize overall balance improvement.
    
    This function calculates a balance index based on the mean and maximum
    standardized mean differences before and after matching.
    
    Args:
        balance_df: DataFrame with balance statistics
        
    Returns:
        Dictionary with balance indices
    """
    var_df = balance_df.copy()

    # Calculate mean SMD before and after
    mean_smd_before = var_df['smd_before'].mean()
    mean_smd_after = var_df['smd_after'].mean()

    # Calculate maximum SMD before and after
    max_smd_before = var_df['smd_before'].max()
    max_smd_after = var_df['smd_after'].max()

    # Balance improvement ratio (1 means perfect balance, >1 means improvement)
    mean_balance_ratio = mean_smd_before / mean_smd_after if mean_smd_after > 0 else np.inf
    max_balance_ratio = max_smd_before / max_smd_after if max_smd_after > 0 else np.inf

    # Percent of variables with improved balance
    n_improved = (var_df['smd_after'] < var_df['smd_before']).sum()
    pct_improved = 100 * n_improved / len(var_df) if len(var_df) > 0 else np.nan

    # Calculate a composite balance index (0-100)
    # This is a weighted average of mean SMD reduction and percent improved
    mean_smd_reduction_pct = 100 * (mean_smd_before - mean_smd_after) / mean_smd_before if mean_smd_before > 0 else 0
    balance_index = (0.7 * mean_smd_reduction_pct + 0.3 * pct_improved) if not np.isnan(pct_improved) else mean_smd_reduction_pct

    # Clip balance index to 0-100
    balance_index = max(0, min(100, balance_index))

    return {
        'mean_smd_before': mean_smd_before,
        'mean_smd_after': mean_smd_after,
        'max_smd_before': max_smd_before,
        'max_smd_after': max_smd_after,
        'mean_balance_ratio': mean_balance_ratio,
        'max_balance_ratio': max_balance_ratio,
        'n_variables_improved': n_improved,
        'pct_variables_improved': pct_improved,
        'balance_index': balance_index
    }


def calculate_overall_balance(
    balance_df: pd.DataFrame,
) -> Dict[str, float]:
    """Calculate overall balance metrics from a balance statistics DataFrame.
    
    This function computes summary statistics from a balance DataFrame to get
    an overall assessment of balance across all variables.
    
    Args:
        balance_df: DataFrame with balance statistics from calculate_balance_stats
        
    Returns:
        Dictionary with overall balance metrics:
        - mean_smd_before: Average SMD before matching
        - mean_smd_after: Average SMD after matching
        - max_smd_before: Maximum SMD before matching
        - max_smd_after: Maximum SMD after matching
        - prop_balanced_before: Proportion of covariates with SMD < 0.1 before matching
        - prop_balanced_after: Proportion of covariates with SMD < 0.1 after matching
        - mean_reduction: Average reduction in SMD
        - mean_reduction_percent: Average percent reduction in SMD
    """
    df = balance_df.copy()

    # Calculate overall metrics
    results = {
        'mean_smd_before': df['smd_before'].mean(),
        'mean_smd_after': df['smd_after'].mean(),
        'max_smd_before': df['smd_before'].max(),
        'max_smd_after': df['smd_after'].max(),
        'prop_balanced_before': (df['smd_before'] < 0.1).mean(),
        'prop_balanced_after': (df['smd_after'] < 0.1).mean(),
        'mean_reduction': df['smd_reduction'].mean(),
        'mean_reduction_percent': df['smd_reduction_percent'].mean()
    }

    return results
