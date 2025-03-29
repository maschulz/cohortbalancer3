"""
Balance assessment metrics for CohortBalancer2.

This module provides functions for assessing balance between treatment and control groups
before and after matching, including standardized mean differences, variance ratios, and
other balance metrics.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Import logger
from cohortbalancer3.utils.logging import get_logger
from cohortbalancer3.validation import (
    validate_data,
)

# Create a logger for this module
logger = get_logger(__name__)


def standardized_mean_difference(
    data: pd.DataFrame,
    var_name: str,
    treatment_col: str,
    matched_indices: Optional[pd.Index] = None,
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
    matched_indices: Optional[pd.Index] = None,
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
    treatment_col: str,
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
    # Check if matched data has both treatment and control
    matched_has_control = (matched_data[treatment_col] == 0).any()

    if not matched_has_control:
        logger.warning(
            "Matched data has no control units. Balance statistics after matching cannot be calculated."
        )

        # Still validate the original data normally
        validate_data(data=data, treatment_col=treatment_col, covariates=covariates)

        # For matched data, don't require both groups
        validate_data(
            data=matched_data,
            treatment_col=treatment_col,
            covariates=covariates,
            require_both_groups=False,
        )

        # Create a result DataFrame with only "before" statistics
        results = []
        for cov in covariates:
            # Before matching
            smd_before = standardized_mean_difference(
                data=data, var_name=cov, treatment_col=treatment_col
            )

            vr_before = variance_ratio(
                data=data, var_name=cov, treatment_col=treatment_col
            )

            # After matching - set to NaN since we can't calculate
            smd_after = np.nan
            vr_after = np.nan

            results.append(
                {
                    "variable": cov,
                    "smd_before": smd_before,
                    "smd_after": smd_after,
                    "var_ratio_before": vr_before,
                    "var_ratio_after": vr_after,
                }
            )

        return pd.DataFrame(results)

    # Normal case - both treatment and control in matched data
    # Validate input data
    validate_data(data=data, treatment_col=treatment_col, covariates=covariates)

    validate_data(data=matched_data, treatment_col=treatment_col, covariates=covariates)

    # Initialize results
    results = []

    # Calculate statistics for each covariate
    for cov in covariates:
        # Before matching
        smd_before = standardized_mean_difference(
            data=data, var_name=cov, treatment_col=treatment_col
        )

        vr_before = variance_ratio(data=data, var_name=cov, treatment_col=treatment_col)

        # After matching
        smd_after = standardized_mean_difference(
            data=matched_data, var_name=cov, treatment_col=treatment_col
        )

        vr_after = variance_ratio(
            data=matched_data, var_name=cov, treatment_col=treatment_col
        )

        results.append(
            {
                "variable": cov,
                "smd_before": smd_before,
                "smd_after": smd_after,
                "var_ratio_before": vr_before,
                "var_ratio_after": vr_after,
            }
        )

    return pd.DataFrame(results)


def calculate_rubin_rules(balance_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate Rubin's rules for assessing balance.

    Rubin suggested that for balanced matching:
    1. Standardized mean differences should be < 0.25
    2. Variance ratios should be between 0.5 and 2

    Args:
        balance_df: DataFrame with balance statistics

    Returns:
        Dictionary with Rubin's rules results
    """
    logger.debug("Calculating Rubin's rules for balance assessment")

    df = balance_df.copy()

    # Check if we have after-matching statistics
    has_after_stats = not df["smd_after"].isna().all()

    # Filter out rows where smd_after is NaN if necessary
    if has_after_stats:
        valid_df = df[~df["smd_after"].isna()]
    else:
        valid_df = df
        logger.warning(
            "No after-matching statistics available, using before-matching statistics for Rubin's rules"
        )

    # For SMD rule, check what percentage are < 0.25
    if has_after_stats:
        n_smd_small = (valid_df["smd_after"] < 0.25).sum()
    else:
        n_smd_small = (valid_df["smd_before"] < 0.25).sum()

    pct_smd_small = 100 * n_smd_small / len(valid_df) if len(valid_df) > 0 else np.nan

    # For variance ratio rule, check what percentage are between 0.5 and 2
    if has_after_stats:
        n_var_ratio_good = (
            (valid_df["var_ratio_after"] >= 0.5) & (valid_df["var_ratio_after"] <= 2)
        ).sum()
    else:
        n_var_ratio_good = (
            (valid_df["var_ratio_before"] >= 0.5) & (valid_df["var_ratio_before"] <= 2)
        ).sum()

    pct_var_ratio_good = (
        100 * n_var_ratio_good / len(valid_df) if len(valid_df) > 0 else np.nan
    )

    # For combined rule, check what percentage satisfy both criteria
    if has_after_stats:
        n_both_good = (
            (valid_df["smd_after"] < 0.25)
            & (valid_df["var_ratio_after"] >= 0.5)
            & (valid_df["var_ratio_after"] <= 2)
        ).sum()
    else:
        n_both_good = (
            (valid_df["smd_before"] < 0.25)
            & (valid_df["var_ratio_before"] >= 0.5)
            & (valid_df["var_ratio_before"] <= 2)
        ).sum()

    pct_both_good = 100 * n_both_good / len(valid_df) if len(valid_df) > 0 else np.nan

    logger.debug(
        f"Rubin's rules results: {pct_smd_small:.1f}% have SMD < 0.25, {pct_var_ratio_good:.1f}% have variance ratio between 0.5-2"
    )

    return {
        "n_variables_total": len(df),
        "n_smd_small": n_smd_small,
        "pct_smd_small": pct_smd_small,
        "n_var_ratio_good": n_var_ratio_good,
        "pct_var_ratio_good": pct_var_ratio_good,
        "n_both_good": n_both_good,
        "pct_both_good": pct_both_good,
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
    mean_smd_before = var_df["smd_before"].mean()
    mean_smd_after = var_df["smd_after"].mean()

    # Calculate maximum SMD before and after
    max_smd_before = var_df["smd_before"].max()
    max_smd_after = var_df["smd_after"].max()

    # Balance improvement ratio (1 means perfect balance, >1 means improvement)
    mean_balance_ratio = (
        mean_smd_before / mean_smd_after if mean_smd_after > 0 else np.inf
    )
    max_balance_ratio = max_smd_before / max_smd_after if max_smd_after > 0 else np.inf

    # Percent of variables with improved balance
    n_improved = (var_df["smd_after"] < var_df["smd_before"]).sum()
    pct_improved = 100 * n_improved / len(var_df) if len(var_df) > 0 else np.nan

    # Calculate a composite balance index (0-100)
    # This is a weighted average of mean SMD reduction and percent improved
    mean_smd_reduction_pct = (
        100 * (mean_smd_before - mean_smd_after) / mean_smd_before
        if mean_smd_before > 0
        else 0
    )
    balance_index = (
        (0.7 * mean_smd_reduction_pct + 0.3 * pct_improved)
        if not np.isnan(pct_improved)
        else mean_smd_reduction_pct
    )

    # Clip balance index to 0-100
    balance_index = max(0, min(100, balance_index))

    return {
        "mean_smd_before": mean_smd_before,
        "mean_smd_after": mean_smd_after,
        "max_smd_before": max_smd_before,
        "max_smd_after": max_smd_after,
        "mean_balance_ratio": mean_balance_ratio,
        "max_balance_ratio": max_balance_ratio,
        "n_variables_improved": n_improved,
        "pct_variables_improved": pct_improved,
        "balance_index": balance_index,
    }


def calculate_overall_balance(
    balance_df: pd.DataFrame, threshold: float = 0.1
) -> Dict[str, float]:
    """Calculate overall balance metrics from a balance statistics DataFrame.

    This function computes summary statistics from a balance DataFrame to get
    an overall assessment of balance across all variables.

    Args:
        balance_df: DataFrame with balance statistics from calculate_balance_stats
        threshold: Threshold for considering a variable balanced (default: 0.1)

    Returns:
        Dictionary with overall balance metrics:
        - mean_smd_before: Average SMD before matching
        - mean_smd_after: Average SMD after matching
        - max_smd_before: Maximum SMD before matching
        - max_smd_after: Maximum SMD after matching
        - prop_balanced_before: Proportion of covariates with SMD < threshold before matching
        - prop_balanced_after: Proportion of covariates with SMD < threshold after matching
        - mean_reduction: Average reduction in SMD
        - percent_balanced_improved: Percentage of variables where balance improved
    """
    logger.debug("Calculating overall balance metrics")

    df = balance_df.copy()

    # Check if we have after-matching statistics
    has_after_stats = not df["smd_after"].isna().all()

    # Basic statistics that are always available
    results = {
        "mean_smd_before": df["smd_before"].mean(),
        "max_smd_before": df["smd_before"].max(),
        "prop_balanced_before": (df["smd_before"] < threshold).mean(),
    }

    # Add after-matching statistics if available
    if has_after_stats:
        # Filter out NaN values for after-matching statistics
        valid_df = df[~df["smd_after"].isna()]

        # Calculate statistics
        results.update(
            {
                "mean_smd_after": valid_df["smd_after"].mean(),
                "max_smd_after": valid_df["smd_after"].max(),
                "prop_balanced_after": (valid_df["smd_after"] < threshold).mean(),
            }
        )

        # Calculate balance improvement metrics
        valid_df["smd_reduction"] = valid_df["smd_before"] - valid_df["smd_after"]
        valid_df["smd_reduction_percent"] = (
            100 * valid_df["smd_reduction"] / valid_df["smd_before"]
        )
        valid_df.loc[valid_df["smd_before"] == 0, "smd_reduction_percent"] = 0

        results.update(
            {
                "mean_reduction": valid_df["smd_reduction"].mean(),
                "mean_reduction_percent": valid_df["smd_reduction_percent"].mean(),
                "percent_balanced_improved": 100
                * (valid_df["smd_after"] < valid_df["smd_before"]).mean(),
            }
        )
    else:
        # Fill in missing values if we don't have after-matching statistics
        results.update(
            {
                "mean_smd_after": np.nan,
                "max_smd_after": np.nan,
                "prop_balanced_after": np.nan,
                "mean_reduction": np.nan,
                "mean_reduction_percent": np.nan,
                "percent_balanced_improved": np.nan,
            }
        )

    logger.debug(
        f"Overall balance metrics: mean SMD before={results['mean_smd_before']:.3f}, "
        f"mean SMD after={results.get('mean_smd_after', np.nan):.3f}, "
        f"prop balanced before={results['prop_balanced_before']:.3f}, "
        f"prop balanced after={results.get('prop_balanced_after', np.nan):.3f}"
    )

    return results
