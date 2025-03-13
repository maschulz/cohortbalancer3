"""
Treatment effect estimation for CohortBalancer2.

This module provides functions for estimating treatment effects from matched data,
including various estimation methods and confidence interval calculation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def estimate_treatment_effect(
    data: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    matched_indices: Optional[pd.Index] = None,
    method: str = "mean_difference",
    covariates: Optional[List[str]] = None,
    estimand: str = "ate",
    bootstrap_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """Estimate treatment effects from matched data.
    
    Args:
        data: DataFrame containing the data
        outcome: Name of the outcome column
        treatment_col: Name of the treatment indicator column
        matched_indices: Optional indices of matched units
        method: Estimation method ('mean_difference', 'regression_adjustment')
        covariates: List of covariates for regression adjustment method
        estimand: Type of estimand ('ate', 'att', 'atc')
        bootstrap_iterations: Number of bootstrap iterations for CI
        confidence_level: Confidence level for CI
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with treatment effect estimates and confidence intervals
    """
    # Subset data if matched_indices provided
    if matched_indices is not None:
        data = data.loc[matched_indices].copy()

    # Estimate treatment effect using the specified method
    if method == "mean_difference":
        result = _estimate_mean_difference(
            data=data,
            outcome=outcome,
            treatment_col=treatment_col,
            estimand=estimand
        )
    elif method == "regression_adjustment":
        if covariates is None:
            raise ValueError("Covariates must be provided for regression adjustment method")

        result = _estimate_regression_adjustment(
            data=data,
            outcome=outcome,
            treatment_col=treatment_col,
            covariates=covariates,
            estimand=estimand
        )
    else:
        raise ValueError(f"Unknown estimation method: {method}")

    # Calculate confidence intervals using bootstrap
    if bootstrap_iterations > 0:
        ci_lower, ci_upper = _bootstrap_confidence_interval(
            data=data,
            outcome=outcome,
            treatment_col=treatment_col,
            method=method,
            covariates=covariates,
            estimand=estimand,
            bootstrap_iterations=bootstrap_iterations,
            confidence_level=confidence_level,
            random_state=random_state
        )
        result.update({
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": confidence_level
        })

    # Add sample sizes
    result.update({
        "n_treatment": (data[treatment_col] == 1).sum(),
        "n_control": (data[treatment_col] == 0).sum(),
        "n_total": len(data)
    })

    return result


def _estimate_mean_difference(
    data: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    estimand: str = "ate"
) -> Dict[str, float]:
    """Estimate treatment effect using simple mean difference.
    
    Args:
        data: DataFrame containing the data
        outcome: Name of the outcome column
        treatment_col: Name of the treatment indicator column
        estimand: Type of estimand ('ate', 'att', 'atc')
        
    Returns:
        Dictionary with treatment effect estimate
    """
    # Extract treatment and control groups
    treat_vals = data.loc[data[treatment_col] == 1, outcome].dropna()
    control_vals = data.loc[data[treatment_col] == 0, outcome].dropna()

    # Calculate means
    treat_mean = treat_vals.mean()
    control_mean = control_vals.mean()

    # Calculate treatment effect based on estimand
    if estimand == "ate":
        # Average Treatment Effect (ATE)
        effect = treat_mean - control_mean
    elif estimand == "att":
        # Average Treatment Effect on the Treated (ATT)
        # For matched data with 1:1 matching, this equals ATE
        effect = treat_mean - control_mean
    elif estimand == "atc":
        # Average Treatment Effect on the Control (ATC)
        # For matched data with 1:1 matching, this equals ATE
        effect = treat_mean - control_mean
    else:
        raise ValueError(f"Unknown estimand: {estimand}")

    # Run a t-test for statistical significance
    t_stat, p_value = stats.ttest_ind(
        treat_vals, control_vals, equal_var=False  # Welch's t-test
    )

    return {
        "effect": effect,
        "treat_mean": treat_mean,
        "control_mean": control_mean,
        "t_statistic": t_stat,
        "p_value": p_value,
        "method": "mean_difference",
        "estimand": estimand
    }


def _estimate_regression_adjustment(
    data: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    covariates: List[str],
    estimand: str = "ate"
) -> Dict[str, float]:
    """Estimate treatment effect using regression adjustment.
    
    This method uses OLS regression to adjust for remaining imbalance
    in covariates after matching.
    
    Args:
        data: DataFrame containing the data
        outcome: Name of the outcome column
        treatment_col: Name of the treatment indicator column
        covariates: List of covariates for adjustment
        estimand: Type of estimand ('ate', 'att', 'atc')
        
    Returns:
        Dictionary with treatment effect estimate
    """
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        raise ImportError(
            "Statsmodels is required for regression adjustment. "
            "Install it with 'pip install statsmodels'"
        )

    # Create a copy of the data with only necessary columns
    model_data = data[[outcome, treatment_col] + covariates].copy()

    # Drop rows with missing values
    model_data = model_data.dropna()

    # Create formula for regression
    formula = f"{outcome} ~ {treatment_col} + " + " + ".join(covariates)

    # Fit regression model
    model = ols(formula, data=model_data).fit()

    # Extract coefficient for treatment column (this is the treatment effect)
    effect = model.params[treatment_col]
    se = model.bse[treatment_col]
    t_stat = model.tvalues[treatment_col]
    p_value = model.pvalues[treatment_col]

    # Extract model statistics
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj

    return {
        "effect": effect,
        "standard_error": se,
        "t_statistic": t_stat,
        "p_value": p_value,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "method": "regression_adjustment",
        "estimand": estimand
    }


def _bootstrap_confidence_interval(
    data: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    method: str = "mean_difference",
    covariates: Optional[List[str]] = None,
    estimand: str = "ate",
    bootstrap_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """Calculate bootstrap confidence intervals for treatment effect.
    
    Args:
        data: DataFrame containing the data
        outcome: Name of the outcome column
        treatment_col: Name of the treatment indicator column
        method: Estimation method ('mean_difference', 'regression_adjustment')
        covariates: List of covariates for regression adjustment method
        estimand: Type of estimand ('ate', 'att', 'atc')
        bootstrap_iterations: Number of bootstrap iterations
        confidence_level: Confidence level for CI
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Set random state
    rng = np.random.RandomState(random_state)

    # Array to store bootstrap estimates
    bootstrap_estimates = np.zeros(bootstrap_iterations)

    # Perform bootstrap
    for i in range(bootstrap_iterations):
        # Sample with replacement
        bootstrap_indices = rng.choice(
            data.index, size=len(data), replace=True
        )
        bootstrap_data = data.loc[bootstrap_indices].copy()

        # Estimate treatment effect
        if method == "mean_difference":
            result = _estimate_mean_difference(
                data=bootstrap_data,
                outcome=outcome,
                treatment_col=treatment_col,
                estimand=estimand
            )
        elif method == "regression_adjustment":
            result = _estimate_regression_adjustment(
                data=bootstrap_data,
                outcome=outcome,
                treatment_col=treatment_col,
                covariates=covariates,
                estimand=estimand
            )
        else:
            raise ValueError(f"Unknown estimation method: {method}")

        # Store estimate
        bootstrap_estimates[i] = result["effect"]

    # Calculate percentile confidence interval
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
    ci_upper = np.percentile(bootstrap_estimates, upper_percentile)

    return ci_lower, ci_upper


def estimate_multiple_outcomes(
    data: pd.DataFrame,
    outcomes: List[str],
    treatment_col: str,
    matched_indices: Optional[pd.Index] = None,
    method: str = "mean_difference",
    covariates: Optional[List[str]] = None,
    estimand: str = "ate",
    bootstrap_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """Estimate treatment effects for multiple outcomes.
    
    Args:
        data: DataFrame containing the data
        outcomes: List of outcome column names
        treatment_col: Name of the treatment indicator column
        matched_indices: Optional indices of matched units
        method: Estimation method ('mean_difference', 'regression_adjustment')
        covariates: List of covariates for regression adjustment method
        estimand: Type of estimand ('ate', 'att', 'atc')
        bootstrap_iterations: Number of bootstrap iterations for CI
        confidence_level: Confidence level for CI
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with treatment effect estimates for each outcome
    """
    # Subset data if matched_indices provided
    if matched_indices is not None:
        data = data.loc[matched_indices].copy()

    # Initialize results
    results = []

    # Estimate treatment effect for each outcome
    for outcome in outcomes:
        try:
            result = estimate_treatment_effect(
                data=data,
                outcome=outcome,
                treatment_col=treatment_col,
                method=method,
                covariates=covariates,
                estimand=estimand,
                bootstrap_iterations=bootstrap_iterations,
                confidence_level=confidence_level,
                random_state=random_state
            )

            # Add outcome name
            result["outcome"] = outcome

            # Add to results
            results.append(result)
        except Exception as e:
            # Skip outcomes with errors
            print(f"Error estimating treatment effect for {outcome}: {e}")
            continue

    # Create DataFrame from results
    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Reorder columns
    column_order = [
        "outcome", "effect", "ci_lower", "ci_upper", "treat_mean", "control_mean",
        "t_statistic", "p_value", "method", "estimand", "n_treatment", "n_control", "n_total"
    ]
    column_order = [col for col in column_order if col in results_df.columns]

    # Add remaining columns
    column_order.extend([col for col in results_df.columns if col not in column_order])

    return results_df[column_order]
