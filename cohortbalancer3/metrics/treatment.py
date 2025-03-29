"""
Treatment effect estimation for CohortBalancer2.

This module provides functions for estimating treatment effects from matched data,
including various estimation methods and confidence interval calculation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Import logger
from cohortbalancer3.utils.logging import get_logger
from cohortbalancer3.validation import validate_data

# Create a logger for this module
logger = get_logger(__name__)


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
    random_state: Optional[int] = None,
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
    # Validate method
    valid_methods = {"mean_difference", "regression_adjustment"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown estimation method: {method}. Must be one of: {', '.join(valid_methods)}"
        )

    # Validate estimand
    valid_estimands = {"ate", "att", "atc"}
    if estimand not in valid_estimands:
        raise ValueError(
            f"Unknown estimand: {estimand}. Must be one of: {', '.join(valid_estimands)}"
        )

    # Validate confidence level
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"Confidence level must be between 0 and 1, got {confidence_level}"
        )

    # Subset data if matched_indices provided
    if matched_indices is not None:
        data = data.loc[matched_indices].copy()

    # Validate input data
    required_cols = [outcome, treatment_col]
    if method == "regression_adjustment" and covariates is not None:
        required_cols.extend(covariates)

    validate_data(
        data=data,
        treatment_col=treatment_col,
        covariates=[] if method != "regression_adjustment" else covariates or [],
        outcomes=[outcome],
    )

    # Continue with the estimation
    if method == "mean_difference":
        result = _estimate_mean_difference(
            data=data, outcome=outcome, treatment_col=treatment_col, estimand=estimand
        )
    elif method == "regression_adjustment":
        if covariates is None:
            raise ValueError(
                "Covariates must be provided for regression adjustment method"
            )

        result = _estimate_regression_adjustment(
            data=data,
            outcome=outcome,
            treatment_col=treatment_col,
            covariates=covariates,
            estimand=estimand,
        )

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
            random_state=random_state,
        )
        result.update(
            {
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "confidence_level": confidence_level,
            }
        )

    # Add sample sizes
    result.update(
        {
            "n_treatment": (data[treatment_col] == 1).sum(),
            "n_control": (data[treatment_col] == 0).sum(),
            "n_total": len(data),
        }
    )

    return result


def _estimate_mean_difference(
    data: pd.DataFrame, outcome: str, treatment_col: str, estimand: str = "ate"
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
        treat_vals,
        control_vals,
        equal_var=False,  # Welch's t-test
    )

    return {
        "effect": effect,
        "treat_mean": treat_mean,
        "control_mean": control_mean,
        "t_statistic": t_stat,
        "p_value": p_value,
        "method": "mean_difference",
        "estimand": estimand,
    }


def _estimate_regression_adjustment(
    data: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    covariates: List[str],
    estimand: str = "ate",
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
        "estimand": estimand,
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
    random_state: Optional[int] = None,
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
        bootstrap_indices = rng.choice(data.index, size=len(data), replace=True)
        bootstrap_data = data.loc[bootstrap_indices].copy()

        # Estimate treatment effect
        if method == "mean_difference":
            result = _estimate_mean_difference(
                data=bootstrap_data,
                outcome=outcome,
                treatment_col=treatment_col,
                estimand=estimand,
            )
        elif method == "regression_adjustment":
            result = _estimate_regression_adjustment(
                data=bootstrap_data,
                outcome=outcome,
                treatment_col=treatment_col,
                covariates=covariates,
                estimand=estimand,
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
    random_state: Optional[int] = None,
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
    logger.debug(
        f"Estimating treatment effects for {len(outcomes)} outcomes using {method} method"
    )

    # Validate method and estimand
    valid_methods = {"mean_difference", "regression_adjustment"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown estimation method: {method}. Must be one of: {', '.join(valid_methods)}"
        )

    valid_estimands = {"ate", "att", "atc"}
    if estimand not in valid_estimands:
        raise ValueError(
            f"Unknown estimand: {estimand}. Must be one of: {', '.join(valid_estimands)}"
        )

    # Validate confidence level
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"Confidence level must be between 0 and 1, got {confidence_level}"
        )

    # Validate that outcomes list is not empty
    if not outcomes:
        raise ValueError("List of outcomes cannot be empty")

    # Subset data if matched_indices provided
    if matched_indices is not None:
        data = data.loc[matched_indices].copy()

    # Check if the matched data has both treatment and control units
    has_control = (data[treatment_col] == 0).any()
    has_treatment = (data[treatment_col] == 1).any()

    if not has_control:
        logger.warning(
            "No control units found in the data. Cannot estimate treatment effects."
        )

        # Create a DataFrame with missing values
        results = []
        for outcome in outcomes:
            results.append(
                {
                    "outcome": outcome,
                    "effect": np.nan,
                    "std_error": np.nan,
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "method": method,
                    "estimand": estimand,
                }
            )

        return pd.DataFrame(results)

    if not has_treatment:
        logger.warning(
            "No treatment units found in the data. Cannot estimate treatment effects."
        )

        # Create a DataFrame with missing values
        results = []
        for outcome in outcomes:
            results.append(
                {
                    "outcome": outcome,
                    "effect": np.nan,
                    "std_error": np.nan,
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "method": method,
                    "estimand": estimand,
                }
            )

        return pd.DataFrame(results)

    # Validate input data - only needed if we have both treatment and control
    validate_data(
        data=data,
        treatment_col=treatment_col,
        covariates=[] if method != "regression_adjustment" else covariates or [],
        outcomes=outcomes,
    )

    # Initialize results
    results = []

    # Estimate treatment effect for each outcome
    for outcome in outcomes:
        try:
            logger.debug(f"Estimating treatment effect for outcome: {outcome}")
            result = estimate_treatment_effect(
                data=data,
                outcome=outcome,
                treatment_col=treatment_col,
                method=method,
                covariates=covariates,
                estimand=estimand,
                bootstrap_iterations=bootstrap_iterations,
                confidence_level=confidence_level,
                random_state=random_state,
            )

            # Add outcome name to the result
            result["outcome"] = outcome
            results.append(result)

            logger.debug(
                f"Estimated effect for {outcome}: {result['effect']:.3f} [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}], p={result['p_value']:.3f}"
            )

        except Exception as e:
            logger.error(f"Error estimating treatment effect for {outcome}: {str(e)}")
            # Add a placeholder with error
            results.append(
                {
                    "outcome": outcome,
                    "effect": np.nan,
                    "std_error": np.nan,
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "method": method,
                    "estimand": estimand,
                    "error": str(e),
                }
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Define expected columns - only select those that exist
    expected_columns = [
        "outcome",
        "effect",
        "std_error",
        "t_statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
        "method",
        "estimand",
        "error",
    ]

    # Only include columns that actually exist in the DataFrame
    column_order = [col for col in expected_columns if col in results_df.columns]

    # Reorder columns for better readability if there are any to reorder
    if column_order:
        results_df = results_df[column_order]

    return results_df
