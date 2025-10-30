"""Utility functions for propensity score matching and treatment effect estimation.

This module provides helper functions shared across different metrics calculations.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logit
import pandas as pd
from scipy.stats import chi2

from cohortbalancer3.utils.logging import get_logger

if TYPE_CHECKING:
    from cohortbalancer3.datatypes import MatcherConfig

# Set up logger
logger = get_logger(__name__)


def get_caliper_for_matching(
    config: "MatcherConfig",
    propensity_scores: np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
    data: pd.DataFrame | None = None,
    treat_mask: np.ndarray | None = None
) -> float | None:
    """Get caliper value for matching based on explicit configuration.

    This function handles all caliper calculation logic, including:
    - Direct numeric values from `config.caliper_value`
    - Automatic calculation when `config.caliper_value` is 'auto'
    - No caliper if `config.caliper_method` or `config.caliper_value` is None

    The calculation for 'auto' depends on `config.caliper_method`.

    Args:
        config: The MatcherConfig object.
        propensity_scores: Propensity scores (required for 'propensity' or 'logit' caliper).
        distance_matrix: Distance matrix (for 'mahalanobis' or 'euclidean' caliper).
        data: The full DataFrame (for covariate-based caliper).
        treat_mask: Boolean mask for treatment units (for covariate-based caliper).

    Returns:
        Caliper value to use for matching, or None.

    Raises:
        ValueError: If 'auto' caliper is requested but required data is not provided,
                   or if the configuration is invalid.
    """
    caliper_method = config.caliper_method
    caliper_value = config.caliper_value

    if caliper_method is None or caliper_value is None:
        return None

    if isinstance(caliper_value, (int, float)):
        return float(caliper_value)

    if isinstance(caliper_value, str) and caliper_value.lower() == "auto":
        # --- Propensity-based Caliper ---
        if caliper_method in ["propensity", "logit"]:
            if propensity_scores is None:
                raise ValueError("Propensity scores are required for 'auto' propensity caliper.")
            
            ps_clipped = np.clip(propensity_scores, 1e-6, 1 - 1e-6)
            logit_ps = logit(ps_clipped)
            logit_ps_sd = np.std(logit_ps)
            
            auto_caliper = config.caliper_scale * logit_ps_sd
            logger.info(f"Auto caliper for '{caliper_method}': {auto_caliper:.4f} "
                        f"({config.caliper_scale} * SD of logit propensity = {logit_ps_sd:.4f})")
            return auto_caliper

        # --- Distance Matrix-based Caliper ---
        elif caliper_method in ["mahalanobis", "euclidean"]:
            # This unified path uses the Chi-squared distribution for both Mahalanobis and Euclidean.
            # It works for both matrix-based and fast_greedy methods.
            if caliper_method == 'euclidean':
                logger.warning("Using a Chi-squared-based caliper for Euclidean distance assumes uncorrelated covariates. "
                             "This is an approximation and may not be optimal if covariates are highly correlated.")

            k = len(config.covariates)
            p_value = config.caliper_scale # Interpret scale as p-value
            
            if not (0 < p_value < 1):
                raise ValueError("For 'auto' Mahalanobis/Euclidean calipers, 'caliper_scale' must be a p-value between 0 and 1.")

            # The threshold is the sqrt of the critical value of the chi2 distribution
            critical_value = chi2.ppf(1 - p_value, df=k)
            auto_caliper = np.sqrt(critical_value)
            
            logger.info(f"Auto caliper for '{caliper_method}': {auto_caliper:.4f} "
                        f"(sqrt of chi2 critical value for p={p_value}, k={k})")
            return auto_caliper

        # --- Covariate-based Caliper ---
        else:
            # Assume caliper_method is a column name
            col_name = caliper_method
            if data is None or col_name not in data.columns:
                raise ValueError(f"Column '{col_name}' for caliper not found in data.")
            if treat_mask is None:
                 raise ValueError(f"Treatment mask is required for covariate-based caliper.")

            # Calculate pooled standard deviation of the covariate
            treat_vals = data.loc[treat_mask, col_name]
            control_vals = data.loc[~treat_mask, col_name]
            pooled_std = np.sqrt((np.var(treat_vals, ddof=1) + np.var(control_vals, ddof=1)) / 2)
            
            if pooled_std == 0:
                 logger.warning(f"Standard deviation of caliper column '{col_name}' is zero. Caliper may not be effective.")
                 return 0.0

            auto_caliper = config.caliper_scale * pooled_std
            logger.info(f"Auto caliper for covariate '{col_name}': {auto_caliper:.4f} "
                        f"({config.caliper_scale} * Pooled SD = {pooled_std:.4f})")
            return auto_caliper

    raise ValueError(f"Invalid caliper_value specification: {caliper_value}. "
                     "Must be a numeric value, 'auto', or None.")
