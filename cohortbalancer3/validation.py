"""
Data validation utilities for CohortBalancer3.

This module provides centralized data validation functions to ensure
input data meets the requirements for matching algorithms.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from cohortbalancer3.utils.logging import get_logger

# Create a logger for this module
logger = get_logger(__name__)


def validate_dataframe_index(data: pd.DataFrame) -> None:
    """Validate that the DataFrame index consists of supported types.
    
    CohortBalancer3 supports only integer and string index types to ensure
    consistent index handling across all operations, especially matching.
    All index values must be of the same type (either all integers or all strings).
    
    Args:
        data: DataFrame containing the data
        
    Raises:
        TypeError: If index contains unsupported types (not int or str) or
                  if the index contains mixed types
    """
    # Check for empty DataFrame
    if data.empty:
        return
    
    # Get the first index value to determine the expected type
    sample_idx = data.index[0]
    
    # Check if the first index type is supported
    if not isinstance(sample_idx, (int, np.integer, str)):
        raise TypeError(
            f"Index type not supported: {type(sample_idx)}. "
            f"CohortBalancer3 only supports integer and string indices."
        )
    
    # Determine the expected type - either integer or string
    is_integer_idx = isinstance(sample_idx, (int, np.integer))
    expected_type = "integer" if is_integer_idx else "string"
    
    # Check that all indices have the same type
    for i, idx in enumerate(data.index):
        if is_integer_idx and not isinstance(idx, (int, np.integer)):
            raise TypeError(
                f"Mixed index types detected at position {i}. "
                f"Expected all {expected_type} indices, but found {type(idx)}. "
                f"CohortBalancer3 requires homogeneous index types (all integer or all string)."
            )
        elif not is_integer_idx and not isinstance(idx, str):
            raise TypeError(
                f"Mixed index types detected at position {i}. "
                f"Expected all {expected_type} indices, but found {type(idx)}. "
                f"CohortBalancer3 requires homogeneous index types (all integer or all string)."
            )


def validate_data(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: List[str] = None,
    outcomes: Optional[List[str]] = None,
    propensity_col: Optional[str] = None,
    exact_match_cols: Optional[List[str]] = None,
    require_both_groups: bool = True
) -> None:
    """Validate data for matching.
    
    This function performs a series of validations to ensure the input data
    meets the requirements for propensity score matching.
    
    Args:
        data: DataFrame containing the data
        treatment_col: Name of treatment column
        covariates: List of covariate columns (optional)
        outcomes: List of outcome columns (optional)
        propensity_col: Column with propensity scores (optional)
        exact_match_cols: Columns to match exactly on (optional)
        require_both_groups: Whether to require both treatment and control groups
        
    Raises:
        ValueError: If data fails validation checks
    """
    logger.debug(f"Validating data with {len(data)} observations")
    
    # Validate DataFrame index
    validate_dataframe_index(data)
    
    # Validate treatment column
    validate_treatment_column(data, treatment_col, require_both_groups)
    
    # Validate covariates if provided
    if covariates is not None and len(covariates) > 0:
        logger.debug(f"Validating {len(covariates)} covariate columns")
        # Check that all covariates exist in the data
        missing_covariates = [col for col in covariates if col not in data.columns]
        if missing_covariates:
            raise ValueError(f"Covariates not found in data: {missing_covariates}")
        
        # Validate that covariates are numeric and have no missing values
        validate_numeric_columns(data, covariates)
        validate_no_missing_values(data, covariates)
    
    # Validate outcomes if provided
    if outcomes is not None and len(outcomes) > 0:
        logger.debug(f"Validating {len(outcomes)} outcome columns")
        # Check that all outcomes exist in the data
        missing_outcomes = [col for col in outcomes if col not in data.columns]
        if missing_outcomes:
            raise ValueError(f"Outcomes not found in data: {missing_outcomes}")
        
        # Validate that outcomes are numeric and have no missing values
        validate_numeric_columns(data, outcomes)
        validate_no_missing_values(data, outcomes)
    
    # Validate propensity column if provided
    if propensity_col is not None:
        logger.debug(f"Validating propensity score column: {propensity_col}")
        if propensity_col not in data.columns:
            raise ValueError(f"Propensity column '{propensity_col}' not found in data")
        validate_propensity_scores(data, propensity_col)
    
    # Validate exact match columns if provided
    if exact_match_cols is not None and len(exact_match_cols) > 0:
        logger.debug(f"Validating {len(exact_match_cols)} exact match columns")
        # Check that all exact match columns exist in the data
        missing_exact_cols = [col for col in exact_match_cols if col not in data.columns]
        if missing_exact_cols:
            raise ValueError(f"Exact match columns not found in data: {missing_exact_cols}")
        
        # Check for missing values in exact match columns
        validate_no_missing_values(data, exact_match_cols)
    
    logger.info("Data validation successful")


def validate_treatment_column(
    data: pd.DataFrame, 
    treatment_col: str, 
    require_both_groups: bool = True
) -> None:
    """Validate that treatment column contains only binary values (0/1).
    
    Args:
        data: DataFrame containing the data
        treatment_col: Name of the treatment indicator column
        require_both_groups: Whether to require both treatment and control units
        
    Raises:
        ValueError: If treatment column validation fails
    """
    # Check that treatment column contains only binary values
    treatment_values = data[treatment_col].unique()
    if not set(treatment_values).issubset({0, 1}):
        raise ValueError(
            f"Treatment column '{treatment_col}' must contain only binary values (0/1), "
            f"found: {sorted(treatment_values)}"
        )
    
    # Count treatment and control units
    n_treatment = (data[treatment_col] == 1).sum()
    n_control = (data[treatment_col] == 0).sum()
    
    # Always check for at least one treated unit
    if n_treatment == 0:
        raise ValueError(f"No treatment units found in '{treatment_col}' (no 1s)")
    
    # Check for control units only if required
    if require_both_groups and n_control == 0:
        raise ValueError(f"No control units found in '{treatment_col}' (no 0s)")


def validate_numeric_columns(data: pd.DataFrame, columns: List[str]) -> None:
    """Validate that columns contain only numeric data.
    
    Args:
        data: DataFrame containing the data
        columns: List of column names to check
        
    Raises:
        ValueError: If any column contains non-numeric data
    """
    for col in columns:
        if not np.issubdtype(data[col].dtype, np.number):
            raise ValueError(
                f"Column '{col}' must contain only numeric values, "
                f"but has dtype {data[col].dtype}"
            )


def validate_no_missing_values(data: pd.DataFrame, columns: List[str]) -> None:
    """Validate that columns have no missing values.
    
    Args:
        data: DataFrame containing the data
        columns: List of column names to check
        
    Raises:
        ValueError: If any column contains missing values
    """
    for col in columns:
        if data[col].isna().any():
            n_missing = data[col].isna().sum()
            raise ValueError(
                f"Column '{col}' contains {n_missing} missing values. "
                f"Please handle missing values before matching."
            )


def validate_propensity_scores(data: pd.DataFrame, propensity_col: str) -> None:
    """Validate that propensity scores are between 0 and 1.
    
    Args:
        data: DataFrame containing the data
        propensity_col: Name of propensity score column
        
    Raises:
        ValueError: If propensity scores are outside [0, 1]
    """
    p_scores = data[propensity_col]
    if (p_scores < 0).any() or (p_scores > 1).any():
        raise ValueError(
            f"Propensity scores in '{propensity_col}' must be between 0 and 1. "
            f"Found min={p_scores.min()}, max={p_scores.max()}"
        )


def validate_matcher_config(config) -> None:
    """Validate MatcherConfig for required fields and proper values.
    
    Args:
        config: MatcherConfig object to validate
        
    Raises:
        ValueError: If configuration fails validation checks
    """
    logger.debug("Validating matcher configuration")
    
    # Validate required fields
    if not hasattr(config, 'treatment_col') or not config.treatment_col:
        raise ValueError("treatment_col is required in MatcherConfig")
    
    if not hasattr(config, 'covariates') or not config.covariates:
        raise ValueError("covariates list is required in MatcherConfig")
    
    # Validate match method
    valid_match_methods = ['greedy', 'optimal', 'propensity']
    if config.match_method not in valid_match_methods:
        raise ValueError(f"match_method must be one of {valid_match_methods}, got {config.match_method}")
    
    # Validate distance method
    valid_distance_methods = ['euclidean', 'mahalanobis', 'propensity', 'logit']
    if config.distance_method not in valid_distance_methods:
        raise ValueError(f"distance_method must be one of {valid_distance_methods}, got {config.distance_method}")
    
    # Validate ratio
    if config.ratio <= 0:
        raise ValueError(f"ratio must be positive, got {config.ratio}")
    
    # Validate propensity model if estimation is enabled
    if config.estimate_propensity:
        valid_propensity_models = ['logistic', 'random_forest', 'xgboost', 'custom']
        if config.propensity_model not in valid_propensity_models:
            raise ValueError(f"propensity_model must be one of {valid_propensity_models}, got {config.propensity_model}")
        
        # Validate CV folds
        if config.cv_folds <= 1:
            raise ValueError(f"cv_folds must be greater than 1, got {config.cv_folds}")
    
    # Validate estimand if outcomes are specified
    if config.outcomes:
        valid_estimands = ['ate', 'att', 'atc']
        if config.estimand not in valid_estimands:
            raise ValueError(f"estimand must be one of {valid_estimands}, got {config.estimand}")
        
        valid_effect_methods = ['mean_difference', 'regression_adjustment']
        if config.effect_method not in valid_effect_methods:
            raise ValueError(f"effect_method must be one of {valid_effect_methods}, got {config.effect_method}")
    
    logger.info("Matcher configuration validation successful") 