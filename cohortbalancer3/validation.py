"""
Data validation utilities for CohortBalancer3.

This module provides centralized data validation functions to ensure
input data meets the requirements for matching algorithms.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd


def validate_data(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: List[str] = None,
    outcomes: Optional[List[str]] = None,
    propensity_col: Optional[str] = None,
    exact_match_cols: Optional[List[str]] = None,
    require_both_groups: bool = True
) -> None:
    """Validate input data for matching.
    
    Performs comprehensive validation on the input data to ensure it meets
    all requirements for matching algorithms:
    - All required columns exist
    - Treatment column contains only binary values (0/1)
    - All columns contain numeric data
    - No missing values in required columns
    
    Args:
        data: DataFrame containing the data
        treatment_col: Name of the treatment indicator column
        covariates: List of covariate column names
        outcomes: Optional list of outcome column names
        propensity_col: Optional name of existing propensity score column
        exact_match_cols: Optional list of columns to match exactly on
        require_both_groups: Whether to require both treatment and control groups
        
    Raises:
        ValueError: If any validation check fails
    """
    # Check if DataFrame is empty
    if data.empty:
        raise ValueError("Input data is empty")
    
    # Collect all columns that need validation
    required_cols = [treatment_col]
    if covariates is not None:
        required_cols.extend(covariates)
    if outcomes is not None:
        required_cols.extend(outcomes)
    if propensity_col is not None:
        required_cols.append(propensity_col)
    if exact_match_cols is not None:
        required_cols.extend(exact_match_cols)
    
    # Remove duplicates while preserving order
    required_cols = list(dict.fromkeys(required_cols))
    
    # Check that all required columns exist
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    # Check treatment column
    validate_treatment_column(data, treatment_col, require_both_groups)
    
    # Check that all columns are numeric
    validate_numeric_columns(data, required_cols)
    
    # Check for missing values
    validate_no_missing_values(data, required_cols)
    
    # If propensity_col is provided, validate it's between 0 and 1
    if propensity_col is not None:
        validate_propensity_scores(data, propensity_col)


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
    """Validate configuration settings for the Matcher.
    
    Args:
        config: MatcherConfig object
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate match_method
    valid_match_methods = {"greedy", "optimal"}
    if config.match_method not in valid_match_methods:
        raise ValueError(
            f"Invalid match_method '{config.match_method}'. "
            f"Must be one of: {', '.join(valid_match_methods)}"
        )
    
    # Validate distance_method
    valid_distance_methods = {"euclidean", "mahalanobis", "propensity", "logit"}
    if config.distance_method not in valid_distance_methods:
        raise ValueError(
            f"Invalid distance_method '{config.distance_method}'. "
            f"Must be one of: {', '.join(valid_distance_methods)}"
        )
    
    # Validate propensity estimation settings
    if config.estimate_propensity:
        valid_propensity_models = {"logistic", "random_forest", "xgboost", "custom"}
        if config.propensity_model not in valid_propensity_models:
            raise ValueError(
                f"Invalid propensity_model '{config.propensity_model}'. "
                f"Must be one of: {', '.join(valid_propensity_models)}"
            )
    
    # Validate matching ratio
    if config.ratio <= 0:
        raise ValueError(f"Matching ratio must be positive, got {config.ratio}")
    
    # Validate caliper if provided
    if config.caliper is not None:
        if isinstance(config.caliper, str):
            if config.caliper.lower() != 'auto':
                raise ValueError(f"String caliper value must be 'auto', got {config.caliper}")
        elif isinstance(config.caliper, (int, float)):
            if config.caliper <= 0:
                raise ValueError(f"Numeric caliper must be positive, got {config.caliper}")
        else:
            raise ValueError(f"Caliper must be a positive number, 'auto', or None, got {type(config.caliper)}")
    
    # Validate caliper_scale
    if config.caliper_scale <= 0:
        raise ValueError(f"Caliper scale must be positive, got {config.caliper_scale}")
    
    # Validate effect estimation settings
    if config.outcomes:
        valid_estimands = {"ate", "att", "atc"}
        if config.estimand not in valid_estimands:
            raise ValueError(
                f"Invalid estimand '{config.estimand}'. "
                f"Must be one of: {', '.join(valid_estimands)}"
            )
        
        valid_effect_methods = {"mean_difference", "regression_adjustment"}
        if config.effect_method not in valid_effect_methods:
            raise ValueError(
                f"Invalid effect_method '{config.effect_method}'. "
                f"Must be one of: {', '.join(valid_effect_methods)}"
            )
        
        if config.effect_method == "regression_adjustment" and not config.adjustment_covariates:
            raise ValueError(
                "adjustment_covariates must be provided when effect_method is 'regression_adjustment'"
            )
    
    # Validate confidence level
    if not 0 < config.confidence_level < 1:
        raise ValueError(
            f"confidence_level must be between 0 and 1, got {config.confidence_level}"
        ) 