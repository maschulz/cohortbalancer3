"""
Propensity score estimation for CohortBalancer3.

This module provides functions for estimating propensity scores using various models,
as well as utilities for assessing propensity score quality and overlap.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import warnings
import functools

import numpy as np
import pandas as pd
from scipy import stats

# Import validation functions
from cohortbalancer3.validation import validate_data, validate_treatment_column

# Try to import scikit-learn
try:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def suppress_warnings(func):
    """Decorator to suppress warnings in a function."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper


def estimate_propensity_scores(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: List[str],
    model_type: str = "logistic",
    model_params: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    calibration: bool = True,
    calibration_method: str = "isotonic",
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """Estimate propensity scores using a classification model.
    
    Args:
        data: DataFrame containing the data
        treatment_col: Name of the column containing treatment indicators
        covariates: List of column names to use for propensity model
        model_type: Type of model to use ("logistic", "random_forest", "xgboost", "custom")
        model_params: Parameters to pass to the model constructor
        cv: Number of cross-validation folds
        calibration: Whether to calibrate the probability estimates
        calibration_method: Method for calibration ('isotonic' or 'sigmoid')
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with propensity scores, model, and metrics
    """
    # Validate input data
    validate_data(
        data=data,
        treatment_col=treatment_col,
        covariates=covariates
    )
    
    # Validate model type
    valid_model_types = {"logistic", "logisticcv", "random_forest", "xgboost", "custom"}
    if model_type not in valid_model_types:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of: {', '.join(valid_model_types)}")
    
    # Validate calibration method
    if calibration:
        valid_calibration_methods = {"isotonic", "sigmoid"}
        if calibration_method not in valid_calibration_methods:
            raise ValueError(f"Unknown calibration method: {calibration_method}. Must be one of: {', '.join(valid_calibration_methods)}")
    
    # Validate CV parameter
    if cv < 2:
        raise ValueError(f"Number of cross-validation folds must be at least 2, got {cv}")
    
    if not HAS_SKLEARN:
        raise ImportError(
            "Scikit-learn is required for propensity score estimation. "
            "Install it with 'pip install scikit-learn'"
        )

    # Create a copy of model_params to avoid modifying the original
    model_params = model_params.copy() if model_params else {}

    # Extract features and treatment indicators
    X = data[covariates].values
    y = data[treatment_col].values

    # Standardize features for some models
    if model_type in ["logistic"]:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Get the propensity model
    model = get_propensity_model(model_type, model_params, random_state)

    # Use cross-validation to estimate propensity scores without data leakage
    cv_results = estimate_propensity_scores_with_cv(
        X=X,
        y=y,
        model=model,
        cv=cv,
        calibration=calibration,
        calibration_method=calibration_method,
        random_state=random_state
    )

    # Add propensity scores and model to the result
    result = {
        "propensity_scores": cv_results["propensity_scores"],
        "model": cv_results["final_model"],
        "cv_results": cv_results["cv_results"],
        "model_type": model_type,
        "auc": cv_results["auc"],
        "calibration": calibration
    }

    return result


def get_propensity_model(
    model_type: str = "logistic",
    model_params: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None
) -> Any:
    """Create a propensity score model based on the specified type.
    
    Args:
        model_type: Type of model to use ("logistic", "random_forest", "xgboost", "custom")
        model_params: Parameters to pass to the model constructor
        random_state: Random state for reproducibility
        
    Returns:
        A scikit-learn compatible model instance
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "Scikit-learn is required for propensity score estimation. "
            "Install it with 'pip install scikit-learn'"
        )

    model_params = model_params or {}

    # Add random_state to model_params if provided
    if random_state is not None:
        model_params["random_state"] = random_state

    if model_type == "logistic":
        return LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            **model_params
        )
    elif model_type == "logisticcv":
        return LogisticRegressionCV(
            max_iter=1000,
            solver='lbfgs',
            **model_params
        )

    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            **model_params
        )
    elif model_type == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Install it with 'pip install xgboost'")

        return XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss',
            **model_params
        )
    elif model_type == "custom":
        if "model" not in model_params:
            raise ValueError("For custom model type, you must provide a 'model' in model_params")
        return model_params["model"]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def estimate_propensity_scores_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    cv: int = 5,
    calibration: bool = True,
    calibration_method: str = "isotonic",
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """Estimate propensity scores using cross-validation to prevent overfitting.
    
    This function uses K-fold cross-validation to estimate propensity scores
    without data leakage, which can occur when the same data is used for
    both fitting the propensity model and subsequent matching.
    
    Args:
        X: Feature matrix
        y: Treatment indicator
        model: Classification model
        cv: Number of cross-validation folds
        calibration: Whether to calibrate the probability estimates
        calibration_method: Method for calibration ('isotonic' or 'sigmoid')
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with propensity scores, model, and metrics
    """
    # Create cross-validation splitter
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Initialize array for propensity scores
    propensity_scores = np.zeros_like(y, dtype=float)

    # Storage for per-fold metrics
    aucs = []
    fold_models = []

    # Perform cross-validation
    for train_idx, test_idx in cv_splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit the model
        model_clone = clone_model(model)
        model_clone.fit(X_train, y_train)

        # Calibrate if requested
        if calibration:
            model_clone = calibrate_model(
                model=model_clone,
                X=X_train,
                y=y_train,
                method=calibration_method
            )

        # Predict propensity scores for test fold
        propensity_scores[test_idx] = model_clone.predict_proba(X_test)[:, 1]

        # Calculate AUC for this fold
        fold_auc = roc_auc_score(y_test, propensity_scores[test_idx])
        aucs.append(fold_auc)

        # Store the model
        fold_models.append(model_clone)

    # Calculate overall AUC
    overall_auc = roc_auc_score(y, propensity_scores)

    # Cross-validate the model with scikit-learn's cross_validate
    cv_results = cross_validate(
        model, X, y,
        cv=cv_splitter,
        scoring=['roc_auc', 'average_precision'],
        return_train_score=True
    )

    # Fit a final model on all data
    final_model = clone_model(model)
    final_model.fit(X, y)

    # Calibrate final model if requested
    if calibration:
        final_model = calibrate_model(
            model=final_model,
            X=X,
            y=y,
            method=calibration_method
        )

    return {
        "propensity_scores": propensity_scores,
        "final_model": final_model,
        "fold_models": fold_models,
        "cv_results": cv_results,
        "auc": overall_auc,
        "fold_aucs": aucs,
        "mean_fold_auc": np.mean(aucs)
    }


def clone_model(model: Any) -> Any:
    """Clone a scikit-learn model.
    
    This function attempts to clone a model using scikit-learn's clone function.
    If that fails, it attempts to create a new instance with the same parameters.
    
    Args:
        model: Model to clone
        
    Returns:
        Cloned model
    """
    try:
        from sklearn.base import clone
        return clone(model)
    except (ImportError, TypeError):
        # Fallback option: try to create a new instance with the same parameters
        try:
            return model.__class__(**model.get_params())
        except:
            # Last resort: just return the model itself (not ideal)
            warnings.warn("Could not clone model, using original instance")
            return model


def calibrate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    method: str = "isotonic"
) -> Any:
    """Calibrate a model to produce well-calibrated probabilities.
    
    Args:
        model: Fitted model to calibrate
        X: Feature matrix
        y: Target vector
        method: Calibration method ('sigmoid' or 'isotonic')
        
    Returns:
        Calibrated model
    """
    # For models that are already well-calibrated (e.g., logistic regression),
    # calibration might not be necessary
    if hasattr(model, 'predict_proba') and not hasattr(model, 'calibrated_classifiers_'):
        # Import FrozenEstimator here to avoid circular imports
        try:
            from sklearn.frozen import FrozenEstimator
            # Use the new recommended approach with FrozenEstimator
            frozen_model = FrozenEstimator(model)
            calibrated_model = CalibratedClassifierCV(
                estimator=frozen_model,
                method=method
            )
            return calibrated_model.fit(X, y)
        except ImportError:
            # Fall back to the deprecated approach for older scikit-learn versions
            calibrated_model = CalibratedClassifierCV(
                estimator=model,
                method=method,
                cv='prefit'
            )
            return calibrated_model.fit(X, y)
    else:
        return model


def trim_by_propensity(
    data: pd.DataFrame,
    propensity_scores: np.ndarray,
    treatment_col: str,
    method: str = "common_support",
    trim_percent: float = 0.05
) -> pd.DataFrame:
    """Trim data based on propensity scores to improve common support.
    
    Args:
        data: DataFrame containing the data
        propensity_scores: Array of propensity scores for each unit
        treatment_col: Name of the treatment indicator column
        method: Trimming method ('common_support', 'percentile')
        trim_percent: Percentage to trim from the tails (if method = 'percentile')
        
    Returns:
        DataFrame with trimmed data
    """
    # Validate input data 
    validate_treatment_column(data, treatment_col)
    
    # Validate that data and propensity scores have the same length
    if len(data) != len(propensity_scores):
        raise ValueError("Data and propensity scores must have the same length")
    
    # Validate method parameter
    valid_methods = {"common_support", "percentile"}
    if method not in valid_methods:
        raise ValueError(f"Unknown trimming method: {method}. Must be one of: {', '.join(valid_methods)}")
    
    # Validate trim_percent parameter
    if not 0 <= trim_percent < 0.5:
        raise ValueError(f"Trim percentage must be between 0 and 0.5, got {trim_percent}")
    
    # Validate propensity scores
    if np.any(propensity_scores < 0) or np.any(propensity_scores > 1):
        raise ValueError(f"Propensity scores must be between 0 and 1, found min={np.min(propensity_scores)}, max={np.max(propensity_scores)}")
    
    treatment = data[treatment_col].values
    
    # Split by treatment group
    treated_ps = propensity_scores[treatment == 1]
    control_ps = propensity_scores[treatment == 0]

    # Determine the trimming thresholds
    if method == "common_support":
        # Find the common support region
        min_treated, max_treated = np.min(treated_ps), np.max(treated_ps)
        min_control, max_control = np.min(control_ps), np.max(control_ps)

        # Common support is the intersection of the two ranges
        lower_bound = max(min_treated, min_control)
        upper_bound = min(max_treated, max_control)
    
    elif method == "percentile":
        # Trim based on percentiles within each group
        lower_bound_treated = np.percentile(treated_ps, 100 * trim_percent)
        upper_bound_treated = np.percentile(treated_ps, 100 * (1 - trim_percent))
        
        lower_bound_control = np.percentile(control_ps, 100 * trim_percent)
        upper_bound_control = np.percentile(control_ps, 100 * (1 - trim_percent))

        # Combine thresholds from both groups for overall bounds
        lower_bound = min(lower_bound_treated, lower_bound_control)
        upper_bound = max(upper_bound_treated, upper_bound_control)
    
    # Perform the trimming
    keep_mask = (propensity_scores >= lower_bound) & (propensity_scores <= upper_bound)
    trimmed_data = data.iloc[keep_mask].copy()
    
    # Print trimming results
    n_original = len(data)
    n_trimmed = len(trimmed_data)
    n_removed = n_original - n_trimmed
    percent_removed = 100 * n_removed / n_original if n_original > 0 else 0
    
    print(f"Trimming results: {n_removed} units removed ({percent_removed:.1f}%)")
    print(f"  - Lower bound: {lower_bound:.3f}, Upper bound: {upper_bound:.3f}")
    print(f"  - Treatment group: {(treatment == 1).sum()} -> {(trimmed_data[treatment_col] == 1).sum()}")
    print(f"  - Control group: {(treatment == 0).sum()} -> {(trimmed_data[treatment_col] == 0).sum()}")
    
    return trimmed_data


def assess_common_support(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    bins: int = 20
) -> Dict[str, Any]:
    """Assess common support between treatment and control propensity distributions.
    
    Args:
        propensity_scores: Array of propensity scores
        treatment: Binary treatment indicator array
        bins: Number of bins for histogram
        
    Returns:
        Dictionary with common support metrics
    """
    # Input validation
    if len(propensity_scores) != len(treatment):
        raise ValueError("Propensity scores and treatment must have the same length")
    
    if len(propensity_scores) == 0:
        raise ValueError("Propensity scores array is empty")
    
    # Check that treatment is binary
    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(f"Treatment must contain only binary values (0/1), found: {unique_treatment}")
        
    # Check that propensity scores are between 0 and 1
    if np.any(propensity_scores < 0) or np.any(propensity_scores > 1):
        raise ValueError(f"Propensity scores must be between 0 and 1, found min={np.min(propensity_scores)}, max={np.max(propensity_scores)}")
    
    # Check bins parameter
    if bins < 2:
        raise ValueError(f"Number of bins must be at least 2, got {bins}")

    # Split by treatment group
    treated_ps = propensity_scores[treatment == 1]
    control_ps = propensity_scores[treatment == 0]
    
    # Check that we have both treatment and control units
    if len(treated_ps) == 0:
        raise ValueError("No treatment units found (no 1s in treatment array)")
    
    if len(control_ps) == 0:
        raise ValueError("No control units found (no 0s in treatment array)")

    # Calculate common support range
    min_treated, max_treated = np.min(treated_ps), np.max(treated_ps)
    min_control, max_control = np.min(control_ps), np.max(control_ps)

    cs_min = max(min_treated, min_control)
    cs_max = min(max_treated, max_control)

    # Create histograms for visualization
    all_range = (min(min_treated, min_control), max(max_treated, max_control))
    hist_treated, bin_edges = np.histogram(treated_ps, bins=bins, range=all_range, density=True)
    hist_control, _ = np.histogram(control_ps, bins=bins, range=all_range, density=True)

    # Calculate overlap coefficient
    bin_width = (all_range[1] - all_range[0]) / bins
    overlap = np.sum(np.minimum(hist_treated, hist_control) * bin_width)

    return {
        "common_support_min": cs_min,
        "common_support_max": cs_max,
        "overlap_coefficient": overlap,
        "hist_treated": hist_treated,
        "hist_control": hist_control,
        "bin_edges": bin_edges
    }


@suppress_warnings
def assess_propensity_overlap(
    data: pd.DataFrame,
    propensity_col: str,
    treatment_col: str,
    matched_indices: Optional[pd.Index] = None
) -> Dict[str, float]:
    """Assess the overlap of propensity scores between treatment and control groups.
    
    This function computes various metrics to evaluate the quality of propensity score
    overlap, which is crucial for valid causal inference. It can evaluate both the 
    original data and the matched data if matched indices are provided.
    
    Args:
        data: DataFrame containing the data
        propensity_col: Name of the propensity score column
        treatment_col: Name of the treatment indicator column (must be binary)
        matched_indices: Optional indices of matched units to assess post-matching overlap
        
    Returns:
        Dictionary with overlap metrics:
        - ks_statistic: Kolmogorov-Smirnov statistic (smaller is better)
        - ks_pvalue: p-value for the KS test
        - overlap_coefficient: Overlap coefficient between distributions (higher is better)
        - common_support_range: Range of common support as a tuple (min, max)
        - treated_range: Range of treated group propensity scores
        - control_range: Range of control group propensity scores
    """
    # Validate input data
    validate_data(
        data=data,
        treatment_col=treatment_col,
        propensity_col=propensity_col
    )
    
    # Use matched data if indices are provided
    if matched_indices is not None:
        data = data.loc[matched_indices]

    # Extract propensity scores and treatment indicators
    ps = data[propensity_col].values
    treatment = data[treatment_col].values

    # Split by treatment group
    treated_ps = ps[treatment == 1]
    control_ps = ps[treatment == 0]

    # Compute KS test
    ks_statistic, ks_pvalue = stats.ks_2samp(treated_ps, control_ps)

    # Get common support using the existing function
    common_support = assess_common_support(ps, treatment)

    # Calculate ranges
    treated_range = (np.min(treated_ps), np.max(treated_ps))
    control_range = (np.min(control_ps), np.max(control_ps))
    
    # Calculate common support range
    common_support_min = max(treated_range[0], control_range[0])
    common_support_max = min(treated_range[1], control_range[1])
    common_support_range = (common_support_min, common_support_max)
    
    # Calculate proportion of units in common support
    in_common_support = ((ps >= common_support_min) & (ps <= common_support_max)).mean()
    
    # Calculate proportion of treated and control units in common support
    treated_in_cs = ((treated_ps >= common_support_min) & (treated_ps <= common_support_max)).mean()
    control_in_cs = ((control_ps >= common_support_min) & (control_ps <= common_support_max)).mean()

    return {
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "overlap_coefficient": common_support["overlap_coefficient"],
        "common_support_range": common_support_range,
        "treated_range": treated_range,
        "control_range": control_range,
        "prop_in_common_support": in_common_support,
        "prop_treated_in_cs": treated_in_cs,
        "prop_control_in_cs": control_in_cs
    }


@suppress_warnings
def calculate_propensity_quality(
    data: pd.DataFrame,
    propensity_col: str,
    treatment_col: str,
    matched_indices: Optional[pd.Index] = None
) -> Dict[str, float]:
    """Calculate quality metrics for propensity scores.
    
    Args:
        data: DataFrame containing the data
        propensity_col: Name of propensity score column
        treatment_col: Name of treatment indicator column
        matched_indices: Optional indices of matched units
        
    Returns:
        Dictionary of quality metrics
    """
    # Validate input data
    validate_data(
        data=data,
        treatment_col=treatment_col,
        propensity_col=propensity_col
    )
    
    # Extract propensity scores and treatment indicators
    ps = data[propensity_col].values
    treatment = data[treatment_col].values

    # Calculate metrics before matching
    ks_stat_before, ks_pval_before = stats.ks_2samp(
        ps[treatment == 1], ps[treatment == 0]
    )

    mean_diff_before = np.mean(ps[treatment == 1]) - np.mean(ps[treatment == 0])

    # Calculate common support ratio (proportion of units in common support)
    common_support_ratio = assess_common_support(ps, treatment)["overlap_coefficient"]

    # Calculate metrics after matching (if matched_indices provided)
    if matched_indices is not None:
        matched_data = data.loc[matched_indices]
        
        # Validate matched data
        validate_data(
            data=matched_data,
            treatment_col=treatment_col,
            propensity_col=propensity_col
        )
        
        matched_ps = matched_data[propensity_col].values
        matched_treatment = matched_data[treatment_col].values

        ks_stat_after, ks_pval_after = stats.ks_2samp(
            matched_ps[matched_treatment == 1], matched_ps[matched_treatment == 0]
        )

        mean_diff_after = np.mean(matched_ps[matched_treatment == 1]) - np.mean(matched_ps[matched_treatment == 0])
    else:
        ks_stat_after = np.nan
        ks_pval_after = np.nan
        mean_diff_after = np.nan

    # Return metrics
    return {
        "ks_statistic_before": ks_stat_before,
        "ks_pvalue_before": ks_pval_before,
        "mean_diff_before": mean_diff_before,
        "ks_statistic_after": ks_stat_after,
        "ks_pvalue_after": ks_pval_after,
        "mean_diff_after": mean_diff_after,
        "common_support_ratio": common_support_ratio
    }
