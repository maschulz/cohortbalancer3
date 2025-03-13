"""
Propensity score utilities for CohortBalancer2.

This module provides functions for estimating propensity scores, evaluating
propensity score models, and assessing common support between treatment and
control groups.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Import scikit-learn components
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Import XGBoost if available
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Define a simple decorator to suppress specific warnings
def suppress_warnings(func):
    """Decorator to suppress warnings during function execution."""
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
    """Trim the dataset based on propensity scores to improve overlap.
    
    Args:
        data: DataFrame containing the data
        propensity_scores: Array of propensity scores
        treatment_col: Name of the column containing treatment indicators
        method: Trimming method ('common_support' or 'percentile')
        trim_percent: Percentage to trim for percentile method or
                     threshold for common support method
        
    Returns:
        Trimmed DataFrame
    """
    # Create a copy of the data
    data_copy = data.copy()

    # Add propensity scores to the data
    data_copy['_ps_temp'] = propensity_scores

    # Get treatment mask
    treatment_mask = data_copy[treatment_col] == 1

    if method == "common_support":
        # Find common support region
        treatment_min = data_copy.loc[treatment_mask, '_ps_temp'].min()
        treatment_max = data_copy.loc[treatment_mask, '_ps_temp'].max()
        control_min = data_copy.loc[~treatment_mask, '_ps_temp'].min()
        control_max = data_copy.loc[~treatment_mask, '_ps_temp'].max()

        # Define the common support region with a buffer
        common_min = max(treatment_min, control_min) - trim_percent
        common_max = min(treatment_max, control_max) + trim_percent

        # Trim data to common support region
        in_support = (data_copy['_ps_temp'] >= common_min) & (data_copy['_ps_temp'] <= common_max)
        trimmed_data = data_copy[in_support].copy()

    elif method == "percentile":
        # Trim extreme percentiles for treatment and control separately
        t_low = np.percentile(data_copy.loc[treatment_mask, '_ps_temp'], trim_percent)
        t_high = np.percentile(data_copy.loc[treatment_mask, '_ps_temp'], 100 - trim_percent)
        c_low = np.percentile(data_copy.loc[~treatment_mask, '_ps_temp'], trim_percent)
        c_high = np.percentile(data_copy.loc[~treatment_mask, '_ps_temp'], 100 - trim_percent)

        # Trim treatment units
        t_in_range = (
            (data_copy['_ps_temp'] >= t_low) &
            (data_copy['_ps_temp'] <= t_high) &
            treatment_mask
        )

        # Trim control units
        c_in_range = (
            (data_copy['_ps_temp'] >= c_low) &
            (data_copy['_ps_temp'] <= c_high) &
            ~treatment_mask
        )

        # Combine masks
        in_range = t_in_range | c_in_range
        trimmed_data = data_copy[in_range].copy()

    else:
        raise ValueError(f"Unknown trimming method: {method}")

    # Remove temporary propensity score column
    if '_ps_temp' in trimmed_data.columns:
        trimmed_data = trimmed_data.drop(columns=['_ps_temp'])

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

    # Split by treatment group
    treated_ps = propensity_scores[treatment == 1]
    control_ps = propensity_scores[treatment == 0]

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
    common_support_range = (common_support["common_support_min"], common_support["common_support_max"])

    return {
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "overlap_coefficient": common_support["overlap_coefficient"],
        "common_support_range": common_support_range,
        "treated_range": treated_range,
        "control_range": control_range
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
