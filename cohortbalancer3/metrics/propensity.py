"""Propensity score estimation for CohortBalancer3.

This module provides functions for estimating propensity scores using various models,
as well as utilities for assessing propensity score quality and overlap.
"""

import functools
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Import logger
from cohortbalancer3.utils.logging import get_logger

# Import validation functions
from cohortbalancer3.validation import validate_data

# Create a logger for this module
logger = get_logger(__name__)

# Try to import scikit-learn
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning(
        "Scikit-learn is not installed. Some propensity score estimation methods will not be available."
    )

# Try to import XGBoost
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning(
        "XGBoost is not installed. XGBoost propensity score estimation will not be available."
    )


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
    covariates: list[str],
    model_type: str = "logistic",
    model_params: dict[str, Any] | None = None,
    cv: int = 5,
    calibration: bool = True,
    calibration_method: str = "isotonic",
    random_state: int | None = None,
) -> dict[str, Any]:
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
    validate_data(data=data, treatment_col=treatment_col, covariates=covariates)

    # Validate model type
    valid_model_types = {"logistic", "logisticcv", "random_forest", "xgboost", "custom"}
    if model_type not in valid_model_types:
        raise ValueError(
            f"Unknown model type: {model_type}. Must be one of: {', '.join(valid_model_types)}"
        )

    # Validate calibration method
    if calibration:
        valid_calibration_methods = {"isotonic", "sigmoid"}
        if calibration_method not in valid_calibration_methods:
            raise ValueError(
                f"Unknown calibration method: {calibration_method}. Must be one of: {', '.join(valid_calibration_methods)}"
            )

    # Validate CV parameter
    if cv < 2:
        raise ValueError(
            f"Number of cross-validation folds must be at least 2, got {cv}"
        )

    if not HAS_SKLEARN:
        raise ImportError(
            "Scikit-learn is required for propensity score estimation. "
            "Install it with 'pip install scikit-learn'"
        )

    logger.info(
        f"Estimating propensity scores using {model_type} model with {cv}-fold cross-validation"
    )
    if calibration:
        logger.info(
            f"Probability calibration enabled using {calibration_method} method"
        )

    # Create a copy of model_params to avoid modifying the original
    model_params = model_params.copy() if model_params else {}

    # Extract features and treatment indicators
    X = data[covariates].values
    y = data[treatment_col].values

    logger.debug(
        f"Treatment prevalence: {np.mean(y):.3f} ({np.sum(y)} out of {len(y)} units)"
    )

    # Standardize features for some models
    if model_type in ["logistic", "logisticcv"]:
        logger.debug("Standardizing features for logistic regression")
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
        random_state=random_state,
    )

    # Add propensity scores and model to the result
    result = {
        "propensity_scores": cv_results["propensity_scores"],
        "model": cv_results["final_model"],
        "cv_results": cv_results["cv_results"],
        "model_type": model_type,
        "auc": cv_results["auc"],
        "calibration": calibration,
    }

    logger.info(f"Propensity score estimation complete. AUC: {cv_results['auc']:.3f}")
    return result


def get_propensity_model(
    model_type: str = "logistic",
    model_params: dict[str, Any] | None = None,
    random_state: int | None = None,
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
        return LogisticRegression(max_iter=1000, solver="lbfgs", **model_params)
    if model_type == "logisticcv":
        return LogisticRegressionCV(max_iter=1000, solver="lbfgs", **model_params)

    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, **model_params)
    if model_type == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Install it with 'pip install xgboost'"
            )

        return XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="logloss",
            **model_params,
        )
    if model_type == "custom":
        if "model" not in model_params:
            raise ValueError(
                "For custom model type, you must provide a 'model' in model_params"
            )
        return model_params["model"]
    raise ValueError(f"Unknown model type: {model_type}")


def estimate_propensity_scores_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    cv: int = 5,
    calibration: bool = True,
    calibration_method: str = "isotonic",
    random_state: int | None = None,
) -> dict[str, Any]:
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

    logger.debug(f"Starting {cv}-fold cross-validation for propensity score estimation")

    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.debug(
            f"Fold {fold_idx + 1}/{cv}: Training on {len(X_train)} samples, testing on {len(X_test)} samples"
        )

        # Fit the model
        model_clone = clone_model(model)

        try:
            if calibration:
                model_clone = calibrate_model(
                    model=model_clone, X=X_train, y=y_train, method=calibration_method
                )
            else:
                model_clone.fit(X_train, y_train)

            # Predict probabilities for test set
            preds = model_clone.predict_proba(X_test)[:, 1]

            # Store propensity scores for this fold
            propensity_scores[test_idx] = preds

            # Calculate AUC for this fold
            fold_auc = roc_auc_score(y_test, preds)
            aucs.append(fold_auc)
            fold_models.append(model_clone)

            logger.debug(f"Fold {fold_idx + 1}/{cv}: AUC = {fold_auc:.3f}")

        except Exception as e:
            logger.error(f"Error in fold {fold_idx + 1}/{cv}: {e!s}")
            raise

    # Train a final model on all data
    logger.debug("Training final model on all data")
    final_model = clone_model(model)

    if calibration:
        final_model = calibrate_model(
            model=final_model, X=X, y=y, method=calibration_method
        )
    else:
        final_model.fit(X, y)

    # Calculate overall AUC
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    logger.info(f"Cross-validation AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    return {
        "propensity_scores": propensity_scores,
        "final_model": final_model,
        "cv_results": {"fold_aucs": aucs, "mean_auc": mean_auc, "std_auc": std_auc},
        "fold_models": fold_models,
        "auc": mean_auc,
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
    logger.debug(f"Cloning model of type {type(model).__name__}")

    try:
        from sklearn.base import clone

        cloned_model = clone(model)
        logger.debug("Model cloned successfully using sklearn.base.clone")
        return cloned_model
    except (ImportError, TypeError) as e:
        logger.warning(f"Could not clone model using sklearn.base.clone: {e!s}")

        # Fallback option: try to create a new instance with the same parameters
        try:
            logger.debug(
                "Attempting to clone by creating new instance with same parameters"
            )
            cloned_model = model.__class__(**model.get_params())
            logger.debug("Model cloned successfully by creating new instance")
            return cloned_model
        except Exception as e2:
            # Last resort: just return the model itself (not ideal)
            logger.warning(f"Could not clone model by creating new instance: {e2!s}")
            logger.warning("Using original model instance (not recommended)")
            return model


def calibrate_model(
    model: Any, X: np.ndarray, y: np.ndarray, method: str = "isotonic"
) -> Any:
    """Calibrate a model to produce well-calibrated probabilities.

    Args:
        model: Model to calibrate
        X: Feature matrix
        y: Target vector
        method: Calibration method ('sigmoid' or 'isotonic')

    Returns:
        Calibrated model

    """
    logger.debug(f"Calibrating model using {method} method")

    # First, ensure the base model is fitted
    try:
        # Create a copy of the model
        base_model = clone_model(model)

        # Fit the base model
        logger.debug("Fitting base model before calibration")
        base_model.fit(X, y)

        # Now create a calibrated model
        logger.debug("Creating calibrated model")
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv=5,  # Use cross-validation for better calibration
        )

        # Fit the calibrated model
        logger.debug("Fitting calibrated model")
        fitted_calibrated_model = calibrated_model.fit(X, y)

        logger.debug("Model calibration complete")
        return fitted_calibrated_model

    except Exception as e:
        logger.error(f"Error during model calibration: {e!s}")
        # If calibration fails, at least return a fitted base model
        logger.warning("Calibration failed, returning non-calibrated model")
        model.fit(X, y)
        return model


def trim_by_propensity(
    data: pd.DataFrame,
    propensity_scores: np.ndarray,
    treatment_col: str,
    method: str = "common_support",
    trim_percent: float = 0.05,
) -> pd.DataFrame:
    """Trim the dataset based on propensity scores to ensure overlap.

    Args:
        data: DataFrame containing the data
        propensity_scores: Array of propensity scores
        treatment_col: Name of the column containing treatment indicators
        method: Trimming method ('common_support' or 'percentile')
        trim_percent: Threshold for trimming (for percentile method)

    Returns:
        Trimmed DataFrame

    """
    # Validate method
    valid_methods = {"common_support", "percentile"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown trimming method: {method}. Must be one of: {', '.join(valid_methods)}"
        )

    # Validate trim_percent
    if not (0 <= trim_percent <= 0.5):
        raise ValueError(f"trim_percent must be between 0 and 0.5, got {trim_percent}")

    # Get treatment indicator
    treatment = data[treatment_col].values

    logger.info(f"Trimming data using {method} method")
    logger.debug(
        f"Original data: {len(data)} observations ({np.sum(treatment)} treated, {len(treatment) - np.sum(treatment)} control)"
    )

    # Create a copy of the data to trim
    data_trimmed = data.copy()

    # Calculate common support region
    treated_mask = treatment == 1
    control_mask = ~treated_mask

    treated_ps = propensity_scores[treated_mask]
    control_ps = propensity_scores[control_mask]

    if method == "common_support":
        min_treated = np.min(treated_ps)
        max_treated = np.max(treated_ps)
        min_control = np.min(control_ps)
        max_control = np.max(control_ps)

        # Common support region
        min_common = max(min_treated, min_control)
        max_common = min(max_treated, max_control)

        logger.debug(f"Common support region: [{min_common:.3f}, {max_common:.3f}]")
        logger.debug(f"Treated range: [{min_treated:.3f}, {max_treated:.3f}]")
        logger.debug(f"Control range: [{min_control:.3f}, {max_control:.3f}]")

        # Keep only observations in the common support region
        in_region = (propensity_scores >= min_common) & (
            propensity_scores <= max_common
        )
        data_trimmed = data_trimmed[in_region]

    elif method == "percentile":
        # Calculate percentile thresholds
        treated_low = np.percentile(treated_ps, trim_percent * 100)
        treated_high = np.percentile(treated_ps, (1 - trim_percent) * 100)
        control_low = np.percentile(control_ps, trim_percent * 100)
        control_high = np.percentile(control_ps, (1 - trim_percent) * 100)

        logger.debug(
            f"Trimming at {trim_percent * 100:.1f}% and {(1 - trim_percent) * 100:.1f}% percentiles"
        )
        logger.debug(f"Treated thresholds: [{treated_low:.3f}, {treated_high:.3f}]")
        logger.debug(f"Control thresholds: [{control_low:.3f}, {control_high:.3f}]")

        # Keep observations within the percentile thresholds
        keep_treated = (
            (propensity_scores >= treated_low)
            & (propensity_scores <= treated_high)
            & treated_mask
        )
        keep_control = (
            (propensity_scores >= control_low)
            & (propensity_scores <= control_high)
            & control_mask
        )
        keep_mask = keep_treated | keep_control

        data_trimmed = data_trimmed[keep_mask]

    n_removed = len(data) - len(data_trimmed)
    if n_removed > 0:
        logger.info(
            f"Removed {n_removed} observations ({n_removed / len(data) * 100:.1f}%) during trimming"
        )

        # Count treated and control in removed
        treated_after = data_trimmed[treatment_col].sum()
        control_after = len(data_trimmed) - treated_after
        treated_removed = np.sum(treatment) - treated_after
        control_removed = (len(treatment) - np.sum(treatment)) - control_after

        logger.debug(
            f"Removed {treated_removed} treated and {control_removed} control observations"
        )
        logger.debug(
            f"After trimming: {len(data_trimmed)} observations ({treated_after} treated, {control_after} control)"
        )
    else:
        logger.info("No observations were removed during trimming")

    return data_trimmed


def assess_common_support(
    propensity_scores: np.ndarray, treatment: np.ndarray, bins: int = 20
) -> dict[str, Any]:
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
        raise ValueError(
            f"Treatment must contain only binary values (0/1), found: {unique_treatment}"
        )

    # Check that propensity scores are between 0 and 1
    if np.any(propensity_scores < 0) or np.any(propensity_scores > 1):
        raise ValueError(
            f"Propensity scores must be between 0 and 1, found min={np.min(propensity_scores)}, max={np.max(propensity_scores)}"
        )

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
    hist_treated, bin_edges = np.histogram(
        treated_ps, bins=bins, range=all_range, density=True
    )
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
        "bin_edges": bin_edges,
    }


@suppress_warnings
def assess_propensity_overlap(
    data: pd.DataFrame,
    propensity_col: str,
    treatment_col: str,
    matched_indices: pd.Index | None = None,
) -> dict[str, float]:
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
    validate_data(data=data, treatment_col=treatment_col, propensity_col=propensity_col)

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
    treated_in_cs = (
        (treated_ps >= common_support_min) & (treated_ps <= common_support_max)
    ).mean()
    control_in_cs = (
        (control_ps >= common_support_min) & (control_ps <= common_support_max)
    ).mean()

    return {
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "overlap_coefficient": common_support["overlap_coefficient"],
        "common_support_range": common_support_range,
        "treated_range": treated_range,
        "control_range": control_range,
        "prop_in_common_support": in_common_support,
        "prop_treated_in_cs": treated_in_cs,
        "prop_control_in_cs": control_in_cs,
    }


@suppress_warnings
def calculate_propensity_quality(
    data: pd.DataFrame,
    propensity_col: str,
    treatment_col: str,
    matched_indices: pd.Index | None = None,
) -> dict[str, float]:
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
    validate_data(data=data, treatment_col=treatment_col, propensity_col=propensity_col)

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
            propensity_col=propensity_col,
        )

        matched_ps = matched_data[propensity_col].values
        matched_treatment = matched_data[treatment_col].values

        ks_stat_after, ks_pval_after = stats.ks_2samp(
            matched_ps[matched_treatment == 1], matched_ps[matched_treatment == 0]
        )

        mean_diff_after = np.mean(matched_ps[matched_treatment == 1]) - np.mean(
            matched_ps[matched_treatment == 0]
        )
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
        "common_support_ratio": common_support_ratio,
    }
