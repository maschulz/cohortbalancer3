"""
Distance calculation functions for matching algorithms.

This module provides functions for calculating distances between treatment and control units,
which are used by the matching algorithms.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logit

# Import logger
from cohortbalancer3.utils.logging import get_logger

# Create a logger for this module
logger = get_logger(__name__)


def calculate_distance_matrix(
    X_treat: np.ndarray,
    X_control: np.ndarray,
    method: str = "euclidean",
    standardize: bool = True,
    weights: Optional[np.ndarray] = None,
    cov_matrix: Optional[np.ndarray] = None,
    logit_transform: bool = False,
) -> np.ndarray:
    """Calculate distance matrix between treatment and control groups.
    
    Args:
        X_treat: Array of treatment group features, shape (n_treatment, n_features)
        X_control: Array of control group features, shape (n_control, n_features)
        method: Distance calculation method ('euclidean', 'mahalanobis', 'propensity', 'logit')
        standardize: Whether to standardize features before calculating distances
        weights: Feature weights for euclidean distance, shape (n_features,)
        cov_matrix: Covariance matrix for Mahalanobis distance, shape (n_features, n_features)
        logit_transform: Whether to apply logit transformation (for propensity scores)
    
    Returns:
        Distance matrix, shape (n_treatment, n_control)
    """
    logger.debug(f"Calculating distance matrix using method: {method}")
    logger.debug(f"Input dimensions: X_treat {X_treat.shape}, X_control {X_control.shape}")
    
    # Validate inputs
    if X_treat.ndim != 2 or X_control.ndim != 2:
        raise ValueError("X_treat and X_control must be 2D arrays")
    
    if X_treat.shape[1] != X_control.shape[1]:
        raise ValueError(f"X_treat and X_control must have the same number of features, "
                         f"but got {X_treat.shape[1]} and {X_control.shape[1]}")
    
    # Validate method parameter
    valid_methods = {"euclidean", "mahalanobis", "propensity", "logit"}
    if method not in valid_methods:
        raise ValueError(f"Unknown distance method: {method}. Must be one of: {', '.join(valid_methods)}")
    
    # Validate weights if provided
    if weights is not None:
        logger.debug(f"Using feature weights with shape: {weights.shape}")
        if len(weights) != X_treat.shape[1]:
            raise ValueError(f"Weights length ({len(weights)}) must match number of features ({X_treat.shape[1]})")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
    
    # Validate covariance matrix if provided
    if cov_matrix is not None:
        logger.debug(f"Using provided covariance matrix with shape: {cov_matrix.shape}")
        if cov_matrix.shape != (X_treat.shape[1], X_treat.shape[1]):
            raise ValueError(f"Covariance matrix shape {cov_matrix.shape} does not match "
                             f"expected shape ({X_treat.shape[1]}, {X_treat.shape[1]})")
    
    if standardize:
        logger.debug("Standardization is enabled")
    
    # Calculate distances using the specified method
    if method == "euclidean":
        distance_matrix = euclidean_distance(
            X_treat, X_control, weights=weights, standardize=standardize
        )
    elif method == "mahalanobis":
        distance_matrix = mahalanobis_distance(X_treat, X_control, cov=cov_matrix)
    elif method == "propensity":
        if logit_transform:
            logger.debug("Applying logit transformation to propensity scores")
        distance_matrix = propensity_distance(X_treat, X_control, apply_logit=logit_transform)
    elif method == "logit":
        # Logit is just propensity with logit transform
        logger.debug("Using logit method (propensity with logit transform)")
        distance_matrix = propensity_distance(X_treat, X_control, apply_logit=True)
    else:
        raise ValueError(f"Unsupported distance method: {method}")
    
    logger.debug(f"Distance matrix calculated with shape: {distance_matrix.shape}")
    logger.debug(f"Distance matrix stats: min={distance_matrix.min():.4f}, mean={distance_matrix.mean():.4f}, max={distance_matrix.max():.4f}")
    
    return distance_matrix


def standardize_data(
    X_treat: np.ndarray, X_control: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize treatment and control data to have mean 0 and std 1.
    
    Args:
        X_treat: Treatment data array
        X_control: Control data array
        
    Returns:
        Tuple of (standardized X_treat, standardized X_control, means, stds)
    """
    logger.debug(f"Standardizing data: X_treat {X_treat.shape}, X_control {X_control.shape}")
    
    # Assuming all data is numeric with no missing values

    # Combine for consistent standardization
    X_combined = np.vstack((X_treat, X_control))

    # Calculate mean and std for each feature
    means = np.mean(X_combined, axis=0)
    stds = np.std(X_combined, axis=0)

    # Replace zero std with 1 to avoid division by zero
    zero_std_count = np.sum(stds == 0)
    if zero_std_count > 0:
        logger.warning(f"Found {zero_std_count} feature(s) with zero standard deviation. These will not be standardized.")
        stds[stds == 0] = 1.0

    # Standardize features
    X_treat_std = (X_treat - means) / stds
    X_control_std = (X_control - means) / stds

    logger.debug(f"Standardization complete. Mean of combined data: {np.mean(np.vstack((X_treat_std, X_control_std))), np.std(np.vstack((X_treat_std, X_control_std)))}")
    
    return X_treat_std, X_control_std, means, stds


def euclidean_distance(
    X_treat: np.ndarray,
    X_control: np.ndarray,
    weights: Optional[np.ndarray] = None,
    standardize: bool = True,
) -> np.ndarray:
    """Calculate Euclidean distance between treatment and control units.
    
    Args:
        X_treat: Array of treatment group features, shape (n_treatment, n_features)
        X_control: Array of control group features, shape (n_control, n_features)
        weights: Feature weights, shape (n_features,)
        standardize: Whether to standardize features before calculating distances
    
    Returns:
        Distance matrix, shape (n_treatment, n_control)
    """
    logger.debug("Calculating Euclidean distances")
    
    if standardize:
        logger.debug("Standardizing data before Euclidean distance calculation")
        X_treat, X_control, _, _ = standardize_data(X_treat, X_control)

    if weights is not None:
        logger.debug(f"Applying feature weights to Euclidean distance calculation")
        # Apply weights
        # Ensure weights is a 1D array
        weights = np.asarray(weights).ravel()

        # Weighted Euclidean distance: apply sqrt(weight) to each feature
        # This works because √(Σwᵢ(xᵢ-yᵢ)²) = √(Σ(√wᵢ(xᵢ-yᵢ))²)
        X_treat = X_treat * np.sqrt(weights)
        X_control = X_control * np.sqrt(weights)

    # Calculate pairwise distances
    distance_matrix = cdist(X_treat, X_control, metric='euclidean')
    logger.debug(f"Euclidean distance matrix calculated with shape {distance_matrix.shape}")
    
    return distance_matrix


def mahalanobis_distance(
    X_treat: np.ndarray,
    X_control: np.ndarray,
    cov: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate Mahalanobis distance between treatment and control units.
    
    The Mahalanobis distance accounts for covariance between features.
    d(x,y) = √((x-y)ᵀ Σ⁻¹ (x-y))
    
    Args:
        X_treat: Array of treatment group features, shape (n_treatment, n_features)
        X_control: Array of control group features, shape (n_control, n_features)
        cov: Covariance matrix, shape (n_features, n_features). If None, calculated from data.
    
    Returns:
        Distance matrix, shape (n_treatment, n_control)
    """
    logger.debug("Calculating Mahalanobis distances")
    
    if X_treat.shape[1] != X_control.shape[1]:
        raise ValueError(
            f"Incompatible feature dimensions: {X_treat.shape[1]} != {X_control.shape[1]}"
        )

    if cov is None:
        logger.debug("No covariance matrix provided, estimating from data")
        # Combine data to estimate covariance
        X_combined = np.vstack((X_treat, X_control))
        cov = np.cov(X_combined, rowvar=False)
    else:
        logger.debug(f"Using provided covariance matrix with shape {cov.shape}")

    try:
        # Calculate inverse of covariance matrix
        # Add small regularization term to ensure positive definiteness
        logger.debug("Computing inverse of covariance matrix")
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix inversion failed, using pseudoinverse instead")
        # If inversion fails, use pseudoinverse
        cov_inv = np.linalg.pinv(cov)

    # Calculate Mahalanobis distance using SciPy's cdist
    distance_matrix = cdist(X_treat, X_control, metric='mahalanobis', VI=cov_inv)
    logger.debug(f"Mahalanobis distance matrix calculated with shape {distance_matrix.shape}")
    
    return distance_matrix


def propensity_distance(
    X_treat: np.ndarray,
    X_control: np.ndarray,
    apply_logit: bool = True,
) -> np.ndarray:
    """Calculate distance based on propensity scores.
    
    Args:
        X_treat: Propensity scores for treatment units
        X_control: Propensity scores for control units
        apply_logit: Whether to apply logit transformation
        
    Returns:
        Distance matrix
    """
    logger.debug(f"Calculating propensity distances (apply_logit={apply_logit})")
    
    # Ensure propensity scores are 1D
    X_treat = X_treat.ravel().reshape(-1, 1)
    X_control = X_control.ravel().reshape(-1, 1)
    
    logger.debug(f"Propensity score ranges - Treatment: [{X_treat.min():.4f}, {X_treat.max():.4f}], "
                 f"Control: [{X_control.min():.4f}, {X_control.max():.4f}]")

    if apply_logit:
        # Apply logit transformation to propensity scores
        # Clip to avoid log(0) or log(inf)
        logger.debug("Applying logit transformation with clipping to [0.001, 0.999]")
        X_treat_clipped = np.clip(X_treat, 0.001, 0.999)
        X_control_clipped = np.clip(X_control, 0.001, 0.999)

        X_treat_transformed = logit(X_treat_clipped)
        X_control_transformed = logit(X_control_clipped)

        # Calculate absolute differences
        distance_matrix = cdist(X_treat_transformed, X_control_transformed, metric='euclidean')
    else:
        # Calculate absolute differences without transformation
        distance_matrix = cdist(X_treat, X_control, metric='euclidean')
    
    logger.debug(f"Propensity distance matrix calculated with shape {distance_matrix.shape}")
    
    return distance_matrix


def standardized_distance(
    X_treat: np.ndarray,
    X_control: np.ndarray,
    method: str = "euclidean",
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate standardized distance matrix.
    
    This function standardizes the features and then calculates
    the distance matrix.
    
    Args:
        X_treat: Array of treatment group features
        X_control: Array of control group features
        method: Distance calculation method
        weights: Feature weights
    
    Returns:
        Standardized distance matrix
    """
    logger.debug(f"Calculating standardized distances using {method} method")
    
    X_treat_std, X_control_std, _, _ = standardize_data(X_treat, X_control)

    return calculate_distance_matrix(
        X_treat_std, X_control_std, method=method, standardize=False, weights=weights
    )
