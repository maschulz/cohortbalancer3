"""
Distance calculation functions for matching algorithms.

This module provides functions for calculating distances between treatment and control units,
which are used by the matching algorithms.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logit
from sklearn.preprocessing import StandardScaler

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
    
    # Standardize if required (and not propensity-based method)
    if standardize and method not in ["propensity", "logit"]:
        logger.debug("Standardizing data before distance calculation")
        X_treat, X_control = standardize_data(X_treat, X_control)
    
    # Apply weights for Euclidean distance if provided
    if method == "euclidean" and weights is not None:
        logger.debug("Applying feature weights to Euclidean distance calculation")
        # Apply sqrt(weight) to features for weighted Euclidean distance
        weights_sqrt = np.sqrt(np.asarray(weights).ravel())
        X_treat = X_treat * weights_sqrt
        X_control = X_control * weights_sqrt
    
    # Calculate distances using the specified method
    if method == "euclidean":
        distance_matrix = cdist(X_treat, X_control, metric='euclidean')
    elif method == "mahalanobis":
        # For Mahalanobis, we need to handle the covariance matrix
        if cov_matrix is None:
            logger.debug("No covariance matrix provided, estimating from data")
            # Combine data to estimate covariance
            X_combined = np.vstack((X_treat, X_control))
            cov_matrix = np.cov(X_combined, rowvar=False)
        
        try:
            # Add small regularization term for numerical stability
            cov_inv = np.linalg.inv(cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0]))
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix inversion failed, using pseudoinverse")
            cov_inv = np.linalg.pinv(cov_matrix)
        
        distance_matrix = cdist(X_treat, X_control, metric='mahalanobis', VI=cov_inv)
    elif method in ["propensity", "logit"]:
        # Ensure propensity scores are 1D
        X_treat_1d = X_treat.ravel()
        X_control_1d = X_control.ravel()
        
        logger.debug(f"Propensity score ranges - Treatment: [{X_treat_1d.min():.4f}, {X_treat_1d.max():.4f}], "
                    f"Control: [{X_control_1d.min():.4f}, {X_control_1d.max():.4f}]")
        
        # Apply logit transform if needed (always for 'logit' method)
        apply_logit = logit_transform or method == "logit"
        
        if apply_logit:
            logger.debug("Applying logit transformation with clipping to [0.001, 0.999]")
            # Reshape for proper broadcasting
            X_treat_flat = X_treat_1d.reshape(-1, 1)
            X_control_flat = X_control_1d.reshape(-1, 1)
            
            # Clip to avoid log(0) or log(inf)
            X_treat_clipped = np.clip(X_treat_flat, 0.001, 0.999)
            X_control_clipped = np.clip(X_control_flat, 0.001, 0.999)
            
            # Apply logit transform
            X_treat_transformed = logit(X_treat_clipped)
            X_control_transformed = logit(X_control_clipped)
            
            # Calculate distances
            distance_matrix = cdist(X_treat_transformed, X_control_transformed, metric='euclidean')
        else:
            # Without logit transform, just calculate absolute differences
            X_treat_flat = X_treat_1d.reshape(-1, 1)
            X_control_flat = X_control_1d.reshape(-1, 1)
            distance_matrix = cdist(X_treat_flat, X_control_flat, metric='euclidean')
    
    logger.debug(f"Distance matrix calculated with shape: {distance_matrix.shape}")
    logger.debug(f"Distance matrix stats: min={distance_matrix.min():.4f}, "
                f"mean={distance_matrix.mean():.4f}, max={distance_matrix.max():.4f}")
    
    return distance_matrix


def standardize_data(
    X_treat: np.ndarray, X_control: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize treatment and control data to have mean 0 and std 1.
    
    Uses sklearn's StandardScaler for more robust standardization.
    
    Args:
        X_treat: Treatment data array
        X_control: Control data array
        
    Returns:
        Tuple of (standardized X_treat, standardized X_control)
    """
    logger.debug(f"Standardizing data: X_treat {X_treat.shape}, X_control {X_control.shape}")
    
    # Combine for consistent standardization
    X_combined = np.vstack((X_treat, X_control))
    
    # Use sklearn's StandardScaler which handles zero variance features
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_combined)
    
    # Check for zero variance features and warn
    zero_std_features = np.where(scaler.scale_ < 1e-10)[0]
    if len(zero_std_features) > 0:
        logger.warning(f"Found {len(zero_std_features)} feature(s) with near-zero standard deviation. "
                      f"These will be set to zero in the standardized data.")
    
    # Transform the data
    X_treat_std = scaler.transform(X_treat)
    X_control_std = scaler.transform(X_control)
    
    logger.debug(f"Standardization complete")
    
    return X_treat_std, X_control_std
