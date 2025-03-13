"""
Distance calculation functions for matching algorithms.

This module provides functions for calculating distances between treatment and control units,
which are used by the matching algorithms.
"""

from typing import Optional, Tuple

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
    # Validate inputs
    if X_treat.ndim != 2 or X_control.ndim != 2:
        raise ValueError("X_treat and X_control must be 2D arrays")
    
    if X_treat.shape[1] != X_control.shape[1]:
        raise ValueError(f"X_treat and X_control must have same number of features")
    
    if method not in {"euclidean", "mahalanobis", "propensity", "logit"}:
        raise ValueError(f"Unknown distance method: {method}")
    
    if weights is not None and len(weights) != X_treat.shape[1]:
        raise ValueError(f"Weights length must match number of features")

    # Handle propensity-based methods
    if method in ["propensity", "logit"]:
        return _calculate_propensity_distances(X_treat, X_control, method, logit_transform)
    
    # Standardize if requested
    if standardize:
        X_treat, X_control = _standardize_data(X_treat, X_control)
    
    # Apply weights for Euclidean distance
    if method == "euclidean" and weights is not None:
        weights_sqrt = np.sqrt(weights.ravel())
        X_treat = X_treat * weights_sqrt
        X_control = X_control * weights_sqrt
        
    # Calculate distances
    if method == "euclidean":
        return cdist(X_treat, X_control, metric='euclidean')
    else:  # mahalanobis
        if cov_matrix is None:
            X_combined = np.vstack((X_treat, X_control))
            cov_matrix = np.cov(X_combined, rowvar=False)
        
        try:
            # Add small regularization term for numerical stability
            cov_inv = np.linalg.inv(cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0]))
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix inversion failed, using pseudoinverse")
            cov_inv = np.linalg.pinv(cov_matrix)
        
        return cdist(X_treat, X_control, metric='mahalanobis', VI=cov_inv)

def _standardize_data(X_treat: np.ndarray, X_control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize treatment and control data using sklearn's StandardScaler."""
    X_combined = np.vstack((X_treat, X_control))
    scaler = StandardScaler()
    scaler.fit(X_combined)
    
    # Check for zero variance features
    zero_std_features = np.where(scaler.scale_ < 1e-10)[0]
    if len(zero_std_features) > 0:
        logger.warning(f"Found {len(zero_std_features)} feature(s) with near-zero standard deviation")
    
    return scaler.transform(X_treat), scaler.transform(X_control)

def _calculate_propensity_distances(
    X_treat: np.ndarray,
    X_control: np.ndarray,
    method: str,
    logit_transform: bool
) -> np.ndarray:
    """Calculate distances for propensity-based methods."""
    # Ensure 1D arrays
    X_treat_1d = X_treat.ravel()
    X_control_1d = X_control.ravel()
    
    # Apply logit transform if needed
    if logit_transform or method == "logit":
        # Clip to avoid numerical issues
        X_treat_1d = np.clip(X_treat_1d, 0.001, 0.999)
        X_control_1d = np.clip(X_control_1d, 0.001, 0.999)
        X_treat_1d = logit(X_treat_1d)
        X_control_1d = logit(X_control_1d)
    
    return cdist(
        X_treat_1d.reshape(-1, 1),
        X_control_1d.reshape(-1, 1),
        metric='euclidean'
    )
