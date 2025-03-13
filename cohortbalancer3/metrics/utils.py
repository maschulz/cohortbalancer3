"""
Utility functions for propensity score matching and treatment effect estimation.

This module provides helper functions shared across different metrics calculations.
"""

import numpy as np
from scipy.special import logit
from typing import Optional, Union, Dict, Any

from cohortbalancer3.utils.logging import get_logger

# Set up logger
logger = get_logger(__name__)


def calculate_recommended_caliper(
    propensity_scores: Optional[np.ndarray] = None,
    distance_matrix: Optional[np.ndarray] = None,
    method: str = "propensity",
    caliper_scale: float = 0.2,
    percentile: float = 90.0
) -> Optional[float]:
    """Calculate recommended caliper based on method and data.
    
    For propensity score methods, the recommended caliper is:
    caliper_scale × standard deviation of the logit of propensity scores
    (default scale factor is 0.2, based on Austin 2011).
    
    For Mahalanobis and other distance methods, the caliper is based on
    the distribution of distances (defaulting to the specified percentile).
    
    Args:
        propensity_scores: Propensity scores (required for propensity methods)
        distance_matrix: Distance matrix (required for non-propensity methods)
        method: Distance calculation method ('propensity', 'logit', 'mahalanobis', 'euclidean')
        caliper_scale: Scaling factor for caliper (default: 0.2 for propensity)
        percentile: Percentile of distance distribution to use for non-propensity methods
        
    Returns:
        Recommended caliper value, or None if caliper cannot be calculated
    """
    if method in ["propensity", "logit"]:
        if propensity_scores is None:
            logger.warning("Cannot calculate recommended caliper: propensity scores not provided")
            return None
        
        # Clip propensity scores to avoid numerical issues with logit
        ps_clipped = np.clip(propensity_scores, 0.001, 0.999)
        
        # Apply logit transformation
        logit_ps = logit(ps_clipped)
        
        # Calculate SD of logit propensity scores
        logit_ps_sd = np.std(logit_ps)
        
        # Recommended caliper: caliper_scale × SD of logit of propensity
        rec_caliper = caliper_scale * logit_ps_sd
        logger.info(f"Recommended caliper for {method} method: {rec_caliper:.4f} "
                   f"({caliper_scale} × SD of logit propensity={logit_ps_sd:.4f})")
        return rec_caliper
    
    elif distance_matrix is not None:
        # For other methods (mahalanobis, euclidean), base on distance distribution
        # Use only finite distances
        finite_mask = np.isfinite(distance_matrix)
        if not np.any(finite_mask):
            logger.warning("Cannot calculate recommended caliper: no finite distances in matrix")
            return None
        
        finite_distances = distance_matrix[finite_mask]
        
        if method == "mahalanobis":
            # For Mahalanobis, use percentile of distance distribution
            rec_caliper = np.percentile(finite_distances, percentile)
            logger.info(f"Recommended caliper for {method} method: {rec_caliper:.4f} "
                       f"({percentile}th percentile of distance distribution)")
        else:
            # For euclidean and other methods, use median of distance distribution
            rec_caliper = np.median(finite_distances)
            logger.info(f"Recommended caliper for {method} method: {rec_caliper:.4f} "
                       f"(median of distance distribution)")
        
        return rec_caliper
    
    else:
        logger.warning(f"Cannot calculate recommended caliper for {method} method: "
                      "neither propensity scores nor distance matrix provided")
        return None


def get_caliper_for_matching(
    config_caliper: Union[float, str, None],
    propensity_scores: Optional[np.ndarray] = None,
    distance_matrix: Optional[np.ndarray] = None, 
    method: str = "propensity",
    caliper_scale: float = 0.2
) -> Optional[float]:
    """Get caliper value for matching based on configuration and data.
    
    Args:
        config_caliper: Caliper from configuration (float, 'auto', or None)
        propensity_scores: Propensity scores (required for auto caliper with propensity methods)
        distance_matrix: Distance matrix (required for auto caliper with non-propensity methods)
        method: Distance calculation method
        caliper_scale: Scaling factor for automatic caliper calculation
    
    Returns:
        Caliper value to use for matching, or None if no caliper should be applied
    """
    # If caliper is None, return None (no caliper)
    if config_caliper is None:
        return None
    
    # If caliper is a numeric value, return it directly
    if isinstance(config_caliper, (int, float)):
        return float(config_caliper)
    
    # If caliper is 'auto', calculate recommended caliper
    if isinstance(config_caliper, str) and config_caliper.lower() == 'auto':
        return calculate_recommended_caliper(
            propensity_scores=propensity_scores,
            distance_matrix=distance_matrix,
            method=method,
            caliper_scale=caliper_scale
        )
    
    # Otherwise, invalid caliper specification
    raise ValueError(f"Invalid caliper specification: {config_caliper}. "
                    "Must be a positive number, 'auto', or None.") 