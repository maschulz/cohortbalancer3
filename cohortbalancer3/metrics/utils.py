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


def get_caliper_for_matching(
    config_caliper: Union[float, str, None],
    propensity_scores: Optional[np.ndarray] = None,
    distance_matrix: Optional[np.ndarray] = None, 
    method: str = "propensity",
    caliper_scale: float = 0.2,
    percentile: float = 90.0
) -> Optional[float]:
    """Get caliper value for matching based on configuration and data.
    
    This function handles all caliper calculation logic, including:
    - Direct numeric values
    - Automatic calculation based on data
    - No caliper (None)
    
    For propensity score methods with 'auto' caliper, the recommended value is:
    caliper_scale × standard deviation of the logit of propensity scores
    (default scale factor is 0.2, based on Austin 2011).
    
    For Mahalanobis and Euclidean with 'auto' caliper, the value is the specified percentile
    of the distance distribution (default: 90th percentile).
    
    Args:
        config_caliper: Caliper specification (float, 'auto', or None)
        propensity_scores: Propensity scores (required for auto caliper with propensity methods)
        distance_matrix: Distance matrix (required for auto caliper with non-propensity methods)
        method: Distance calculation method ('propensity', 'logit', 'mahalanobis', 'euclidean')
        caliper_scale: Scaling factor for propensity-based caliper calculation (default: 0.2)
        percentile: Percentile of distance distribution to use for Mahalanobis and Euclidean methods (default: 90.0)
    
    Returns:
        Caliper value to use for matching, or None if no caliper should be applied
        
    Raises:
        ValueError: If 'auto' caliper is requested but required data is not provided,
                   or if the caliper specification is invalid
    """
    # If caliper is None, return None (no caliper)
    if config_caliper is None:
        return None
    
    # If caliper is a numeric value, return it directly
    if isinstance(config_caliper, (int, float)):
        return float(config_caliper)
    
    # Handle 'auto' caliper calculation
    if isinstance(config_caliper, str) and config_caliper.lower() == 'auto':
        # For propensity-based methods
        if method in ["propensity", "logit"]:
            if propensity_scores is None:
                raise ValueError(
                    f"Cannot calculate auto caliper for {method} method: "
                    f"propensity scores are required but not provided."
                )
            
            # Clip propensity scores to avoid numerical issues with logit
            ps_clipped = np.clip(propensity_scores, 0.001, 0.999)
            
            # Apply logit transformation
            logit_ps = logit(ps_clipped)
            
            # Calculate SD of logit propensity scores
            logit_ps_sd = np.std(logit_ps)
            
            # Recommended caliper: caliper_scale × SD of logit of propensity
            rec_caliper = caliper_scale * logit_ps_sd
            logger.info(f"Auto caliper for {method} method: {rec_caliper:.4f} "
                       f"({caliper_scale} × SD of logit propensity={logit_ps_sd:.4f})")
            return rec_caliper
        
        # For non-propensity methods
        else:
            if distance_matrix is None:
                raise ValueError(
                    f"Cannot calculate auto caliper for {method} method: "
                    f"distance matrix is required but not provided."
                )
            
            # Check for finite values in distance matrix
            finite_mask = np.isfinite(distance_matrix)
            if not np.any(finite_mask):
                raise ValueError(
                    f"Cannot calculate auto caliper for {method} method: "
                    f"no finite distances in matrix."
                )
            
            finite_distances = distance_matrix[finite_mask]
            
            if method in ["mahalanobis", "euclidean"]:
                # For Mahalanobis and Euclidean, use percentile of distance distribution
                rec_caliper = np.percentile(finite_distances, percentile)
                logger.info(f"Auto caliper for {method} method: {rec_caliper:.4f} "
                           f"({percentile}th percentile of distance distribution)")
            else:
                # For other methods, use median of distance distribution
                rec_caliper = np.median(finite_distances)
                logger.info(f"Auto caliper for {method} method: {rec_caliper:.4f} "
                           f"(median of distance distribution)")
            
            return rec_caliper
    
    # Otherwise, invalid caliper specification
    raise ValueError(f"Invalid caliper specification: {config_caliper}. "
                    "Must be a positive number, 'auto', or None.") 