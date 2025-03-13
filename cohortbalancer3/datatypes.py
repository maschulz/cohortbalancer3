"""
Datatypes for CohortBalancer3.

This module defines the data structures used by the CohortBalancer3 package,
including the configuration settings and result container.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class MatcherConfig:
    """Unified configuration for CohortMatcher with flattened parameters."""
    # Core parameters
    treatment_col: str
    covariates: List[str]
    
    # Matching parameters
    match_method: str = "greedy"  # "greedy", "optimal", "propensity"
    distance_method: str = "euclidean"  # "euclidean", "mahalanobis", "propensity", "logit" 
    exact_match_cols: List[str] = field(default_factory=list)
    standardize: bool = True
    caliper: Optional[Union[float, str]] = None  # Numeric value, "auto", or None
    caliper_scale: float = 0.2  # Scaling factor for automatic caliper calculation
    replace: bool = False
    ratio: float = 1.0
    random_state: Optional[int] = None
    weights: Optional[Dict[str, float]] = None
    
    # Propensity parameters
    estimate_propensity: bool = False
    propensity_col: Optional[str] = None
    logit_transform: bool = True
    common_support_trimming: bool = False
    trim_threshold: float = 0.05
    propensity_model: str = "logistic"  # "logistic", "random_forest", "xgboost", "custom"
    model_params: Dict[str, Any] = field(default_factory=dict)
    cv_folds: int = 5
    
    # Balance parameters
    calculate_balance: bool = True
    max_standardized_diff: float = 0.1
    
    # Outcome parameters
    outcomes: List[str] = field(default_factory=list)
    estimand: str = "ate"  # "ate", "att", "atc"
    effect_method: str = "mean_difference"  # "mean_difference", "regression_adjustment"
    adjustment_covariates: Optional[List[str]] = None
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95


@dataclass
class MatchResults:
    """Container for all matching results."""
    # Original data
    original_data: pd.DataFrame
    
    # Matching results
    matched_data: pd.DataFrame
    treatment_indices: pd.Index
    control_indices: pd.Index
    match_pairs: Dict[int, List[int]]
    match_distances: List[float]
    distance_matrix: Optional[np.ndarray] = None
    
    # Propensity score results
    propensity_scores: Optional[np.ndarray] = None
    propensity_model: Optional[Any] = None
    propensity_metrics: Optional[Dict[str, float]] = None
    
    # Balance assessment results
    balance_statistics: Optional[pd.DataFrame] = None
    rubin_statistics: Optional[Dict[str, float]] = None
    balance_index: Optional[Dict[str, float]] = None
    
    # Treatment effect results
    effect_estimates: Optional[pd.DataFrame] = None
    
    # Configuration used
    config: MatcherConfig = None
    
    def get_match_summary(self) -> Dict[str, Union[int, float]]:
        """Get summary statistics about the matching."""
        treatment_col = self.config.treatment_col
        result = {
            "n_treatment_orig": (self.original_data[treatment_col] == 1).sum(),
            "n_control_orig": (self.original_data[treatment_col] == 0).sum(),
            "n_treatment_matched": (self.matched_data[treatment_col] == 1).sum(),
            "n_control_matched": (self.matched_data[treatment_col] == 0).sum(),
        }
        
        # Calculate match ratio
        if result["n_treatment_matched"] > 0:
            result["match_ratio"] = result["n_control_matched"] / result["n_treatment_matched"]
        else:
            result["match_ratio"] = 0
            
        return result
    
    def get_balance_summary(self) -> pd.DataFrame:
        """Get summary of balance statistics."""
        if self.balance_statistics is None:
            raise ValueError("Balance statistics not available")
        return self.balance_statistics
    
    def get_effect_summary(self) -> pd.DataFrame:
        """Get summary of treatment effect estimates."""
        if self.effect_estimates is None:
            raise ValueError("Treatment effect estimates not available")
        return self.effect_estimates
    
    def get_match_pairs(self) -> pd.DataFrame:
        """Get detailed matching information as a DataFrame."""
        rows = []
        for t_idx, c_indices in self.match_pairs.items():
            for c_idx in c_indices:
                rows.append({
                    'treatment_id': self.treatment_indices[t_idx],
                    'control_id': self.control_indices[c_idx]
                })
        
        if not rows:
            return pd.DataFrame(columns=['treatment_id', 'control_id'])
        
        return pd.DataFrame(rows)
