"""
Datatypes for CohortBalancer3.

This module defines the data structures used by the CohortBalancer3 package,
including the configuration settings and result container.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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
    """Container for all matching results.
    
    This class stores the results of a matching operation, including the original data,
    matched data, and matching pairs. It provides methods for retrieving matching information
    and summarizing the results.
    
    Attributes:
        original_data: The original DataFrame before matching
        matched_data: DataFrame containing only the matched units
        pairs: List of tuples (treatment_id, control_id) representing matched pairs
        match_groups: Dictionary mapping treatment IDs to lists of control IDs
        match_distances: List of distances for each matched pair
    """
    # Original and matched data
    original_data: pd.DataFrame
    matched_data: pd.DataFrame
    
    # Matching results as pairs of participant IDs
    # Each tuple is (treatment_id, control_id)
    pairs: List[Tuple[Any, Any]]
    
    # Dictionary mapping treatment IDs to lists of control IDs
    # This is particularly useful for ratio matching and efficient lookups
    match_groups: Dict[Any, List[Any]]
    
    # Distances for each matched pair, in the same order as pairs
    match_distances: List[float]
    
    # Optional distance matrix for debugging/visualization
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
        """Get summary statistics about the matching.
        
        Returns:
            Dictionary with match summary statistics
        """
        treatment_col = self.config.treatment_col
        result = {
            "n_treatment_orig": (self.original_data[treatment_col] == 1).sum(),
            "n_control_orig": (self.original_data[treatment_col] == 0).sum(),
            "n_treatment_matched": (self.matched_data[treatment_col] == 1).sum(),
            "n_control_matched": (self.matched_data[treatment_col] == 0).sum(),
            "n_pairs": len(self.pairs),
            "n_match_groups": len(self.match_groups)
        }
        
        # Calculate match ratio
        if result["n_treatment_matched"] > 0:
            result["match_ratio"] = result["n_control_matched"] / result["n_treatment_matched"]
        else:
            result["match_ratio"] = 0
            
        return result
    
    def get_balance_summary(self) -> pd.DataFrame:
        """Get summary of balance statistics.
        
        Returns:
            DataFrame with balance statistics
            
        Raises:
            ValueError: If balance statistics are not available
        """
        if self.balance_statistics is None:
            raise ValueError("Balance statistics not available")
        return self.balance_statistics
    
    def get_effect_summary(self) -> pd.DataFrame:
        """Get summary of treatment effect estimates.
        
        Returns:
            DataFrame with treatment effect estimates
            
        Raises:
            ValueError: If treatment effect estimates are not available
        """
        if self.effect_estimates is None:
            raise ValueError("Treatment effect estimates not available")
        return self.effect_estimates
    
    def get_match_pairs(self) -> pd.DataFrame:
        """Get detailed matching information as a DataFrame.
        
        This method converts the internal pairs representation into a DataFrame
        with treatment_id and control_id columns, suitable for analysis and export.
        
        Returns:
            DataFrame with columns 'treatment_id' and 'control_id'
        """
        if not self.pairs:
            return pd.DataFrame(columns=['treatment_id', 'control_id'])
        
        rows = []
        for t_id, c_id in self.pairs:
            rows.append({
                'treatment_id': t_id,
                'control_id': c_id
            })
        
        return pd.DataFrame(rows)
    
    def get_match_groups(self) -> pd.DataFrame:
        """Get matching groups as a DataFrame.
        
        This method provides a view of the match groups, particularly useful for
        many-to-one matching where each treatment unit may have multiple controls.
        
        Returns:
            DataFrame with match group information
        """
        rows = []
        for t_id, c_ids in self.match_groups.items():
            for i, c_id in enumerate(c_ids):
                rows.append({
                    'treatment_id': t_id,
                    'control_id': c_id,
                    'match_group': t_id,
                    'match_number': i + 1,
                    'group_size': len(c_ids)
                })
        
        if not rows:
            return pd.DataFrame(columns=['treatment_id', 'control_id', 'match_group', 
                                        'match_number', 'group_size'])
        
        return pd.DataFrame(rows)
