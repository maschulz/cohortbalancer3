"""
Matcher implementation for CohortBalancer3.

This module provides the main Matcher class for performing matching, propensity score estimation,
balance assessment, and treatment effect estimation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import logit

from cohortbalancer3.datatypes import MatcherConfig, MatchResults
# Import from existing modules
from cohortbalancer3.matching.distances import calculate_distance_matrix
from cohortbalancer3.matching.greedy import greedy_match
from cohortbalancer3.matching.optimal import optimal_match
from cohortbalancer3.metrics.balance import (calculate_balance_index,
                                           calculate_balance_stats,
                                           calculate_rubin_rules)
from cohortbalancer3.metrics.propensity import (assess_propensity_overlap,
                                              estimate_propensity_scores,
                                              trim_by_propensity)
from cohortbalancer3.metrics.treatment import estimate_multiple_outcomes
from cohortbalancer3.validation import validate_data, validate_matcher_config


class Matcher:
    """Unified matcher for causal inference."""
    
    def __init__(self, data: pd.DataFrame, config: MatcherConfig):
        """Initialize matcher with data and configuration.
        
        Args:
            data: DataFrame containing the data
            config: Configuration settings
        """
        self.data = data.copy()
        self.config = config
        self.results = None
        
        # Validate configuration
        validate_matcher_config(self.config)
        
        # Validate input data
        validate_data(
            data=self.data,
            treatment_col=self.config.treatment_col,
            covariates=self.config.covariates,
            outcomes=self.config.outcomes,
            propensity_col=self.config.propensity_col,
            exact_match_cols=self.config.exact_match_cols
        )
    
    def match(self) -> 'Matcher':
        """Perform matching according to configuration.
        
        Returns:
            Self, for method chaining
        """
        # Step 1: Estimate propensity scores if needed
        propensity_scores = None
        propensity_model = None
        propensity_metrics = None
        
        if self.config.estimate_propensity or self.config.propensity_col:
            propensity_result = self._estimate_propensity()
            propensity_scores = propensity_result.get('propensity_scores')
            propensity_model = propensity_result.get('model')
            propensity_metrics = propensity_result.get('metrics')
        
        # Step 2: Determine matching direction (always match from smaller to larger group)
        flipped = self._determine_matching_direction()
        treatment_mask = self._get_treatment_mask(flipped)
        
        # Step 3: Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(propensity_scores, treatment_mask)
        
        # Step 4: Perform matching
        match_results = self._perform_matching(distance_matrix, treatment_mask, flipped)
        
        # Get matched data
        matched_data = self.data.loc[match_results['matched_indices']].copy()
        
        # Step 5: Calculate balance statistics
        balance_statistics = None
        rubin_statistics = None
        balance_index = None
        
        if self.config.calculate_balance:
            balance_result = self._calculate_balance(matched_data)
            balance_statistics = balance_result.get('balance_statistics')
            rubin_statistics = balance_result.get('rubin_statistics')
            balance_index = balance_result.get('balance_index')
        
        # Step 6: Estimate treatment effects if outcomes specified
        effect_estimates = None
        
        if self.config.outcomes:
            effect_estimates = self._estimate_effects(matched_data)
        
        # Combine all results
        self.results = MatchResults(
            original_data=self.data,
            matched_data=matched_data,
            treatment_indices=match_results['treatment_indices'],
            control_indices=match_results['control_indices'],
            match_pairs=match_results['match_pairs'],
            match_distances=match_results['match_distances'],
            distance_matrix=distance_matrix,
            propensity_scores=propensity_scores,
            propensity_model=propensity_model,
            propensity_metrics=propensity_metrics,
            balance_statistics=balance_statistics,
            rubin_statistics=rubin_statistics,
            balance_index=balance_index,
            effect_estimates=effect_estimates,
            config=self.config
        )
        
        return self
    
    def get_results(self) -> MatchResults:
        """Get the results of matching.
        
        Returns:
            MatchResults object containing all results
        
        Raises:
            ValueError: If no matching has been performed yet
        """
        if self.results is None:
            raise ValueError("No matching has been performed yet.")
        return self.results
    
    def save_results(self, directory: str) -> 'Matcher':
        """Save matching results to files.
        
        Args:
            directory: Directory to save results
            
        Returns:
            Self, for method chaining
        """
        import os
        import pickle
        from pathlib import Path
        
        if self.results is None:
            raise ValueError("No matching has been performed yet.")
        
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save matched data
        self.results.matched_data.to_csv(os.path.join(directory, "matched_data.csv"), index=True)
        
        # Save match pairs
        self.results.get_match_pairs().to_csv(os.path.join(directory, "match_pairs.csv"), index=False)
        
        # Save balance statistics if available
        if self.results.balance_statistics is not None:
            self.results.balance_statistics.to_csv(os.path.join(directory, "balance_statistics.csv"), index=False)
        
        # Save treatment effect estimates if available
        if self.results.effect_estimates is not None:
            self.results.effect_estimates.to_csv(os.path.join(directory, "effect_estimates.csv"), index=False)
        
        # Save configuration
        with open(os.path.join(directory, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        
        return self
    
    # Private methods for implementation details
    def _validate_data(self):
        """Validate input data and configuration."""
        # This method is now replaced by the centralized validation system.
        # The validation is performed in the __init__ method.
        pass
    
    def _determine_matching_direction(self) -> bool:
        """Determine matching direction based on group sizes.
        
        Returns:
            Boolean indicating whether treatment/control should be flipped for matching
        """
        # Count treatment and control units
        n_treatment = (self.data[self.config.treatment_col] == 1).sum()
        n_control = (self.data[self.config.treatment_col] == 0).sum()
        
        # Determine direction: we want to match from the smaller group to the larger
        # for better matching quality
        return n_treatment > n_control
    
    def _get_treatment_mask(self, flipped: bool) -> np.ndarray:
        """Get boolean mask for treatment units, potentially flipped.
        
        Args:
            flipped: Whether treatment/control are flipped for matching
            
        Returns:
            Boolean mask where True indicates a unit in the "from" group for matching
        """
        if flipped:
            # If flipped, control units (0) are considered "from" group for matching
            return self.data[self.config.treatment_col] == 0
        else:
            # Otherwise, use actual treatment units (1) as "from" group
            return self.data[self.config.treatment_col] == 1
    
    def _estimate_propensity(self) -> Dict[str, Any]:
        """Estimate propensity scores based on configuration.
        
        Returns:
            Dictionary with propensity model, scores, and metrics
        """
        # If propensity column is provided, use it directly
        if self.config.propensity_col and self.config.propensity_col in self.data.columns:
            propensity_scores = self.data[self.config.propensity_col].values
            treatment_mask = self.data[self.config.treatment_col] == 1
            
            # Calculate overlap metrics
            metrics = assess_propensity_overlap(
                data=self.data, 
                propensity_col=self.config.propensity_col,
                treatment_col=self.config.treatment_col
            )
            
            return {
                'propensity_scores': propensity_scores,
                'model': None,
                'metrics': metrics
            }
        
        # Otherwise, estimate propensity scores
        elif self.config.estimate_propensity:
            propensity_result = estimate_propensity_scores(
                data=self.data,
                treatment_col=self.config.treatment_col,
                covariates=self.config.covariates,
                model_type=self.config.propensity_model,
                model_params=self.config.model_params,
                cv=self.config.cv_folds,
                random_state=self.config.random_state
            )
            
            # Trim propensity scores if requested
            if self.config.common_support_trimming:
                trimmed_data = trim_by_propensity(
                    data=self.data,
                    propensity_scores=propensity_result['propensity_scores'],
                    treatment_col=self.config.treatment_col,
                    method='common_support',
                    trim_percent=self.config.trim_threshold
                )
                # Update data with trimmed version
                self.data = trimmed_data
                
                # Recalculate mask for trimmed data
                treatment_mask = self.data[self.config.treatment_col] == 1
                
                # Update propensity scores for trimmed data
                propensity_result['propensity_scores'] = propensity_result['propensity_scores'][trimmed_data.index]
            
            return {
                'propensity_scores': propensity_result['propensity_scores'],
                'model': propensity_result['model'],
                'metrics': propensity_result.get('metrics', {})
            }
        
        # No propensity scores
        return {
            'propensity_scores': None,
            'model': None,
            'metrics': {}
        }
    
    def _calculate_distance_matrix(self, propensity_scores: Optional[np.ndarray], treatment_mask: np.ndarray) -> np.ndarray:
        """Calculate distance matrix for matching.
        
        Args:
            propensity_scores: Optional propensity scores
            treatment_mask: Boolean mask for treatment units
            
        Returns:
            Distance matrix
        """
        # Extract treatment and control features
        if self.config.distance_method == 'propensity' or self.config.distance_method == 'logit':
            if propensity_scores is None:
                raise ValueError("Propensity scores are required for propensity or logit distance methods")
            
            # For propensity-based distances, use propensity scores as features
            X_treat = propensity_scores[treatment_mask].reshape(-1, 1)
            X_control = propensity_scores[~treatment_mask].reshape(-1, 1)
            
            # Calculate distance matrix
            return calculate_distance_matrix(
                X_treat=X_treat,
                X_control=X_control,
                method=self.config.distance_method,
                standardize=self.config.standardize,
                logit_transform=self.config.logit_transform
            )
        else:
            # For other distance methods, use covariates
            X = self.data[self.config.covariates].values
            X_treat = X[treatment_mask]
            X_control = X[~treatment_mask]
            
            # Convert weights to numpy array if provided
            weights = None
            if self.config.weights:
                weights = np.array([self.config.weights.get(cov, 1.0) 
                                  for cov in self.config.covariates])
            
            # Calculate distance matrix
            return calculate_distance_matrix(
                X_treat=X_treat,
                X_control=X_control,
                method=self.config.distance_method,
                standardize=self.config.standardize,
                weights=weights
            )
    
    def _perform_matching(self, distance_matrix: np.ndarray, treatment_mask: np.ndarray, flipped: bool) -> Dict[str, Any]:
        """Perform matching according to configuration.
        
        Args:
            distance_matrix: Pre-computed distance matrix
            treatment_mask: Boolean mask for treatment units
            flipped: Whether treatment/control are flipped for matching
            
        Returns:
            Dictionary with matching results
        """
        # Get treatment and control indices
        treatment_indices = self.data.index[treatment_mask]
        control_indices = self.data.index[~treatment_mask]
        
        # Get exact match columns if provided
        exact_match_cols = self.config.exact_match_cols if self.config.exact_match_cols else None
        
        # Perform matching based on method
        if self.config.match_method == "optimal":
            match_pairs, match_distances = optimal_match(
                data=self.data,
                distance_matrix=distance_matrix,
                treat_mask=treatment_mask,
                exact_match_cols=exact_match_cols,
                caliper=self.config.caliper,
                ratio=self.config.ratio
            )
        else:  # Default to greedy matching
            match_pairs, match_distances = greedy_match(
                data=self.data,
                distance_matrix=distance_matrix,
                treat_mask=treatment_mask,
                exact_match_cols=exact_match_cols,
                caliper=self.config.caliper,
                replace=self.config.replace,
                ratio=self.config.ratio,
                random_state=self.config.random_state
            )
        
        # Get matched indices
        matched_treat_pos = list(match_pairs.keys())
        matched_control_pos = [pos for sublist in match_pairs.values() for pos in sublist]
        
        matched_treat_indices = treatment_indices[matched_treat_pos]
        matched_control_indices = control_indices[matched_control_pos]
        matched_indices = pd.Index(list(matched_treat_indices) + list(matched_control_indices))
        
        # If treatment/control were flipped for matching, flip them back in results
        if flipped:
            # Swap treatment and control
            treatment_indices, control_indices = control_indices, treatment_indices
            matched_treat_indices, matched_control_indices = matched_control_indices, matched_treat_indices
            
            # Also need to flip the match pairs
            # In this case, match_pairs maps control->treatment, but we want treatment->control
            flipped_pairs = {}
            for c_pos, t_pos_list in match_pairs.items():
                for t_pos in t_pos_list:
                    if t_pos not in flipped_pairs:
                        flipped_pairs[t_pos] = []
                    flipped_pairs[t_pos].append(c_pos)
            match_pairs = flipped_pairs
        
        return {
            'treatment_indices': treatment_indices,
            'control_indices': control_indices,
            'matched_treatment_indices': matched_treat_indices,
            'matched_control_indices': matched_control_indices,
            'matched_indices': matched_indices,
            'match_pairs': match_pairs,
            'match_distances': match_distances
        }
    
    def _calculate_balance(self, matched_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate balance statistics.
        
        Args:
            matched_data: DataFrame with matched data
            
        Returns:
            Dictionary with balance statistics results
        """
        # Calculate balance statistics
        balance_statistics = calculate_balance_stats(
            data=self.data,
            matched_data=matched_data,
            covariates=self.config.covariates,
            treatment_col=self.config.treatment_col
        )
        
        # Calculate Rubin's rules
        rubin_statistics = calculate_rubin_rules(balance_statistics)
        
        # Calculate balance index
        balance_index = calculate_balance_index(balance_statistics)
        
        return {
            'balance_statistics': balance_statistics,
            'rubin_statistics': rubin_statistics,
            'balance_index': balance_index
        }
    
    def _estimate_effects(self, matched_data: pd.DataFrame) -> pd.DataFrame:
        """Estimate treatment effects.
        
        Args:
            matched_data: DataFrame with matched data
            
        Returns:
            DataFrame with treatment effect estimates
        """
        # Estimate treatment effects for each outcome
        effect_estimates = estimate_multiple_outcomes(
            data=matched_data,
            outcomes=self.config.outcomes,
            treatment_col=self.config.treatment_col,
            method=self.config.effect_method,
            covariates=self.config.adjustment_covariates,
            estimand=self.config.estimand,
            bootstrap_iterations=self.config.bootstrap_iterations,
            confidence_level=self.config.confidence_level,
            random_state=self.config.random_state
        )
        
        return effect_estimates
