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
from cohortbalancer3.utils.logging import get_logger
from cohortbalancer3.metrics.utils import get_caliper_for_matching
from cohortbalancer3.reporting import create_report


# Create a logger for this module
logger = get_logger(__name__)


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
    
        logger.info(f"Initialized Matcher with {len(data)} observations")
        logger.debug(f"Treatment counts: {data[self.config.treatment_col].value_counts().to_dict()}")
    
    def match(self) -> 'Matcher':
        """Perform matching according to configuration.
        
        Returns:
            Self, for method chaining
        """
        logger.info(f"Starting matching with method: {self.config.match_method}")
        
        # Step 1: Estimate propensity scores if needed
        propensity_scores = None
        propensity_model = None
        propensity_metrics = None
        
        if self.config.estimate_propensity or self.config.propensity_col:
            logger.info("Estimating propensity scores")
            propensity_result = self._estimate_propensity()
            propensity_scores = propensity_result.get('propensity_scores')
            propensity_model = propensity_result.get('model')
            propensity_metrics = propensity_result.get('metrics')
            
            # Store propensity scores as instance variable for use in _perform_matching
            self.propensity_scores = propensity_scores
        
        # Step 2: Determine matching direction (always match from smaller to larger group)
        flipped = self._determine_matching_direction()
        if flipped:
            logger.info("Flipping treatment and control for matching (control -> treatment)")
        
        treatment_mask = self._get_treatment_mask(flipped)
        
        # Step 3: Calculate distance matrix
        logger.info(f"Calculating distance matrix with method: {self.config.distance_method}")
        distance_matrix = self._calculate_distance_matrix(propensity_scores, treatment_mask)
        
        # Step 4: Perform matching
        logger.info("Performing matching")
        match_results = self._perform_matching(distance_matrix, treatment_mask, flipped)
        
        # No special case handling needed anymore - the redesigned _perform_matching ensures correct indices by construction
        
        # Get matched data
        matched_data = self.data.loc[match_results['matched_indices']].copy()
        
        # Verify matching balance for 1:1 matching (as a sanity check)
        if self.config.ratio == 1.0:
            n_treat = (matched_data[self.config.treatment_col] == 1).sum()
            n_control = (matched_data[self.config.treatment_col] == 0).sum()
            if n_treat != n_control:
                logger.warning(f"Unexpected imbalance in final 1:1 matched dataset: {n_treat} treatment and {n_control} control units")
            else:
                logger.info(f"Matching successful: {n_treat} treatment and {n_control} control units (ratio 1:1)")
        else:
            n_treat = (matched_data[self.config.treatment_col] == 1).sum()
            n_control = (matched_data[self.config.treatment_col] == 0).sum()
            actual_ratio = n_control / max(1, n_treat)
            logger.info(f"Matching successful: {n_treat} treatment and {n_control} control units (ratio {actual_ratio:.2f}:1)")
        
        logger.info(f"Completed matching with {len(matched_data)} matched observations")
        
        # Step 5: Calculate balance statistics
        balance_statistics = None
        rubin_statistics = None
        balance_index = None
        
        if self.config.calculate_balance:
            logger.info("Calculating balance statistics")
            balance_result = self._calculate_balance(matched_data)
            balance_statistics = balance_result.get('balance_statistics')
            rubin_statistics = balance_result.get('rubin_statistics')
            balance_index = balance_result.get('balance_index')
        
        # Step 6: Estimate treatment effects if outcomes specified
        effect_estimates = None
        
        if self.config.outcomes:
            logger.info(f"Estimating treatment effects for {len(self.config.outcomes)} outcomes")
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
        
        logger.info(f"Saved results to directory: {directory}")
        return self
    
    def create_report(self, 
                     method_name: str = None,
                     output_dir: str = None,
                     prefix: str = "",
                     report_filename: str = "matching_report.html",
                     export_tables_to_csv: bool = True,
                     dpi: int = 300,
                     max_vars_balance: int = 15,
                     max_vars_dist: int = 8) -> str:
        """Create a comprehensive HTML report of matching results.
        
        This method generates visualizations, exports data tables, and creates an HTML report
        summarizing the matching results. It provides a convenient way to generate publication-quality
        output from matching analyses.
        
        Args:
            method_name: Name of the matching method used (e.g., "Greedy Matching with Propensity Scores")
                        If None, will be inferred from the results configuration
            output_dir: Directory where reports will be saved
                       If None, a temporary directory will be created
            prefix: Prefix to add to filenames
            report_filename: Filename for the HTML report
            export_tables_to_csv: Whether to export data tables to CSV files
            dpi: DPI for saved images
            max_vars_balance: Maximum number of variables to show in balance plot
            max_vars_dist: Maximum number of variables to show in distribution plots
            
        Returns:
            Path to the generated HTML report
            
        Raises:
            ValueError: If no matching has been performed yet
        """
        if self.results is None:
            raise ValueError("No matching has been performed yet.")
        
        report_path = create_report(
            results=self.results,
            method_name=method_name,
            output_dir=output_dir,
            prefix=prefix,
            report_filename=report_filename,
            export_tables_to_csv=export_tables_to_csv,
            dpi=dpi,
            max_vars_balance=max_vars_balance,
            max_vars_dist=max_vars_dist
        )
        
        logger.info(f"Generated HTML report: {report_path}")
        return report_path
    
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
            mask = self.data[self.config.treatment_col] == 0
        else:
            # Otherwise, use actual treatment units (1) as "from" group
            mask = self.data[self.config.treatment_col] == 1
        
        # Ensure the result is a boolean numpy array
        return np.array(mask, dtype=bool)
    
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
        # Get original (un-flipped) treatment and control indices
        original_treatment_indices = self.data.index[self.data[self.config.treatment_col] == 1]
        original_control_indices = self.data.index[self.data[self.config.treatment_col] == 0]
        
        # Get indices for matching (which may be flipped for the algorithm)
        algorithm_treatment_indices = self.data.index[treatment_mask]
        algorithm_control_indices = self.data.index[~treatment_mask]
        
        # Get exact match columns if provided
        exact_match_cols = self.config.exact_match_cols if self.config.exact_match_cols else None
        
        # Determine caliper value (handling 'auto' and numeric values)
        propensity_scores = None
        if hasattr(self, 'propensity_scores') and self.propensity_scores is not None:
            propensity_scores = self.propensity_scores
        
        caliper = get_caliper_for_matching(
            config_caliper=self.config.caliper,
            propensity_scores=propensity_scores,
            distance_matrix=distance_matrix,
            method=self.config.distance_method,
            caliper_scale=self.config.caliper_scale
        )
        
        if caliper is not None:
            logger.info(f"Using caliper: {caliper:.4f} for {self.config.distance_method} distance")
        
        # Perform matching based on method
        if self.config.match_method == "optimal":
            algorithm_match_pairs, match_distances = optimal_match(
                data=self.data,
                distance_matrix=distance_matrix,
                treat_mask=treatment_mask,
                exact_match_cols=exact_match_cols,
                caliper=caliper,
                ratio=self.config.ratio
            )
        else:  # Default to greedy matching
            algorithm_match_pairs, match_distances = greedy_match(
                data=self.data,
                distance_matrix=distance_matrix,
                treat_mask=treatment_mask,
                exact_match_cols=exact_match_cols,
                caliper=caliper,
                replace=self.config.replace,
                ratio=self.config.ratio,
                random_state=self.config.random_state
            )
        
        # Convert algorithm match pairs to actual data indices
        # This step creates an explicit mapping from algorithm positions to data indices
        match_pairs_indices = []
        
        for t_pos, c_pos_list in algorithm_match_pairs.items():
            t_idx = algorithm_treatment_indices[t_pos]
            for c_pos in c_pos_list:
                c_idx = algorithm_control_indices[c_pos]
                match_pairs_indices.append((t_idx, c_idx))
        
        # Now create the match_pairs object in terms of original treatment/control definitions
        # This ensures that regardless of internal flipping, the final match pairs are correct
        final_match_pairs = {}
        
        for t_idx, c_idx in match_pairs_indices:
            # Determine if these indices are actually treatment or control in the original data
            is_t_treatment = t_idx in original_treatment_indices
            is_c_control = c_idx in original_control_indices
            
            # Set the correct treatment and control indices based on the original data
            true_treatment_idx = t_idx if is_t_treatment else c_idx
            true_control_idx = c_idx if is_c_control else t_idx
            
            # Find treatment and control positions for the result dictionary
            treatment_pos = np.where(original_treatment_indices == true_treatment_idx)[0][0]
            control_pos = np.where(original_control_indices == true_control_idx)[0][0]
            
            # Add to final match pairs
            if treatment_pos not in final_match_pairs:
                final_match_pairs[treatment_pos] = []
            final_match_pairs[treatment_pos].append(control_pos)
        
        # Get unique treatment and control indices from the match pairs
        matched_treatment_indices = []
        matched_control_indices = []
        
        for t_pos, c_pos_list in final_match_pairs.items():
            matched_treatment_indices.append(original_treatment_indices[t_pos])
            for c_pos in c_pos_list:
                matched_control_indices.append(original_control_indices[c_pos])
        
        # Create unique lists
        matched_treatment_indices = pd.Index(list(set(matched_treatment_indices)))
        matched_control_indices = pd.Index(list(set(matched_control_indices)))
        
        # Create the final matched indices
        matched_indices = pd.Index(list(matched_treatment_indices) + list(matched_control_indices))
        
        # Log matching information
        logger.debug(f"Matched {len(matched_treatment_indices)} treatment units with {len(matched_control_indices)} control units")
        logger.debug(f"Matching resulted in {len(final_match_pairs)} treatment-control pairs")
        
        if self.config.ratio == 1.0 and len(matched_treatment_indices) != len(matched_control_indices):
            logger.warning(f"Unexpected imbalance in 1:1 matching: {len(matched_treatment_indices)} treatment and {len(matched_control_indices)} control units")
        
        return {
            'treatment_indices': original_treatment_indices,
            'control_indices': original_control_indices,
            'matched_treatment_indices': matched_treatment_indices,
            'matched_control_indices': matched_control_indices,
            'matched_indices': matched_indices,
            'match_pairs': final_match_pairs,
            'match_distances': match_distances
        }
    
    def _calculate_balance(self, matched_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate balance statistics.
        
        Args:
            matched_data: DataFrame with matched data
            
        Returns:
            Dictionary with balance statistics results
        """
        # Log debug information about treatment groups
        logger.debug(f"Original data treatment counts: {self.data[self.config.treatment_col].value_counts().to_dict()}")
        logger.debug(f"Matched data treatment counts: {matched_data[self.config.treatment_col].value_counts().to_dict()}")
        
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
        
        # Log summary of balance results
        if balance_index:
            logger.info(f"Balance index: {balance_index.get('balance_index', 'N/A'):.1f}")
            logger.info(f"Mean SMD before: {balance_index.get('mean_smd_before', 'N/A'):.3f}, " 
                      f"after: {balance_index.get('mean_smd_after', 'N/A'):.3f}")
        
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
