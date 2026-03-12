"""Matcher implementation for CohortBalancer3.

This module provides the main Matcher class for performing matching, propensity score estimation,
balance assessment, and treatment effect estimation.
"""

from typing import Any

import numpy as np
import pandas as pd

from cohortbalancer3.datatypes import MatcherConfig, MatchResults

# Import from existing modules
from cohortbalancer3.matching.distances import calculate_distance_matrix
from cohortbalancer3.matching.fast_greedy import fast_greedy_match
from cohortbalancer3.matching.greedy import greedy_match
from cohortbalancer3.matching.optimal import optimal_match
from cohortbalancer3.metrics.balance import (
    calculate_balance_index,
    calculate_balance_stats,
    calculate_rubin_rules,
)
from cohortbalancer3.metrics.propensity import (
    assess_propensity_overlap,
    estimate_propensity_scores,
    trim_by_propensity,
)
from cohortbalancer3.metrics.treatment import estimate_multiple_outcomes
from cohortbalancer3.metrics.utils import get_caliper_for_matching
from cohortbalancer3.reporting import create_report
from cohortbalancer3.utils.logging import get_logger
from cohortbalancer3.validation import validate_data, validate_matcher_config

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
            exact_match_cols=self.config.exact_match_cols,
        )

        logger.info(f"Initialized Matcher with {len(data)} observations")
        logger.debug(
            f"Treatment counts: {data[self.config.treatment_col].value_counts().to_dict()}"
        )

    def match(self) -> "Matcher":
        """Perform matching according to configuration.

        This method orchestrates the entire matching process, including propensity score
        estimation, distance calculation, matching, balance assessment, and treatment
        effect estimation if requested.

        Returns:
            Self, for method chaining

        """
        logger.info(f"Starting matching with method: {self.config.match_method}")

        # Step 1: Estimate propensity scores if needed
        propensity_scores = None
        propensity_model = None
        propensity_metrics = None

        # Determine if propensity scores are required by configuration
        need_ps_for_distance = self.config.distance_method in ["propensity", "logit"]
        need_ps_for_caliper = (
            self.config.caliper_method in ["propensity", "logit"]
            and self.config.caliper_value is not None
        )
        need_ps_for_fast = self.config.match_method == "fast_greedy"

        has_user_ps_request = self.config.estimate_propensity or bool(self.config.propensity_col)

        # Always estimate when required by configuration (distance, caliper, fast_greedy),
        # and also when user explicitly requested or provided a column
        need_propensity = (
            has_user_ps_request
            or need_ps_for_caliper
            or need_ps_for_fast
            or need_ps_for_distance
        )

        if need_propensity:
            # Temporarily enable estimation if required by config but not explicitly requested
            temporarily_enabled = False
            if not has_user_ps_request:
                temporarily_enabled = True
                self.config.estimate_propensity = True
                logger.info(
                    "Estimating propensity scores because they are required by the matching configuration"
                )
            else:
                logger.info("Estimating propensity scores")

            propensity_result = self._estimate_propensity()
            propensity_scores = propensity_result.get("propensity_scores")
            propensity_model = propensity_result.get("model")
            propensity_metrics = propensity_result.get("metrics")

            # Restore original flag if we temporarily enabled estimation
            if temporarily_enabled:
                self.config.estimate_propensity = False

            # Store propensity scores as instance variable for use in _perform_matching
            self.propensity_scores = propensity_scores

        # Step 2: Determine matching direction (always match from smaller to larger group)
        flipped = self._determine_matching_direction()
        if flipped:
            logger.info(
                "Flipping treatment and control for matching (control -> treatment)"
            )

        treatment_mask = self._get_treatment_mask(flipped)

        # Step 3: Perform matching (which now includes distance calculation)
        logger.info("Performing matching")
        match_results = self._perform_matching(treatment_mask, flipped)

        # Get matched data (directly from the match_results now)
        matched_data = match_results["matched_data"]

        # For logging and ratio calculation, count unique and total units
        n_treat = (matched_data[self.config.treatment_col] == 1).sum()
        n_control = (matched_data[self.config.treatment_col] == 0).sum()
        actual_ratio = n_control / max(1, n_treat)

        # Verify matching balance
        if self.config.ratio == 1.0:
            # For 1:1 matching, unique counts should be equal
            if n_treat != n_control:
                logger.warning(
                    f"Unexpected imbalance in final 1:1 matched dataset: {n_treat} treatment and {n_control} control units"
                )
            else:
                logger.info(
                    f"Matching successful: {n_treat} treatment and {n_control} control units (ratio 1:1)"
                )
        else:
            # For many-to-one matching, check if the ratio is close to the expected ratio
            expected_ratio = self.config.ratio

            # Check if ratio is close to expected (allowing for some deviation due to constraints)
            if abs(actual_ratio - expected_ratio) > 0.2:  # Allow 0.2 deviation
                logger.warning(
                    f"Matching ratio deviation: Expected {expected_ratio:.1f}:1, achieved {actual_ratio:.2f}:1"
                )
            else:
                logger.info(
                    f"Matching successful: {n_treat} treatment and {n_control} control units (ratio {actual_ratio:.2f}:1)"
                )

        logger.info(
            f"Completed matching with {len(matched_data)} total observations in matched dataset"
        )

        # Step 5: Calculate balance statistics
        balance_statistics = None
        rubin_statistics = None
        balance_index = None

        if self.config.calculate_balance:
            logger.info("Calculating balance statistics")
            balance_result = self._calculate_balance(matched_data)
            balance_statistics = balance_result.get("balance_statistics")
            rubin_statistics = balance_result.get("rubin_statistics")
            balance_index = balance_result.get("balance_index")

        # Step 6: Estimate treatment effects if outcomes specified
        effect_estimates = None

        if self.config.outcomes:
            logger.info(
                f"Estimating treatment effects for {len(self.config.outcomes)} outcomes"
            )
            effect_estimates = self._estimate_effects(matched_data)

        # Combine all results using the enhanced MatchResults structure
        self.results = MatchResults(
            original_data=self.data,
            matched_data=matched_data,
            pairs=match_results["pairs"],
            match_groups=match_results["match_groups"],
            match_distances=match_results["match_distances"],
            distance_matrix=match_results["distance_matrix"],
            propensity_scores=propensity_scores,
            propensity_model=propensity_model,
            propensity_metrics=propensity_metrics,
            balance_statistics=balance_statistics,
            rubin_statistics=rubin_statistics,
            balance_index=balance_index,
            effect_estimates=effect_estimates,
            config=self.config,
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

    def save_results(self, directory: str) -> "Matcher":
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
        self.results.matched_data.to_csv(
            os.path.join(directory, "matched_data.csv"), index=True
        )

        # Save match pairs
        self.results.get_match_pairs().to_csv(
            os.path.join(directory, "match_pairs.csv"), index=False
        )

        # Save balance statistics if available
        if self.results.balance_statistics is not None:
            self.results.balance_statistics.to_csv(
                os.path.join(directory, "balance_statistics.csv"), index=False
            )

        # Save treatment effect estimates if available
        if self.results.effect_estimates is not None:
            self.results.effect_estimates.to_csv(
                os.path.join(directory, "effect_estimates.csv"), index=False
            )

        # Save configuration
        with open(os.path.join(directory, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)

        logger.info(f"Saved results to directory: {directory}")
        return self

    def create_report(
        self,
        method_name: str = None,
        output_dir: str = None,
        prefix: str = "",
        report_filename: str = "matching_report.html",
        export_tables_to_csv: bool = True,
        dpi: int = 300,
        max_vars_balance: int = 15,
        max_vars_dist: int = 8,
    ) -> str:
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
            max_vars_dist=max_vars_dist,
        )

        logger.info(f"Generated HTML report: {report_path}")
        return report_path

    # Private methods for implementation details
    def _validate_data(self):
        """Validate input data and configuration."""
        # This method is now replaced by the centralized validation system.
        # The validation is performed in the __init__ method.

    def _determine_matching_direction(self) -> bool:
        """Determine matching direction based on group sizes.

        Returns:
            Boolean indicating whether treatment/control should be flipped for matching

        """
        # Count treatment and control units
        n_treatment = (self.data[self.config.treatment_col] == 1).sum()
        n_control = (self.data[self.config.treatment_col] == 0).sum()

        # Only flip for 1:1 matching. When ratio > 1, flipping would invert
        # the ratio semantics (e.g., instead of each treatment getting 2 controls,
        # each control would get 2 treatments, causing duplicate controls in pairs
        # even with replace=False).
        if self.config.ratio > 1:
            return False

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

    def _estimate_propensity(self) -> dict[str, Any]:
        """Estimate propensity scores based on configuration.

        Returns:
            Dictionary with propensity model, scores, and metrics

        """
        # If propensity column is provided, use it directly
        if (
            self.config.propensity_col
            and self.config.propensity_col in self.data.columns
        ):
            propensity_scores = self.data[self.config.propensity_col].values
            treatment_mask = self.data[self.config.treatment_col] == 1

            # Calculate overlap metrics
            metrics = assess_propensity_overlap(
                data=self.data,
                propensity_col=self.config.propensity_col,
                treatment_col=self.config.treatment_col,
            )

            return {
                "propensity_scores": propensity_scores,
                "model": None,
                "metrics": metrics,
            }

        # Otherwise, estimate propensity scores
        if self.config.estimate_propensity:
            propensity_result = estimate_propensity_scores(
                data=self.data,
                treatment_col=self.config.treatment_col,
                covariates=self.config.covariates,
                model_type=self.config.propensity_model,
                model_params=self.config.model_params,
                cv=self.config.cv_folds,
                random_state=self.config.random_state,
            )

            # Trim propensity scores if requested
            if self.config.common_support_trimming:
                trimmed_data = trim_by_propensity(
                    data=self.data,
                    propensity_scores=propensity_result["propensity_scores"],
                    treatment_col=self.config.treatment_col,
                    method="common_support",
                    trim_percent=self.config.trim_threshold,
                )
                # Update data with trimmed version
                self.data = trimmed_data

                # Recalculate mask for trimmed data
                treatment_mask = self.data[self.config.treatment_col] == 1

                # Update propensity scores for trimmed data
                propensity_result["propensity_scores"] = propensity_result[
                    "propensity_scores"
                ][trimmed_data.index]

            return {
                "propensity_scores": propensity_result["propensity_scores"],
                "model": propensity_result["model"],
                "metrics": propensity_result.get("metrics", {}),
            }

        # No propensity scores
        return {"propensity_scores": None, "model": None, "metrics": {}}

    def _perform_matching(
        self, treatment_mask: np.ndarray, flipped: bool
    ) -> dict[str, Any]:
        """Perform matching according to configuration.

        This method acts as an orchestrator, preparing the necessary inputs for the
        chosen matching algorithm. For matrix-based methods ('greedy', 'optimal'),
        it computes and calipers the distance matrix. For 'fast_greedy', it delegates
        the entire process to the specialized function.

        Args:
            treatment_mask: Boolean mask for treatment units
            flipped: Whether treatment/control are flipped for matching

        Returns:
            Dictionary with matching results including pairs of participant IDs

        """
        # Get propensity scores if they were estimated
        propensity_scores = getattr(self, "propensity_scores", None)

        # Get indices for matching (which may be flipped for the algorithm)
        algorithm_treatment_indices = self.data.index[treatment_mask].tolist()
        algorithm_control_indices = self.data.index[~treatment_mask].tolist()

        # Define a variable to hold the distance matrix for the results object
        distance_matrix_for_results = None

        # --- Fast Greedy Path (Memory-Efficient) ---
        if self.config.match_method == "fast_greedy":
            if propensity_scores is None:
                raise ValueError("Propensity scores are required for 'fast_greedy' matching.")
            
            # For fast_greedy, the final caliper value must be numeric before calling
            # the matching function, as it's applied inside the loop.
            final_caliper_value = get_caliper_for_matching(
                config=self.config,
                propensity_scores=propensity_scores,
                data=self.data,
                treat_mask=treatment_mask
            )
            
            # Create a temporary config with the resolved numeric caliper value
            temp_config = self.config.__class__(**self.config.__dict__)
            temp_config.caliper_value = final_caliper_value
            
            algorithm_match_pairs, match_distances = fast_greedy_match(
                data=self.data,
                treat_mask=treatment_mask,
                propensity_scores=propensity_scores,
                config=temp_config,
            )

        # --- Matrix-Based Path ('greedy', 'optimal') ---
        else:
            # 1. Calculate the primary distance matrix
            logger.info(f"Calculating primary distance matrix with method: {self.config.distance_method}")
            
            if self.config.distance_method in ["propensity", "logit"]:
                if propensity_scores is None:
                    raise ValueError("Propensity scores are required for this distance method.")
                X_treat_primary = propensity_scores[treatment_mask].reshape(-1, 1)
                X_control_primary = propensity_scores[~treatment_mask].reshape(-1, 1)
            else:
                X_treat_primary = self.data[self.config.covariates][treatment_mask].values
                X_control_primary = self.data[self.config.covariates][~treatment_mask].values

            primary_distances = calculate_distance_matrix(
                X_treat=X_treat_primary,
                X_control=X_control_primary,
                method=self.config.distance_method,
                standardize=self.config.standardize,
                weights=np.array([self.config.weights.get(c, 1.0) for c in self.config.covariates]) if self.config.weights else None
            )

            # 2. Apply caliper if specified
            if self.config.caliper_method is not None and self.config.caliper_value is not None:
                final_caliper_value = get_caliper_for_matching(
                    config=self.config,
                    propensity_scores=propensity_scores,
                    distance_matrix=primary_distances,
                    data=self.data,
                    treat_mask=treatment_mask
                )
                
                if self.config.caliper_method == self.config.distance_method:
                    logger.info(f"Applying '{self.config.caliper_method}' caliper directly to distance matrix.")
                    primary_distances[primary_distances > final_caliper_value] = np.inf
                else:
                    logger.info(f"Applying '{self.config.caliper_method}' caliper as a mask on '{self.config.distance_method}' distances.")
                    
                    caliper_calc_method = self.config.caliper_method
                    if self.config.caliper_method in ["propensity", "logit"]:
                        if propensity_scores is None:
                            raise ValueError("Propensity scores are required for this caliper method.")
                        X_treat_caliper = propensity_scores[treatment_mask].reshape(-1, 1)
                        X_control_caliper = propensity_scores[~treatment_mask].reshape(-1, 1)
                    elif self.config.caliper_method in ["mahalanobis", "euclidean"]:
                         # Use all covariates for these distance-based caliper methods
                        X_treat_caliper = self.data[self.config.covariates][treatment_mask].values
                        X_control_caliper = self.data[self.config.covariates][~treatment_mask].values
                    else:
                        # Assume caliper_method is a column name or list of names
                        caliper_covs = [self.config.caliper_method] if isinstance(self.config.caliper_method, str) else self.config.caliper_method
                        X_treat_caliper = self.data[caliper_covs][treatment_mask].values
                        X_control_caliper = self.data[caliper_covs][~treatment_mask].values
                        caliper_calc_method = 'euclidean' # Force euclidean for this case

                    caliper_matrix = calculate_distance_matrix(
                        X_treat=X_treat_caliper,
                        X_control=X_control_caliper,
                        method=caliper_calc_method,
                        standardize=self.config.standardize,
                    )
                    
                    primary_distances[caliper_matrix > final_caliper_value] = np.inf

            # 3. Perform matching
            if self.config.match_method == "optimal":
                algorithm_match_pairs, match_distances = optimal_match(
                    data=self.data,
                    distance_matrix=primary_distances,
                    treat_mask=treatment_mask,
                    exact_match_cols=self.config.exact_match_cols,
                    ratio=self.config.ratio,
                    replace=self.config.replace,
                )
            else:  # Default to greedy matching
                algorithm_match_pairs, match_distances = greedy_match(
                    data=self.data,
                    distance_matrix=primary_distances,
                    treat_mask=treatment_mask,
                    exact_match_cols=self.config.exact_match_cols,
                    replace=self.config.replace,
                    ratio=self.config.ratio,
                    random_state=self.config.random_state,
                )

            distance_matrix_for_results = primary_distances

        # Convert algorithm match pairs to actual participant ID pairs and match groups
        pairs = []
        match_groups = {}

        # Track which units should be in matched dataset
        matched_ids = set()

        # Process matches, handling potential flipping
        for t_pos, c_pos_list in algorithm_match_pairs.items():
            if len(c_pos_list) == 0:
                continue

            # Get the actual indices
            t_idx = algorithm_treatment_indices[t_pos]
            c_indices = [algorithm_control_indices[c_pos] for c_pos in c_pos_list]

            # Handle flipping by identifying the true treatment/control status
            if flipped:
                # If we flipped for the algorithm, t_idx is actually a control unit
                # and c_indices are treatment units
                for c_idx in c_indices:
                    pairs.append((c_idx, t_idx))

                    # Add to match groups (initialized if needed)
                    if c_idx not in match_groups:
                        match_groups[c_idx] = []
                    match_groups[c_idx].append(t_idx)

                    # Add both to the matched dataset
                    matched_ids.add(c_idx)
                    matched_ids.add(t_idx)
            else:
                # Normal case - t_idx is treatment, c_indices are controls
                for c_idx in c_indices:
                    pairs.append((t_idx, c_idx))

                    # Add to match groups (initialized if needed)
                    if t_idx not in match_groups:
                        match_groups[t_idx] = []
                    match_groups[t_idx].append(c_idx)

                    # Add both to the matched dataset
                    matched_ids.add(t_idx)
                    matched_ids.add(c_idx)

        if not self.config.replace:
            # Verify no control is matched to multiple treatment units
            control_ids_in_pairs = [c_id for _, c_id in pairs]
            if len(control_ids_in_pairs) != len(set(control_ids_in_pairs)):
                from collections import Counter
                counts = Counter(control_ids_in_pairs)
                duplicates = {k: v for k, v in counts.items() if v > 1}
                logger.error(
                    f"Controls matched to multiple treatments despite replace=False: {duplicates}"
                )
                # Deduplicate: keep only the first occurrence of each control
                seen_controls = set()
                deduped_pairs = []
                for t_id, c_id in pairs:
                    if c_id not in seen_controls:
                        seen_controls.add(c_id)
                        deduped_pairs.append((t_id, c_id))
                pairs = deduped_pairs

                # Rebuild matched_ids and match_groups from deduplicated pairs
                matched_ids = set()
                match_groups = {}
                for t_id, c_id in pairs:
                    matched_ids.add(t_id)
                    matched_ids.add(c_id)
                    if t_id not in match_groups:
                        match_groups[t_id] = []
                    match_groups[t_id].append(c_id)

            # Select the rows from the original dataset
            matched_data = self.data.loc[list(matched_ids)].copy()
        else:
            # For matching with replacement, construct a new DataFrame to handle duplicate control units
            
            # Track usage count of each control to create duplicate rows with unique indices
            control_usage_count: dict[Any, int] = {}

            # Build the matched_data DataFrame with duplicated control rows while
            # keeping 'pairs' and 'match_groups' referencing original IDs
            rows = []
            new_indices = []

            # Add all treatment rows
            treatment_ids_in_pairs = set(pair[0] for pair in pairs)
            for t_id in treatment_ids_in_pairs:
                rows.append(self.data.loc[t_id].copy())
                new_indices.append(t_id)

            # Add control rows, duplicating as many times as they are used
            for _, c_id in pairs:
                usage_count = control_usage_count.get(c_id, 0)
                control_usage_count[c_id] = usage_count + 1

                unique_c_id = c_id if usage_count == 0 else f"{c_id}_dup{usage_count}"
                rows.append(self.data.loc[c_id].copy())
                new_indices.append(unique_c_id)

            # Create the new DataFrame with unique indices for all units
            matched_data = pd.DataFrame(rows, index=new_indices)

        # Verify the balance of the matched dataset
        n_treatment = (matched_data[self.config.treatment_col] == 1).sum()
        n_control = (matched_data[self.config.treatment_col] == 0).sum()

        logger.debug(
            f"Created matched dataset with {n_treatment} treatment and {n_control} control units"
        )

        # For 1:1 matching, verify we have equal counts
        if self.config.ratio == 1.0 and not self.config.replace:
            if n_treatment != n_control:
                logger.warning(
                    f"Expected equal treatment and control counts for 1:1 matching, but got "
                    f"{n_treatment} treatment and {n_control} control units"
                )

                # This is a critical check - the counts should be equal for 1:1 matching
                assert n_treatment == n_control, (
                    "Critical error in matched dataset construction"
                )

        # Check if we have any matches - if not, create an empty matched_data with the right columns
        if not pairs:
            logger.warning("No matching pairs found. Returning empty matched dataset.")
            matched_data = pd.DataFrame(columns=self.data.columns)

        return {
            "pairs": pairs,
            "match_groups": match_groups,
            "matched_indices": matched_data.index,
            "matched_data": matched_data,
            "match_distances": match_distances,
            "distance_matrix": distance_matrix_for_results,
        }

    def _calculate_balance(self, matched_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate balance statistics.

        Args:
            matched_data: DataFrame with matched data

        Returns:
            Dictionary with balance statistics results

        """
        # If the matched_data is empty (no matches found), return empty results
        if matched_data.empty:
            logger.warning("No matches found. Balance statistics cannot be calculated.")
            return {
                "balance_statistics": None,
                "rubin_statistics": None,
                "balance_index": None,
            }

        # Log debug information about treatment groups
        logger.debug(
            f"Original data treatment counts: {self.data[self.config.treatment_col].value_counts().to_dict()}"
        )
        logger.debug(
            f"Matched data treatment counts: {matched_data[self.config.treatment_col].value_counts().to_dict()}"
        )

        # Calculate balance statistics
        balance_statistics = calculate_balance_stats(
            data=self.data,
            matched_data=matched_data,
            covariates=self.config.covariates,
            treatment_col=self.config.treatment_col,
        )

        # Calculate Rubin's rules
        rubin_statistics = calculate_rubin_rules(balance_statistics)

        # Calculate balance index
        balance_index = calculate_balance_index(balance_statistics)

        # Log summary of balance results
        if balance_index:
            logger.info(
                f"Balance index: {balance_index.get('balance_index', 'N/A'):.1f}"
            )
            logger.info(
                f"Mean SMD before: {balance_index.get('mean_smd_before', 'N/A'):.3f}, "
                f"after: {balance_index.get('mean_smd_after', 'N/A'):.3f}"
            )

        return {
            "balance_statistics": balance_statistics,
            "rubin_statistics": rubin_statistics,
            "balance_index": balance_index,
        }

    def _estimate_effects(self, matched_data: pd.DataFrame) -> pd.DataFrame:
        """Estimate treatment effects.

        Args:
            matched_data: DataFrame with matched data

        Returns:
            DataFrame with treatment effect estimates

        """
        # If the matched_data is empty (no matches found), return empty results
        if matched_data.empty:
            logger.warning("No matches found. Treatment effects cannot be estimated.")
            # Create an empty DataFrame with the expected structure
            return pd.DataFrame(
                columns=[
                    "outcome",
                    "effect",
                    "std_error",
                    "t_statistic",
                    "p_value",
                    "ci_lower",
                    "ci_upper",
                    "method",
                    "estimand",
                ]
            )

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
            random_state=self.config.random_state,
        )

        return effect_estimates
