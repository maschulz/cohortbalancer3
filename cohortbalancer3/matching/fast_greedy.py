"""Fast, memory-efficient greedy matching for large datasets."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.special import logit

from cohortbalancer3.datatypes import MatcherConfig
from cohortbalancer3.matching.distances import calculate_distance_matrix
from cohortbalancer3.metrics.utils import get_caliper_for_matching
from cohortbalancer3.utils.logging import get_logger

logger = get_logger(__name__)


def fast_greedy_match(
    data: pd.DataFrame,
    treat_mask: np.ndarray,
    propensity_scores: np.ndarray,
    config: "MatcherConfig",
) -> tuple[dict[int, list[int]], list[float]]:
    """Implement a memory-efficient greedy matching algorithm for large datasets.

    This algorithm avoids creating a full N x M distance matrix. Instead, it iterates
    through each treatment unit, finds a small subset of candidate control units,
    and then computes distances only for those candidates before making a match.

    The initial candidate pool is selected using a caliper on the **logit-transformed**
    propensity scores. This is a performance optimization (`fast_prefilter_caliper_scale`)
    that significantly reduces the search space.

    If the `distance_method` is 'propensity' or 'logit', the distance is calculated
    purely on the propensity score difference. Otherwise, a multivariate distance
    (e.g., 'mahalanobis') is calculated on the covariates for the candidate pool.

    The user-defined caliper (`config.caliper_method`) is applied *after* the
    primary distance calculation on the final candidate pool.

    Args:
        data: DataFrame containing the data.
        treat_mask: Boolean mask indicating treatment units.
        propensity_scores: Array of propensity scores for all units in `data`.
        config: MatcherConfig object with all matching parameters.

    Returns:
        Tuple of (match_pairs, match_distances). Match pairs are dictionaries
        mapping the positional index of treatment units to a list of positional
        indices of matched control units.

    """
    logger.info("Starting fast greedy matching (memory-efficient)")
    if propensity_scores is None:
        raise ValueError("Propensity scores are required for fast_greedy_match.")

    # Get treatment and control indices from the original dataframe
    treat_indices = np.where(treat_mask)[0]
    control_indices = np.where(~treat_mask)[0]
    n_treat = len(treat_indices)
    n_control = len(control_indices)

    logger.debug(f"Treatment units: {n_treat}, Control units: {n_control}")
    logger.debug(f"Matching with replacement: {config.replace}, ratio: {config.ratio}")

    # The pre-filter is a performance optimization based on logit(propensity score).
    # It creates a candidate pool to avoid calculating the full distance matrix.
    ps_clipped = np.clip(propensity_scores, 1e-6, 1 - 1e-6)
    search_scores = logit(ps_clipped)
    pre_filter_caliper = np.std(search_scores) * config.fast_prefilter_caliper_scale
            
    logger.info(f"Using pre-filtering caliper value: {pre_filter_caliper:.4f} on 'logit' scale.")

    treat_ps_search = search_scores[treat_mask]
    control_ps_search = search_scores[~treat_mask]
    
    # Keep raw and logit propensity scores for distance/caliper calculations
    treat_ps_raw = propensity_scores[treat_mask]
    control_ps_raw = propensity_scores[~treat_mask]
    treat_ps_logit = search_scores[treat_mask]
    control_ps_logit = search_scores[~treat_mask]
    
    # --- Pre-extract all data to numpy for performance ---
    logger.debug("Extracting data to numpy arrays for performance...")
    X_covariates = data[config.covariates].values if config.covariates else None
    X_treat_covariates = X_covariates[treat_mask] if X_covariates is not None else None
    X_control_covariates = X_covariates[~treat_mask] if X_covariates is not None else None
    
    # Extract data for caliper if method is a column or list of columns
    caliper_data = None
    if config.caliper_method and config.caliper_method not in ["propensity", "logit", "mahalanobis", "euclidean"]:
        caliper_covs = [config.caliper_method] if isinstance(config.caliper_method, str) else config.caliper_method
        caliper_data = data[caliper_covs].values
        
    X_treat_caliper_data = caliper_data[treat_mask] if caliper_data is not None else None
    X_control_caliper_data = caliper_data[~treat_mask] if caliper_data is not None else None
    
    # Extract data for exact matching
    exact_match_data = data[config.exact_match_cols].values if config.exact_match_cols else None
    treat_exact_values = exact_match_data[treat_mask] if exact_match_data is not None else None
    control_exact_values = exact_match_data[~treat_mask] if exact_match_data is not None else None

    # --- Match Initialization ---
    matches_per_unit = max(1, int(config.ratio))
    match_pairs: dict[int, list[int]] = {i: [] for i in range(n_treat)}
    match_distances: list[float] = []

    if not config.replace:
        available_mask = np.ones(n_control, dtype=bool)

    # --- Main Matching Loop ---
    n_matched_units = 0
    n_total_matches = 0

    for t_pos, t_idx in enumerate(tqdm(treat_indices, desc="Fast Greedy Matching")):
        # 1. Find candidates using the internal propensity pre-filtering caliper
        candidate_mask = np.abs(control_ps_search - treat_ps_search[t_pos]) <= pre_filter_caliper
        if not config.replace:
            candidate_mask &= available_mask

        candidate_indices = np.where(candidate_mask)[0]
        if len(candidate_indices) == 0:
            continue

        # 2. Calculate primary distances for the candidate pool ONLY
        if config.distance_method == "propensity":
            X_treat_single = treat_ps_raw[t_pos].reshape(1, 1)
            X_control_candidates = control_ps_raw[candidate_indices].reshape(-1, 1)
        elif config.distance_method == "logit":
            X_treat_single = treat_ps_logit[t_pos].reshape(1, 1)
            X_control_candidates = control_ps_logit[candidate_indices].reshape(-1, 1)
        else:
            # Use covariates for distance calculation
            X_treat_single = X_treat_covariates[t_pos].reshape(1, -1)
            X_control_candidates = X_control_covariates[candidate_indices]

        dist_vector = calculate_distance_matrix(
            X_treat=X_treat_single,
            X_control=X_control_candidates,
            method=config.distance_method,
            standardize=config.standardize,
            weights=np.array([config.weights.get(c, 1.0) for c in config.covariates]) if config.weights else None,
        ).ravel()

        # 3. Apply the user-defined caliper to the primary distances
        if config.caliper_method is not None and isinstance(config.caliper_value, (int, float)):
            # If caliper method is different, calculate a separate caliper vector
            if config.caliper_method != config.distance_method:
                logger.debug(f"Applying '{config.caliper_method}' caliper as a mask on '{config.distance_method}' distances for candidates.")
                
                caliper_calc_method = config.caliper_method
                if config.caliper_method == "propensity":
                    X_treat_caliper = treat_ps_raw[t_pos].reshape(1, 1)
                    X_control_caliper_candidates = control_ps_raw[candidate_indices].reshape(-1, 1)
                elif config.caliper_method == "logit":
                    X_treat_caliper = treat_ps_logit[t_pos].reshape(1, 1)
                    X_control_caliper_candidates = control_ps_logit[candidate_indices].reshape(-1, 1)
                elif config.caliper_method in ["mahalanobis", "euclidean"]:
                    # Multivariate distance caliper on covariates
                    X_treat_caliper = X_treat_covariates[t_pos].reshape(1, -1)
                    X_control_caliper_candidates = X_control_covariates[candidate_indices]
                else:
                    # Assumes caliper method is a column name or list of names
                    X_treat_caliper = X_treat_caliper_data[t_pos].reshape(1, -1)
                    X_control_caliper_candidates = X_control_caliper_data[candidate_indices]
                    caliper_calc_method = 'euclidean' # Force euclidean for this case

                caliper_vector = calculate_distance_matrix(
                    X_treat=X_treat_caliper,
                    X_control=X_control_caliper_candidates,
                    method=caliper_calc_method,
                    standardize=config.standardize,
                ).ravel()

                # Apply caliper mask
                dist_vector[caliper_vector > config.caliper_value] = np.inf
            else:
                # If methods are the same, apply caliper directly
                # For Mahalanobis, the caliper and distance are on the same sqrt scale
                dist_vector[dist_vector > config.caliper_value] = np.inf

        # 4. Apply exact matching on candidates if needed
        if config.exact_match_cols:
            treat_exact_vals = treat_exact_values[t_pos]
            control_exact_vals = control_exact_values[candidate_indices]
            exact_match_mask = (control_exact_vals == treat_exact_vals).all(axis=1)
            dist_vector[~exact_match_mask] = np.inf

        # 5. Greedily find best matches from the final candidate pool
        matches_found_for_unit = 0
        sorted_dist_indices = np.argsort(dist_vector)

        for sorted_idx in sorted_dist_indices:
            if matches_found_for_unit >= matches_per_unit:
                break
            
            dist = dist_vector[sorted_idx]
            if np.isinf(dist):
                continue # No more valid candidates

            # Get the original position in the control group
            c_pos = candidate_indices[sorted_idx]
            
            if not config.replace and not available_mask[c_pos]:
                continue
            
            match_pairs[t_pos].append(c_pos)
            match_distances.append(dist)
            matches_found_for_unit += 1

            if not config.replace:
                available_mask[c_pos] = False

        if matches_found_for_unit > 0:
            n_matched_units += 1
            n_total_matches += matches_found_for_unit
    
    logger.info(f"Fast greedy matching complete: {n_matched_units}/{n_treat} treatment units matched")
    logger.info(f"Total matches: {n_total_matches}, average: {n_total_matches / max(1, n_matched_units):.2f} per matched unit")

    return match_pairs, match_distances 