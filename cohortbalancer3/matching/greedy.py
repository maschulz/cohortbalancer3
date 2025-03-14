"""
Greedy matching algorithm implementation using numpy's efficient operations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from cohortbalancer3.utils.logging import get_logger

logger = get_logger(__name__)

def greedy_match(
    data: pd.DataFrame,
    distance_matrix: np.ndarray,
    treat_mask: np.ndarray,
    exact_match_cols: Optional[List[str]] = None,
    caliper: Optional[float] = None,
    replace: bool = False,
    ratio: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[Dict[int, List[int]], List[float]]:
    """Implement greedy matching algorithm.
    
    Args:
        data: DataFrame containing the data
        distance_matrix: Pre-computed distance matrix (n_treatment x n_control)
        treat_mask: Boolean mask indicating treatment units
        exact_match_cols: Columns to match exactly on
        caliper: Maximum allowed distance for a match (if None, no constraint)
        replace: Whether to allow replacement in matching
        ratio: Matching ratio (e.g., 2 means 1:2 matching)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (match_pairs, match_distances)
    """
    logger.debug(f"GREEDY MATCHING: Ratio = {ratio}")
    
    logger.info("Starting greedy matching")
    logger.debug(f"Distance matrix shape: {distance_matrix.shape}")
    logger.debug(f"Treatment units: {np.sum(treat_mask)}, Control units: {np.sum(~treat_mask)}")
    logger.debug(f"Matching with replacement: {replace}, ratio: {ratio}")
    
    # Get treatment and control indices
    treat_indices = np.where(treat_mask)[0]
    control_indices = np.where(~treat_mask)[0]
    
    logger.debug(f"Treatment indices count = {len(treat_indices)}, control indices count = {len(control_indices)}")
    
    # Create working copy of distance matrix
    distances = distance_matrix.copy()
    
    # Apply exact matching constraints if needed
    if exact_match_cols:
        logger.debug(f"Applying exact matching on columns: {exact_match_cols}")
        distances = _apply_exact_matching(data, treat_indices, control_indices, 
                                       distances, exact_match_cols)
        logger.debug(f"After exact matching, {np.sum(~np.isinf(distances))} potential matches remain")
    
    # Apply caliper if specified
    if caliper is not None:
        logger.debug(f"Applying caliper: {caliper:.4f}")
        n_before = np.sum(~np.isinf(distances))
        distances[distances > caliper] = np.inf
        n_after = np.sum(~np.isinf(distances))
        logger.debug(f"Caliper removed {n_before - n_after} potential matches")
    
    # Initialize match storage
    n_treat = len(treat_indices)
    matches_per_unit = max(1, int(ratio))
    logger.debug(f"Attempting to find {matches_per_unit} matches per treatment unit")
    
    match_pairs: Dict[int, List[int]] = {i: [] for i in range(n_treat)}
    match_distances: List[float] = []
    
    # Set random state if provided
    if random_state is not None:
        logger.debug(f"Using random state: {random_state}")
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random
    
    # Track available control units if not replacing
    if not replace:
        available_mask = np.ones(len(control_indices), dtype=bool)
        logger.debug(f"Initially {np.sum(available_mask)} control units available")
    
    # Sort treatment units by number of potential matches (helps with exact matching)
    n_potential_matches = np.sum(~np.isinf(distances), axis=1)
    treat_order = np.argsort(n_potential_matches)
    logger.debug(f"Sorted treatment units by potential matches (min: {n_potential_matches.min()}, "
                f"max: {n_potential_matches.max()})")
    
    # Randomly permute order if requested
    if random_state is not None:
        logger.debug("Randomly permuting treatment unit order")
        rng.shuffle(treat_order)
    
    # Main matching loop
    n_matched_units = 0
    n_total_matches = 0
    
    logger.debug("Starting main matching loop")
    for t_pos in treat_order:
        # Get distances for this treatment unit
        t_distances = distances[t_pos].copy()
        
        # Skip if no valid matches
        if np.all(np.isinf(t_distances)):
            continue
        
        # Find matches for this treatment unit
        matches_found = 0
        for match_idx in range(matches_per_unit):
            if not replace:
                # Mask out unavailable controls
                t_distances[~available_mask] = np.inf
            
            # Find best remaining match
            if np.all(np.isinf(t_distances)):
                logger.debug(f"No more valid matches for treatment unit {t_pos} at match_idx {match_idx}")
                break
                
            c_pos = np.argmin(t_distances)
            match_dist = t_distances[c_pos]
            
            # Store match
            match_pairs[t_pos].append(c_pos)
            match_distances.append(match_dist)
            matches_found += 1
            
            if not replace:
                # Mark control as used
                available_mask[c_pos] = False
                logger.debug(f"Marked control unit {c_pos} as used, {np.sum(available_mask)} remaining")
            
            # Mark this control as used for this iteration
            t_distances[c_pos] = np.inf
        
        if matches_found > 0:
            n_matched_units += 1
            n_total_matches += matches_found
            logger.debug(f"Found {matches_found} matches for treatment unit {t_pos}")
    
    logger.debug(f"Final matches: {n_matched_units}/{n_treat} treatment units matched with {n_total_matches} total matches")
    logger.debug(f"Average matches per unit: {n_total_matches/max(1, n_matched_units):.2f}")
    
    # Count how many treatment units got the full ratio of matches
    full_ratio_count = sum(1 for controls in match_pairs.values() if len(controls) == matches_per_unit)
    logger.debug(f"Treatment units with full {matches_per_unit} matches: {full_ratio_count}/{n_matched_units}")
    
    # Log match counts distribution
    match_counts = {t_idx: len(controls) for t_idx, controls in match_pairs.items() if len(controls) > 0}
    logger.debug(f"Match counts distribution: {match_counts}")
    
    logger.info(f"Greedy matching complete: {n_matched_units}/{n_treat} treatment units matched")
    logger.info(f"Total matches: {n_total_matches}, average: {n_total_matches/max(1, n_matched_units):.2f} per matched unit")
    
    if match_distances:
        logger.debug(f"Match distances - min: {min(match_distances):.4f}, "
                    f"mean: {np.mean(match_distances):.4f}, "
                    f"max: {max(match_distances):.4f}")
    
    return match_pairs, match_distances

def _apply_exact_matching(
    data: pd.DataFrame,
    treat_indices: np.ndarray,
    control_indices: np.ndarray,
    distances: np.ndarray,
    exact_match_cols: List[str]
) -> np.ndarray:
    """Apply exact matching constraints efficiently using pandas operations."""
    logger.debug(f"Applying exact matching on {len(exact_match_cols)} columns")
    
    # Extract matching columns
    treat_data = data.iloc[treat_indices][exact_match_cols]
    control_data = data.iloc[control_indices][exact_match_cols]
    
    # Create hash strings for comparison
    treat_keys = treat_data.astype(str).agg('_'.join, axis=1)
    control_keys = control_data.astype(str).agg('_'.join, axis=1)
    
    # Create match matrix (n_treat x n_control)
    match_matrix = np.zeros((len(treat_indices), len(control_indices)), dtype=bool)
    
    # Vectorized exact matching
    unique_treat_keys = set(treat_keys)
    unique_control_keys = set(control_keys)
    logger.debug(f"Found {len(unique_treat_keys)} unique combinations in treatment group")
    logger.debug(f"Found {len(unique_control_keys)} unique combinations in control group")
    logger.debug(f"Overlap: {len(unique_treat_keys.intersection(unique_control_keys))} combinations")
    
    for i, t_key in enumerate(treat_keys):
        match_matrix[i] = (control_keys == t_key)
    
    # Set distances to infinity where exact matches don't exist
    n_before = np.sum(~np.isinf(distances))
    distances[~match_matrix] = np.inf
    n_after = np.sum(~np.isinf(distances))
    
    logger.debug(f"Exact matching removed {n_before - n_after} potential matches")
    logger.debug(f"Treatment units with at least one match: {np.sum(np.any(~np.isinf(distances), axis=1))}")
    
    return distances
