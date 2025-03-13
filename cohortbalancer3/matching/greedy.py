"""
Greedy matching algorithm implementation.

This module provides an implementation of the greedy matching algorithm,
which finds pairs by iteratively selecting the closest unmatched control unit
for each treatment unit.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
    # Get treatment and control indices (positions)
    treat_indices = np.where(treat_mask)[0]
    control_indices = np.where(~treat_mask)[0]

    # Find exact matches if requested
    exact_match_groups = None
    if exact_match_cols:
        exact_match_groups = _create_exact_match_groups(
            data, treat_indices, control_indices, exact_match_cols
        )

    # Perform matching
    match_pairs, match_distances = _greedy_match_with_constraints(
        distance_matrix=distance_matrix,
        treat_indices=treat_indices,
        control_indices=control_indices,
        exact_match_groups=exact_match_groups,
        caliper=caliper,
        replace=replace,
        ratio=ratio,
        random_state=random_state
    )

    return match_pairs, match_distances


def _greedy_match_with_constraints(
    distance_matrix: np.ndarray,
    treat_indices: np.ndarray,
    control_indices: np.ndarray,
    exact_match_groups: Optional[Dict[int, List[int]]] = None,
    caliper: Optional[float] = None,
    replace: bool = False,
    ratio: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[Dict[int, List[int]], List[float]]:
    """Implement greedy matching with exact matching, caliper, and ratio constraints.
    
    Args:
        distance_matrix: Distance matrix (n_treatment x n_control)
        treat_indices: Indices of treatment units
        control_indices: Indices of control units
        exact_match_groups: Dictionary mapping treatment indices to potential control matches
                           based on exact matching criteria
        caliper: Maximum allowed distance for a match (if None, no constraint)
        replace: Whether to allow replacement in matching
        ratio: Matching ratio (e.g., 2 means 1:2 matching)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (match_pairs, match_distances)
    """
    rng = np.random.RandomState(random_state)

    # Create copies to avoid modifying the inputs
    distance_matrix = distance_matrix.copy()

    # Find indices with finite distances (handling NaNs)
    finite_mask = np.isfinite(distance_matrix)

    # Apply exact matching constraints if provided
    if exact_match_groups is not None:
        # Create a mapping from original indices to positions in the arrays
        treat_idx_to_pos = {idx: pos for pos, idx in enumerate(treat_indices)}
        control_idx_to_pos = {idx: pos for pos, idx in enumerate(control_indices)}

        for t_idx, allowed_c_indices in exact_match_groups.items():
            # Get the position of this treatment unit in the treat_indices array
            if t_idx in treat_idx_to_pos:
                t_pos = treat_idx_to_pos[t_idx]

                # Create a mask for disallowed control units (all disallowed by default)
                disallowed_mask = np.ones(len(control_indices), dtype=bool)

                # Mark allowed control units as not disallowed
                for c_idx in allowed_c_indices:
                    if c_idx in control_idx_to_pos:
                        c_pos = control_idx_to_pos[c_idx]
                        disallowed_mask[c_pos] = False

                # Set distances to infinity for disallowed pairs
                distance_matrix[t_pos, disallowed_mask] = np.inf
                finite_mask[t_pos, disallowed_mask] = False

    # Apply caliper constraint if provided
    if caliper is not None:
        distance_matrix[distance_matrix > caliper] = np.inf
        finite_mask[distance_matrix > caliper] = False

    # Dictionary to store matches
    match_pairs: Dict[int, List[int]] = {i: [] for i in range(len(treat_indices))}
    match_distances: List[float] = []

    # Set for tracking available control units (positions in the control array)
    available_controls = set(range(len(control_indices)))

    # Calculate number of matches needed per treatment unit
    matches_per_treatment = max(1, int(ratio))

    # Sort treatment units by number of potential matches (recommended for better matching)
    treat_order = np.arange(len(treat_indices))
    if exact_match_groups is not None:
        # Sort by number of potential matches within exact match groups
        n_potential_matches = np.sum(finite_mask, axis=1)
        treat_order = np.argsort(n_potential_matches)

    # Randomly permute order if random_state is provided
    if random_state is not None:
        rng.shuffle(treat_order)

    # Main matching loop
    for t_pos in treat_order:
        t_idx = treat_indices[t_pos]

        # Skip if no potential matches
        if not np.any(finite_mask[t_pos]):
            continue

        # Find matches for this treatment unit
        for _ in range(matches_per_treatment):
            if not replace and not available_controls:
                # No more control units available
                break

            # Get distances to control units
            distances = distance_matrix[t_pos]

            if not replace:
                # Set distances for already matched control units to infinity
                for c_pos in set(range(len(control_indices))) - available_controls:
                    distances[c_pos] = np.inf

            # Find the closest control unit
            if np.all(np.isinf(distances)):
                # No valid matches for this treatment unit
                break

            # Find control unit with minimum distance
            c_pos = np.argmin(distances)
            c_idx = control_indices[c_pos]

            match_pairs[t_pos].append(c_pos)
            match_distances.append(distances[c_pos])

            if not replace:
                # Remove control unit from available pool
                available_controls.remove(c_pos)

    return match_pairs, match_distances


def _create_exact_match_groups(
    data: pd.DataFrame,
    treat_indices: np.ndarray,
    control_indices: np.ndarray,
    exact_match_cols: List[str]
) -> Dict[int, List[int]]:
    """Create groups for exact matching.
    
    Args:
        data: DataFrame containing the data
        treat_indices: Indices of treatment units
        control_indices: Indices of control units
        exact_match_cols: Columns to match exactly on
        
    Returns:
        Dictionary mapping treatment indices to lists of potential control matches
    """
    # Extract relevant data
    treat_data = data.iloc[treat_indices]
    control_data = data.iloc[control_indices]

    # Create concatenated string key for each unit
    def create_key(row):
        return '_'.join(str(row[col]) for col in exact_match_cols)

    treat_keys = treat_data.apply(create_key, axis=1)
    control_keys = control_data.apply(create_key, axis=1)

    # Create mapping from keys to control indices
    key_to_controls = {}
    for i, key in enumerate(control_keys):
        if key not in key_to_controls:
            key_to_controls[key] = []
        key_to_controls[key].append(control_indices[i])

    # Create mapping from treatment indices to potential control matches
    exact_match_groups = {}
    for i, key in enumerate(treat_keys):
        t_idx = treat_indices[i]
        exact_match_groups[t_idx] = key_to_controls.get(key, [])

    return exact_match_groups
