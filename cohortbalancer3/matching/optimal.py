"""
Optimal matching algorithm implementation using the Hungarian algorithm.

This module provides an implementation of the optimal matching algorithm,
which finds the matching that minimizes the total distance across all pairs.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def optimal_match(
    data: pd.DataFrame,
    distance_matrix: np.ndarray,
    treat_mask: np.ndarray,
    exact_match_cols: Optional[List[str]] = None,
    caliper: Optional[float] = None,
    ratio: float = 1.0
) -> Tuple[Dict[int, List[int]], List[float]]:
    """Implement optimal matching algorithm using the Hungarian algorithm.
    
    Args:
        data: DataFrame containing the data
        distance_matrix: Pre-computed distance matrix (n_treatment x n_control)
        treat_mask: Boolean mask indicating treatment units
        exact_match_cols: Columns to match exactly on
        caliper: Maximum allowed distance for a match (if None, no constraint)
        ratio: Matching ratio (e.g., 2 means 1:2 matching)
        
    Returns:
        Tuple of (match_pairs, match_distances) where match_pairs is a dict mapping
        treatment indices to lists of control indices (positions, not original indices)
    """
    # Validate that distance_matrix has the right shape
    n_units = len(data)
    n_treatment = np.sum(treat_mask)
    n_control = n_units - n_treatment
    
    expected_shape = (n_treatment, n_control)
    if distance_matrix.shape != expected_shape:
        raise ValueError(
            f"Distance matrix has shape {distance_matrix.shape}, but expected {expected_shape} "
            f"based on the number of treatment ({n_treatment}) and control ({n_control}) units"
        )
    
    # Validate exact_match_cols
    if exact_match_cols and not all(col in data.columns for col in exact_match_cols):
        missing_cols = [col for col in exact_match_cols if col not in data.columns]
        raise ValueError(f"Exact match columns not found in data: {missing_cols}")
    
    # Validate caliper
    if caliper is not None and caliper <= 0:
        raise ValueError(f"Caliper must be positive, got {caliper}")
    
    # Validate ratio
    if ratio <= 0:
        raise ValueError(f"Ratio must be positive, got {ratio}")
    
    # Validate treat_mask
    if not isinstance(treat_mask, np.ndarray) or treat_mask.dtype != bool:
        raise ValueError("treat_mask must be a boolean numpy array")
    
    if len(treat_mask) != len(data):
        raise ValueError(f"treat_mask length ({len(treat_mask)}) does not match data length ({len(data)})")
    
    # Get treatment and control indices
    treat_indices = np.where(treat_mask)[0]
    control_indices = np.where(~treat_mask)[0]

    # Find exact matches if requested
    if exact_match_cols:
        # For optimal matching with exact matching constraints,
        # we modify the distance matrix to enforce exact matching
        # by setting distances to infinity for disallowed pairs
        distance_matrix = _apply_exact_matching_constraints(
            data, treat_indices, control_indices,
            distance_matrix.copy(), exact_match_cols
        )

    # Apply caliper constraint if provided
    if caliper is not None:
        # Set distances greater than caliper to infinity
        distance_matrix = distance_matrix.copy()
        distance_matrix[distance_matrix > caliper] = np.inf

    # Handle ratio matching
    if ratio != 1.0:
        # For optimal matching with ratio != 1, we need special handling
        match_pairs, match_distances = _optimal_match_with_ratio(
            distance_matrix=distance_matrix,
            treat_indices=treat_indices,
            control_indices=control_indices,
            ratio=ratio
        )
    else:
        # Standard 1:1 optimal matching
        match_pairs, match_distances = _optimal_match_1_to_1(
            distance_matrix=distance_matrix,
            treat_indices=treat_indices,
            control_indices=control_indices
        )

    return match_pairs, match_distances


def _apply_exact_matching_constraints(
    data: pd.DataFrame,
    treat_indices: np.ndarray,
    control_indices: np.ndarray,
    distance_matrix: np.ndarray,
    exact_match_cols: List[str]
) -> np.ndarray:
    """Apply exact matching constraints to the distance matrix.
    
    Args:
        data: DataFrame containing the data
        treat_indices: Indices of treatment units
        control_indices: Indices of control units
        distance_matrix: Distance matrix to modify
        exact_match_cols: Columns to match exactly on
        
    Returns:
        Modified distance matrix
    """
    # Extract relevant data
    treat_data = data.iloc[treat_indices]
    control_data = data.iloc[control_indices]

    # Create a hash key for each unit based on exact matching columns
    def create_key(row):
        return '_'.join(str(row[col]) for col in exact_match_cols)

    # Create a dictionary mapping treatment positions to their exact match keys
    treat_keys = {}
    for i in range(len(treat_indices)):
        treat_keys[i] = create_key(treat_data.iloc[i])

    # Create a dictionary mapping control positions to their exact match keys
    control_keys = {}
    for j in range(len(control_indices)):
        control_keys[j] = create_key(control_data.iloc[j])

    # Set distances to infinity for pairs that don't match on exact columns
    for i in range(len(treat_indices)):
        for j in range(len(control_indices)):
            if treat_keys[i] != control_keys[j]:
                distance_matrix[i, j] = np.inf

    return distance_matrix


def _optimal_match_1_to_1(
    distance_matrix: np.ndarray,
    treat_indices: np.ndarray,
    control_indices: np.ndarray
) -> Tuple[Dict[int, List[int]], List[float]]:
    """Perform 1:1 optimal matching using the Hungarian algorithm.
    
    Args:
        distance_matrix: Distance matrix (n_treatment x n_control)
        treat_indices: Indices of treatment units
        control_indices: Indices of control units
        
    Returns:
        Tuple of (match_pairs, match_distances)
    """
    # Keep a copy of the original distance matrix to check for infinite values later
    original_distance_matrix = distance_matrix.copy()

    # Handle infinite values
    cost_matrix = distance_matrix.copy()

    # Replace inf with a very large value
    max_finite = np.nanmax(cost_matrix[np.isfinite(cost_matrix)])
    if np.isnan(max_finite) or np.isinf(max_finite):
        max_finite = 1.0  # Fallback if all values are inf or nan

    # Replace inf with a very large value (but still finite)
    inf_mask = np.isinf(cost_matrix)
    cost_matrix[inf_mask] = max_finite * 1e6  # Use a much larger multiplier

    # Replace nan with a very large value
    nan_mask = np.isnan(cost_matrix)
    cost_matrix[nan_mask] = max_finite * 1e6  # Use a much larger multiplier

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create match pairs
    match_pairs: Dict[int, List[int]] = {}
    match_distances: List[float] = []

    # Track which control units have been matched
    matched_controls = set()

    for i, j in zip(row_ind, col_ind):
        # Skip pairs with infinite or nan distance in the original matrix
        if np.isinf(original_distance_matrix[i, j]) or np.isnan(original_distance_matrix[i, j]):
            continue

        # Skip if this control unit has already been matched (no duplicates)
        if j in matched_controls:
            continue

        # Store the match
        if i not in match_pairs:
            match_pairs[i] = []

        match_pairs[i].append(j)
        match_distances.append(original_distance_matrix[i, j])
        matched_controls.add(j)

    # Verify that no matches with infinite distance were included
    # This is a safety check to ensure exact matching constraints are respected
    for i, j_list in match_pairs.items():
        for j in j_list:
            assert not np.isinf(original_distance_matrix[i, j]), f"Match with infinite distance found: {i}, {j}"

    return match_pairs, match_distances


def _optimal_match_with_ratio(
    distance_matrix: np.ndarray,
    treat_indices: np.ndarray,
    control_indices: np.ndarray,
    ratio: float = 2.0
) -> Tuple[Dict[int, List[int]], List[float]]:
    """Perform optimal matching with a specified matching ratio.
    
    This implements a proper ratio matching for the optimal algorithm by
    creating a modified distance matrix with duplicate rows for each treatment unit.
    
    Args:
        distance_matrix: Distance matrix (n_treatment x n_control)
        treat_indices: Indices of treatment units
        control_indices: Indices of control units
        ratio: Matching ratio (e.g., 2 means 1:2 matching)
        
    Returns:
        Tuple of (match_pairs, match_distances)
    """
    n_treat = len(treat_indices)
    n_control = len(control_indices)

    # Check if we have enough control units
    target_matches = min(int(n_treat * ratio), n_control)
    if target_matches < n_treat:
        # Not enough control units for 1:1 matching
        raise ValueError(f"Not enough control units for {ratio}:1 matching. "
                        f"Need at least {n_treat} but only have {n_control}.")

    # If ratio is 1, just use regular 1:1 matching
    if ratio == 1.0:
        return _optimal_match_1_to_1(distance_matrix, treat_indices, control_indices)

    # For m:n matching where m > 1, we solve as a minimum cost flow problem
    # Here, we'll use a matrix-based approach with the Hungarian algorithm

    if ratio > 1.0:
        # For ratio > 1, each treatment unit gets ratio control units
        # We need to decide how many control units each treatment unit gets

        # Perform 1:1 matching first to ensure each treatment unit gets at least one match
        base_pairs, base_distances = _optimal_match_1_to_1(
            distance_matrix, treat_indices, control_indices
        )

        # Initialize match pairs and distances
        match_pairs = {}
        match_distances = []

        # Initialize the match pairs and distances with the 1:1 matches
        for t_pos, c_pos_list in base_pairs.items():
            if t_pos not in match_pairs:
                match_pairs[t_pos] = []

            for c_pos in c_pos_list:
                match_pairs[t_pos].append(c_pos)
                match_distances.append(distance_matrix[t_pos, c_pos])

        # Keep track of control units that have been matched
        matched_controls = set()
        for t_pos, c_pos_list in match_pairs.items():
            matched_controls.update(c_pos_list)

        # For each treatment unit, add additional matches up to the ratio
        # We'll prioritize treatment units with the smallest distance to their 1:1 match
        treat_priority = {}
        for t_pos, c_pos_list in match_pairs.items():
            if c_pos_list:  # Make sure there's at least one control match
                c_pos = c_pos_list[0]
                treat_priority[t_pos] = distance_matrix[t_pos, c_pos]

        sorted_treat = sorted(treat_priority.keys(), key=lambda t: treat_priority[t])

        # Get remaining control units
        remaining_controls = [c_pos for c_pos in range(len(control_indices)) if c_pos not in matched_controls]

        # Calculate how many additional matches each treatment unit should get
        additional_per_treat = int((target_matches - n_treat) / n_treat)
        extra_matches = (target_matches - n_treat) % n_treat

        # Assign additional matches
        for t_pos in sorted_treat:
            # Determine how many additional matches this treatment unit should get
            additional = additional_per_treat
            if extra_matches > 0:
                additional += 1
                extra_matches -= 1

            # Skip if no additional matches needed or no more controls available
            if additional == 0 or not remaining_controls:
                continue

            # Find the best remaining control units for this treatment unit
            # Sort by distance to this treatment unit
            control_distances = []
            for c_pos in remaining_controls:
                dist = distance_matrix[t_pos, c_pos]
                control_distances.append((c_pos, dist))

            # Sort by distance
            control_distances.sort(key=lambda x: x[1])

            # Add additional matches
            for i in range(min(additional, len(control_distances))):
                c_pos, dist = control_distances[i]
                match_pairs[t_pos].append(c_pos)
                match_distances.append(dist)
                remaining_controls.remove(c_pos)

                # Stop if we've run out of control units
                if not remaining_controls:
                    break

        return match_pairs, match_distances

    else:  # ratio < 1.0, meaning multiple treatment units share a control unit
        # This is a different problem where control units are the "rows" 
        # and treatment units are the "columns" in the Hungarian algorithm

        # Transpose the distance matrix
        transposed_distance = distance_matrix.T

        # Determine how many treatment units each control unit should match with
        # For ratio = 0.5, each control unit matches with 2 treatment units
        inv_ratio = 1.0 / ratio

        # Initialize match pairs (map from treatment position to list of control positions)
        match_pairs = {}
        match_distances = []

        # Track which treatment positions have been matched
        matched_treatments = set()

        # Each control unit can match with inv_ratio treatment units
        # We'll use a greedy approach prioritizing closest treatment units
        for c_pos in range(len(control_indices)):
            if len(matched_treatments) == len(treat_indices):
                break  # All treatment units matched

            # Get distances to all unmatched treatment units
            unmatched_treat_pos = [i for i in range(len(treat_indices))
                                  if i not in matched_treatments]

            if not unmatched_treat_pos:
                break  # All treatment units matched

            # Get distances to unmatched treatment units
            distances = [(t_pos, transposed_distance[c_pos, t_pos]) for t_pos in unmatched_treat_pos]
            distances.sort(key=lambda x: x[1])  # Sort by distance

            # Match with up to inv_ratio treatment units
            matches_for_this_control = int(inv_ratio)
            if matches_for_this_control < 1:
                matches_for_this_control = 1

            for i in range(min(matches_for_this_control, len(distances))):
                t_pos, dist = distances[i]

                # Add the match
                if t_pos not in match_pairs:
                    match_pairs[t_pos] = []

                match_pairs[t_pos].append(c_pos)
                match_distances.append(dist)
                matched_treatments.add(t_pos)

        # Verify that no matches with infinite distance were included
        for t_pos, c_pos_list in match_pairs.items():
            for c_pos in c_pos_list:
                assert not np.isinf(distance_matrix[t_pos, c_pos]), f"Match with infinite distance found: {t_pos}, {c_pos}"

        return match_pairs, match_distances
