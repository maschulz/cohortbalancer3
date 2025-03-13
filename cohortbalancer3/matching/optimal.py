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
        Tuple of (match_pairs, match_distances)
    """
    # Get treatment and control indices
    treat_indices = np.where(treat_mask)[0]
    control_indices = np.where(~treat_mask)[0]
    
    # Create working copy of distance matrix
    distances = distance_matrix.copy()
    
    # Apply exact matching constraints if needed
    if exact_match_cols:
        distances = _apply_exact_matching(data, treat_indices, control_indices, 
                                       distances, exact_match_cols)
    
    # Apply caliper if specified
    if caliper is not None:
        distances[distances > caliper] = np.inf
    
    # Handle ratio matching
    if ratio != 1.0:
        if ratio > 1.0:
            # For ratio > 1, duplicate control units
            n_copies = int(ratio)
            distances_expanded = np.tile(distances, (1, n_copies))
        else:
            # For ratio < 1, duplicate treatment units
            n_copies = int(1.0 / ratio)
            distances_expanded = np.tile(distances, (n_copies, 1))
    else:
        distances_expanded = distances
    
    # Replace inf with large finite value for linear_sum_assignment
    max_finite = np.nanmax(distances_expanded[~np.isinf(distances_expanded)])
    if np.isnan(max_finite):
        max_finite = 1.0
    cost_matrix = distances_expanded.copy()
    cost_matrix[np.isinf(cost_matrix) | np.isnan(cost_matrix)] = max_finite * 1e6
    
    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Convert back to original indices if ratio != 1
    if ratio != 1.0:
        if ratio > 1.0:
            row_ind_orig = row_ind
            col_ind_orig = col_ind % distances.shape[1]
        else:
            row_ind_orig = row_ind % distances.shape[0]
            col_ind_orig = col_ind
    else:
        row_ind_orig = row_ind
        col_ind_orig = col_ind
    
    # Create match pairs dictionary
    match_pairs: Dict[int, List[int]] = {}
    match_distances: List[float] = []
    
    # Process matches
    for i, j in zip(row_ind_orig, col_ind_orig):
        # Skip matches with infinite distance in original matrix
        if np.isinf(distances[i, j]) or np.isnan(distances[i, j]):
            continue
            
        if i not in match_pairs:
            match_pairs[i] = []
        
        # Only add if not already matched (handles duplicates from ratio matching)
        if j not in match_pairs[i]:
            match_pairs[i].append(j)
            match_distances.append(distances[i, j])
    
    return match_pairs, match_distances


def _apply_exact_matching(
    data: pd.DataFrame,
    treat_indices: np.ndarray,
    control_indices: np.ndarray,
    distances: np.ndarray,
    exact_match_cols: List[str]
) -> np.ndarray:
    """Apply exact matching constraints efficiently using pandas operations."""
    # Extract matching columns
    treat_data = data.iloc[treat_indices][exact_match_cols]
    control_data = data.iloc[control_indices][exact_match_cols]
    
    # Create hash strings for comparison
    treat_keys = treat_data.astype(str).agg('_'.join, axis=1)
    control_keys = control_data.astype(str).agg('_'.join, axis=1)
    
    # Create match matrix (n_treat x n_control)
    match_matrix = np.zeros((len(treat_indices), len(control_indices)), dtype=bool)
    
    # Vectorized exact matching
    for i, t_key in enumerate(treat_keys):
        match_matrix[i] = (control_keys == t_key)
    
    # Set distances to infinity where exact matches don't exist
    distances[~match_matrix] = np.inf
    
    return distances
