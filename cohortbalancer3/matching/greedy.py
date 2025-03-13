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
    
    # Initialize match storage
    n_treat = len(treat_indices)
    matches_per_unit = max(1, int(ratio))
    match_pairs: Dict[int, List[int]] = {i: [] for i in range(n_treat)}
    match_distances: List[float] = []
    
    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    
    # Track available control units if not replacing
    if not replace:
        available_mask = np.ones(len(control_indices), dtype=bool)
    
    # Sort treatment units by number of potential matches (helps with exact matching)
    n_potential_matches = np.sum(~np.isinf(distances), axis=1)
    treat_order = np.argsort(n_potential_matches)
    
    # Randomly permute order if requested
    if random_state is not None:
        rng.shuffle(treat_order)
    
    # Main matching loop
    for t_pos in treat_order:
        # Get distances for this treatment unit
        t_distances = distances[t_pos].copy()
        
        # Skip if no valid matches
        if np.all(np.isinf(t_distances)):
            continue
        
        # Find matches for this treatment unit
        for _ in range(matches_per_unit):
            if not replace:
                # Mask out unavailable controls
                t_distances[~available_mask] = np.inf
            
            # Find best remaining match
            if np.all(np.isinf(t_distances)):
                break
                
            c_pos = np.argmin(t_distances)
            match_dist = t_distances[c_pos]
            
            # Store match
            match_pairs[t_pos].append(c_pos)
            match_distances.append(match_dist)
            
            if not replace:
                # Mark control as used
                available_mask[c_pos] = False
            
            # Mark this control as used for this iteration
            t_distances[c_pos] = np.inf
    
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
