"""Optimal matching algorithm implementation using the Hungarian algorithm.

This module provides an implementation of the optimal matching algorithm,
which finds the matching that minimizes the total distance across all pairs.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from cohortbalancer3.utils.logging import get_logger

# Create a logger for this module
logger = get_logger(__name__)


def optimal_match(
    data: pd.DataFrame,
    distance_matrix: np.ndarray,
    treat_mask: np.ndarray,
    exact_match_cols: list[str] | None = None,
    caliper: float | None = None,
    ratio: float = 1.0,
) -> tuple[dict[int, list[int]], list[float]]:
    """Implement optimal matching algorithm using the Hungarian algorithm.

    The algorithm takes a distance matrix between treatment and control units and
    finds the optimal matching that minimizes the total distance. Indices in the
    returned dictionary are positions in the arrays of treatment and control units,
    not original dataframe indices. The Matcher class will translate these to
    participant IDs.

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
    logger.info("Starting optimal matching")
    logger.debug(f"Distance matrix shape: {distance_matrix.shape}")
    logger.debug(
        f"Treatment units: {np.sum(treat_mask)}, Control units: {np.sum(~treat_mask)}"
    )

    # Get treatment and control indices
    treat_indices = np.where(treat_mask)[0]
    control_indices = np.where(~treat_mask)[0]

    # Create working copy of distance matrix
    distances = distance_matrix.copy()

    # Apply exact matching constraints if needed
    if exact_match_cols:
        logger.debug(f"Applying exact matching on columns: {exact_match_cols}")
        distances = _apply_exact_matching(
            data, treat_indices, control_indices, distances, exact_match_cols
        )
        logger.debug(
            f"After exact matching, {np.sum(~np.isinf(distances))} potential matches remain"
        )

    # Apply caliper if specified
    if caliper is not None:
        logger.debug(f"Applying caliper: {caliper:.4f}")
        n_before = np.sum(~np.isinf(distances))
        distances[distances > caliper] = np.inf
        n_after = np.sum(~np.isinf(distances))
        logger.debug(f"Caliper removed {n_before - n_after} potential matches")

    # Handle ratio matching
    if ratio != 1.0:
        logger.debug(f"Implementing {ratio}:1 matching ratio")
        if ratio > 1.0:
            # For ratio > 1, duplicate control units
            n_copies = int(ratio)
            logger.debug(f"Duplicating control units {n_copies} times")
            distances_expanded = np.tile(distances, (1, n_copies))
        else:
            # For ratio < 1, duplicate treatment units
            n_copies = int(1.0 / ratio)
            logger.debug(f"Duplicating treatment units {n_copies} times")
            distances_expanded = np.tile(distances, (n_copies, 1))
    else:
        distances_expanded = distances

    logger.debug(f"Expanded distance matrix shape: {distances_expanded.shape}")

    # Replace inf with large finite value for linear_sum_assignment
    max_finite = np.nanmax(distances_expanded[~np.isinf(distances_expanded)])
    if np.isnan(max_finite):
        max_finite = 1.0
    cost_matrix = distances_expanded.copy()
    cost_matrix[np.isinf(cost_matrix) | np.isnan(cost_matrix)] = max_finite * 1e6

    # Find optimal assignment
    logger.debug("Running Hungarian algorithm for optimal assignment")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    logger.debug(f"Found {len(row_ind)} potential assignments")

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
    match_pairs: dict[int, list[int]] = {}
    match_distances: list[float] = []

    # Process matches
    for i, j in zip(row_ind_orig, col_ind_orig, strict=False):
        # Skip matches with infinite distance in original matrix
        if np.isinf(distances[i, j]) or np.isnan(distances[i, j]):
            continue

        if i not in match_pairs:
            match_pairs[i] = []

        # Only add if not already matched (handles duplicates from ratio matching)
        if j not in match_pairs[i]:
            match_pairs[i].append(j)
            match_distances.append(distances[i, j])

    logger.info(
        f"Optimal matching complete: {len(match_pairs)} treatment units matched"
    )
    if match_distances:
        logger.debug(
            f"Match distances - min: {min(match_distances):.4f}, "
            f"mean: {np.mean(match_distances):.4f}, "
            f"max: {max(match_distances):.4f}"
        )

    return match_pairs, match_distances


def _apply_exact_matching(
    data: pd.DataFrame,
    treat_indices: np.ndarray,
    control_indices: np.ndarray,
    distances: np.ndarray,
    exact_match_cols: list[str],
) -> np.ndarray:
    """Apply exact matching constraints efficiently using pandas operations."""
    logger.debug(f"Applying exact matching on {len(exact_match_cols)} columns")

    # Extract matching columns
    treat_data = data.iloc[treat_indices][exact_match_cols]
    control_data = data.iloc[control_indices][exact_match_cols]

    # Create hash strings for comparison
    treat_keys = treat_data.astype(str).agg("_".join, axis=1)
    control_keys = control_data.astype(str).agg("_".join, axis=1)

    # Create match matrix (n_treat x n_control)
    match_matrix = np.zeros((len(treat_indices), len(control_indices)), dtype=bool)

    # Vectorized exact matching
    unique_treat_keys = set(treat_keys)
    logger.debug(
        f"Found {len(unique_treat_keys)} unique combinations in treatment group"
    )

    for i, t_key in enumerate(treat_keys):
        match_matrix[i] = control_keys == t_key

    # Set distances to infinity where exact matches don't exist
    n_before = np.sum(~np.isinf(distances))
    distances[~match_matrix] = np.inf
    n_after = np.sum(~np.isinf(distances))

    logger.debug(f"Exact matching removed {n_before - n_after} potential matches")
    logger.debug(
        f"Treatment units with at least one match: {np.sum(np.any(~np.isinf(distances), axis=1))}"
    )

    return distances
