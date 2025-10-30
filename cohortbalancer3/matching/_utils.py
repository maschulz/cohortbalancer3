"""Internal utility functions for matching algorithms."""

import numpy as np
import pandas as pd
from cohortbalancer3.utils.logging import get_logger

logger = get_logger(__name__)

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