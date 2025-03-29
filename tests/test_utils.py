"""Tests for utility functions in the cohortbalancer3 metrics utilities.

This module tests the behavior of utility functions used across
different metrics calculations, with a focus on caliper calculation.
"""

import numpy as np
import pytest
from scipy.special import logit

from cohortbalancer3.metrics.utils import get_caliper_for_matching


class TestCaliperCalculation:
    """Test suite for caliper calculation."""

    def test_direct_caliper_value(self):
        """Test that direct numeric caliper values are passed through."""
        # Test with an integer
        assert get_caliper_for_matching(5) == 5.0

        # Test with a float
        assert get_caliper_for_matching(0.2) == 0.2

    def test_none_caliper(self):
        """Test that None caliper returns None (no caliper)."""
        assert get_caliper_for_matching(None) is None

    def test_auto_caliper_propensity(self):
        """Test auto caliper calculation for propensity score method."""
        # Create sample propensity scores
        np.random.seed(42)
        propensity_scores = np.random.beta(
            2, 5, 100
        )  # Beta distribution gives values between 0 and 1

        # Calculate expected value: caliper_scale * SD of logit(propensity)
        ps_clipped = np.clip(propensity_scores, 0.001, 0.999)
        logit_ps = logit(ps_clipped)
        logit_ps_sd = np.std(logit_ps)
        expected_caliper = 0.2 * logit_ps_sd  # Default scale is 0.2

        # Calculate actual value
        actual_caliper = get_caliper_for_matching(
            "auto", propensity_scores=propensity_scores, method="propensity"
        )

        # Check that the values match
        assert np.isclose(actual_caliper, expected_caliper)

        # Test with custom caliper scale
        custom_scale = 0.5
        expected_custom_caliper = custom_scale * logit_ps_sd
        actual_custom_caliper = get_caliper_for_matching(
            "auto",
            propensity_scores=propensity_scores,
            method="propensity",
            caliper_scale=custom_scale,
        )
        assert np.isclose(actual_custom_caliper, expected_custom_caliper)

    def test_auto_caliper_logit(self):
        """Test auto caliper calculation for logit method."""
        # Create sample propensity scores
        np.random.seed(42)
        propensity_scores = np.random.beta(2, 5, 100)

        # Should behave the same as propensity method
        propensity_caliper = get_caliper_for_matching(
            "auto", propensity_scores=propensity_scores, method="propensity"
        )

        logit_caliper = get_caliper_for_matching(
            "auto", propensity_scores=propensity_scores, method="logit"
        )

        assert np.isclose(propensity_caliper, logit_caliper)

    def test_auto_caliper_mahalanobis(self):
        """Test auto caliper calculation for Mahalanobis method."""
        # Create sample distance matrix
        np.random.seed(42)
        distance_matrix = np.random.gamma(
            2, 0.5, (20, 30)
        )  # Gamma gives positive values

        # Expected caliper: 90th percentile of distances (default)
        expected_caliper = np.percentile(distance_matrix, 90)

        # Calculate actual value
        actual_caliper = get_caliper_for_matching(
            "auto", distance_matrix=distance_matrix, method="mahalanobis"
        )

        # Check that the values match
        assert np.isclose(actual_caliper, expected_caliper)

        # Test with custom percentile
        custom_percentile = 75
        expected_custom_caliper = np.percentile(distance_matrix, custom_percentile)
        actual_custom_caliper = get_caliper_for_matching(
            "auto",
            distance_matrix=distance_matrix,
            method="mahalanobis",
            percentile=custom_percentile,
        )
        assert np.isclose(actual_custom_caliper, expected_custom_caliper)

    def test_auto_caliper_euclidean(self):
        """Test auto caliper calculation for Euclidean method."""
        # Create sample distance matrix
        np.random.seed(42)
        distance_matrix = np.random.gamma(
            2, 0.5, (20, 30)
        )  # Gamma gives positive values

        # Euclidean should now use percentile of distances (same as Mahalanobis)
        expected_caliper = np.percentile(
            distance_matrix, 90
        )  # Default is 90th percentile

        # Calculate actual value
        actual_caliper = get_caliper_for_matching(
            "auto", distance_matrix=distance_matrix, method="euclidean"
        )

        # Check that the values match
        assert np.isclose(actual_caliper, expected_caliper)

        # Test with custom percentile
        custom_percentile = 75
        expected_custom_caliper = np.percentile(distance_matrix, custom_percentile)
        actual_custom_caliper = get_caliper_for_matching(
            "auto",
            distance_matrix=distance_matrix,
            method="euclidean",
            percentile=custom_percentile,
        )
        assert np.isclose(actual_custom_caliper, expected_custom_caliper)

    def test_missing_required_data(self):
        """Test that appropriate errors are raised when required data is missing."""
        # Missing propensity scores for propensity method
        with pytest.raises(ValueError, match="propensity scores are required"):
            get_caliper_for_matching("auto", method="propensity")

        # Missing distance matrix for non-propensity methods
        with pytest.raises(ValueError, match="distance matrix is required"):
            get_caliper_for_matching("auto", method="euclidean")

    def test_invalid_caliper_specification(self):
        """Test that invalid caliper specifications raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid caliper specification"):
            get_caliper_for_matching("invalid")

    def test_no_finite_distances(self):
        """Test handling of distance matrix with no finite values."""
        # Create distance matrix with only NaN values
        distance_matrix = np.full((5, 5), np.nan)

        with pytest.raises(ValueError, match="no finite distances in matrix"):
            get_caliper_for_matching(
                "auto", distance_matrix=distance_matrix, method="euclidean"
            )
