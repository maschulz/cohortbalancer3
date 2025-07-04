"""Tests for utility functions in the cohortbalancer3 metrics utilities.

This module tests the behavior of utility functions used across
different metrics calculations, with a focus on caliper calculation.
"""

import numpy as np
import pytest
from scipy.special import logit
import pandas as pd

from cohortbalancer3.datatypes import MatcherConfig
from cohortbalancer3.metrics.utils import get_caliper_for_matching

# Mock data for testing
@pytest.fixture
def mock_config():
    """Provides a basic MatcherConfig instance."""
    return MatcherConfig(
        treatment_col='treatment',
        covariates=['age', 'income']
    )

class TestCaliperCalculation:
    """Tests for the get_caliper_for_matching utility function."""

    def test_direct_caliper_value(self, mock_config):
        """Test that direct numeric caliper values are passed through."""
        mock_config.caliper_value = 5
        assert get_caliper_for_matching(mock_config) == 5.0
        
        mock_config.caliper_value = 0.25
        assert get_caliper_for_matching(mock_config) == 0.25

    def test_none_caliper(self, mock_config):
        """Test that None caliper returns None (no caliper)."""
        mock_config.caliper_method = None
        assert get_caliper_for_matching(mock_config) is None
        
        mock_config.caliper_method = 'propensity'
        mock_config.caliper_value = None
        assert get_caliper_for_matching(mock_config) is None

    def test_auto_caliper_propensity(self, mock_config):
        """Test auto caliper calculation for propensity score method."""
        np.random.seed(42)
        propensity_scores = np.random.beta(2, 5, 100)
        
        mock_config.caliper_method = "propensity"
        mock_config.caliper_value = "auto"
        mock_config.caliper_scale = 0.2

        ps_clipped = np.clip(propensity_scores, 1e-6, 1 - 1e-6)
        logit_ps = logit(ps_clipped)
        expected_caliper = 0.2 * np.std(logit_ps)

        actual_caliper = get_caliper_for_matching(
            config=mock_config, propensity_scores=propensity_scores
        )
        assert np.isclose(actual_caliper, expected_caliper)

    def test_auto_caliper_mahalanobis(self, mock_config):
        """Test auto caliper calculation for Mahalanobis method using Chi-squared distribution."""
        from scipy.stats import chi2

        mock_config.caliper_method = "mahalanobis"
        mock_config.caliper_value = "auto"
        mock_config.caliper_scale = 0.05 # This is now interpreted as a p-value
        mock_config.covariates = ['age', 'income', 'bmi'] # k=3

        k = len(mock_config.covariates)
        p_value = mock_config.caliper_scale
        
        # The threshold is the sqrt of the critical value of the chi2 distribution
        critical_value = chi2.ppf(1 - p_value, df=k)
        expected_caliper = np.sqrt(critical_value)
        
        actual_caliper = get_caliper_for_matching(config=mock_config)
        assert np.isclose(actual_caliper, expected_caliper)

    def test_auto_caliper_covariate(self, mock_config):
        """Test auto caliper for a specific covariate."""
        np.random.seed(42)
        data = pd.DataFrame({
            'treatment': [1]*50 + [0]*50,
            'age': np.random.normal(50, 5, 100)
        })
        treat_mask = (data['treatment'] == 1)

        mock_config.caliper_method = "age"
        mock_config.caliper_value = "auto"
        mock_config.caliper_scale = 0.25
        
        treat_vals = data.loc[treat_mask, 'age']
        control_vals = data.loc[~treat_mask, 'age']
        pooled_std = np.sqrt((np.var(treat_vals, ddof=1) + np.var(control_vals, ddof=1)) / 2)
        expected_caliper = 0.25 * pooled_std

        actual_caliper = get_caliper_for_matching(
            config=mock_config, data=data, treat_mask=treat_mask
        )
        assert np.isclose(actual_caliper, expected_caliper)

    def test_missing_required_data(self, mock_config):
        """Test that appropriate errors are raised when required data is missing."""
        mock_config.caliper_value = "auto"
        
        mock_config.caliper_method = "propensity"
        with pytest.raises(ValueError, match="Propensity scores are required"):
            get_caliper_for_matching(mock_config)

        # This test is no longer valid as the mahalanobis auto caliper does not require a distance matrix
        # mock_config.caliper_method = "mahalanobis"
        # with pytest.raises(ValueError, match="Distance matrix required"):
        #     get_caliper_for_matching(mock_config)
            
        mock_config.caliper_method = "age"
        with pytest.raises(ValueError, match="Column 'age' for caliper not found"):
            get_caliper_for_matching(mock_config)

    def test_invalid_caliper_specification(self, mock_config):
        """Test that invalid caliper specifications raise appropriate errors."""
        mock_config.caliper_value = "invalid_string"
        with pytest.raises(ValueError, match="Invalid caliper_value specification"):
            get_caliper_for_matching(mock_config)
            
    def test_no_finite_distances(self, mock_config):
        """Test handling of distance matrix with no finite values."""
        distance_matrix = np.full((5, 5), np.inf)
        mock_config.caliper_method = "mahalanobis"
        mock_config.caliper_value = "auto"
        
        # This test is no longer applicable to the Chi-squared method which doesn't use the matrix
        # with pytest.raises(ValueError, match="No finite distances in matrix"):
        #     get_caliper_for_matching(config=mock_config, distance_matrix=distance_matrix)
        pass
