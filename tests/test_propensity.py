"""
Test suite for propensity score estimation in cohortbalancer3.metrics.propensity module.

These tests validate the functionality of propensity score estimation and evaluation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cohortbalancer3.metrics.propensity import (
    assess_common_support,
    assess_propensity_overlap,
    calculate_propensity_quality,
    estimate_propensity_scores,
    get_propensity_model,
    trim_by_propensity,
)


class TestPropensityEstimation:
    """Test suite for propensity score estimation functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for propensity score estimation testing."""
        np.random.seed(42)
        n = 100

        # Create synthetic data with known propensity model
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)

        # Treatment assignment based on X1 and X2
        # Higher values of X1 and X2 increase probability of treatment
        propensity = 1 / (1 + np.exp(-(0.5 * X1 + 0.7 * X2)))
        treatment = np.random.binomial(1, propensity)

        # Create dataframe
        data = pd.DataFrame(
            {
                "treatment": treatment,
                "X1": X1,
                "X2": X2,
                "cat_var": np.random.choice(["A", "B", "C"], size=n),
                "outcome": 3
                + 2 * treatment
                + 0.5 * X1
                + 0.3 * X2
                + np.random.normal(0, 1, n),
            }
        )

        return data

    @pytest.fixture
    def propensity_scores(self, sample_data):
        """Create propensity scores for testing."""
        # Simple propensity scores based on synthetic data
        n = len(sample_data)
        ps = np.random.uniform(0.1, 0.9, n)
        # Make treatment units have slightly higher propensity scores
        ps[sample_data["treatment"] == 1] += 0.1
        ps = np.clip(ps, 0.01, 0.99)  # Keep in (0,1) range
        return ps

    @pytest.mark.parametrize("model_type", ["logistic"])
    def test_get_propensity_model(self, model_type):
        """Test getting different types of propensity models."""
        try:
            model = get_propensity_model(model_type=model_type)
            assert model is not None
            # Check that the model has a fit method
            assert hasattr(model, "fit")
            assert hasattr(model, "predict_proba")
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_get_propensity_model_with_params(self):
        """Test getting a propensity model with custom parameters."""
        try:
            model_params = {"C": 0.5, "class_weight": "balanced"}
            model = get_propensity_model(
                model_type="logistic", model_params=model_params
            )
            assert model is not None
            # Check that the parameters were set correctly
            assert model.C == 0.5
            assert model.class_weight == "balanced"
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_get_propensity_model_with_random_state(self):
        """Test that random_state is passed correctly to models."""
        try:
            random_state = 42
            model = get_propensity_model(
                model_type="logistic", random_state=random_state
            )
            assert model.random_state == random_state
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_trim_by_propensity_common_support(self, sample_data, propensity_scores):
        """Test trimming data based on common support method."""
        # Create a copy with propensity scores
        data = sample_data.copy()

        # Perform trimming using common support method
        trimmed_data = trim_by_propensity(
            data, propensity_scores, treatment_col="treatment", method="common_support"
        )

        # Check that the result is a DataFrame
        assert isinstance(trimmed_data, pd.DataFrame)

        # Get treatment indicator from original data
        treatment = data["treatment"].values

        # Calculate common support region from original data and propensity scores
        treated_ps = propensity_scores[treatment == 1]
        control_ps = propensity_scores[treatment == 0]

        min_common = max(np.min(treated_ps), np.min(control_ps))
        max_common = min(np.max(treated_ps), np.max(control_ps))

        # Get propensity scores for the trimmed data
        ps_trimmed = propensity_scores[trimmed_data.index]

        # Check that all trimmed propensity scores are within the common support region
        # Note: We need to check each score individually because trim_by_propensity implements
        # the common support region filtering using the trimmed data's indices
        assert all(
            (score >= min_common and score <= max_common) for score in ps_trimmed
        )

    def test_trim_by_propensity_percentile(self, sample_data, propensity_scores):
        """Test trimming data based on percentile method."""
        # Create a copy with propensity scores
        data = sample_data.copy()

        trim_percent = 0.1

        # Perform trimming using percentile method
        trimmed_data = trim_by_propensity(
            data,
            propensity_scores,
            treatment_col="treatment",
            method="percentile",
            trim_percent=trim_percent,
        )

        # Check that trimmed data has fewer rows than original
        assert len(trimmed_data) <= len(data)

        # Check that percentile thresholds are respected
        ps_trimmed = propensity_scores[trimmed_data.index]
        treatment = data.loc[trimmed_data.index, "treatment"].values

        treated_ps = propensity_scores[data["treatment"] == 1]
        control_ps = propensity_scores[data["treatment"] == 0]

        treated_low = np.percentile(treated_ps, trim_percent * 100)
        treated_high = np.percentile(treated_ps, (1 - trim_percent) * 100)
        control_low = np.percentile(control_ps, trim_percent * 100)
        control_high = np.percentile(control_ps, (1 - trim_percent) * 100)

        # Check that treated units are within thresholds
        treated_in_trimmed = ps_trimmed[treatment == 1]
        assert np.all(
            (treated_in_trimmed >= treated_low) & (treated_in_trimmed <= treated_high)
        )

        # Check that control units are within thresholds
        control_in_trimmed = ps_trimmed[treatment == 0]
        assert np.all(
            (control_in_trimmed >= control_low) & (control_in_trimmed <= control_high)
        )

    def test_assess_common_support(self, sample_data, propensity_scores):
        """Test assessing common support between treatment and control propensity distributions."""
        treatment = sample_data["treatment"].values

        # Calculate common support metrics
        support_metrics = assess_common_support(
            propensity_scores=propensity_scores, treatment=treatment, bins=20
        )

        # Check that the function returns the expected keys
        expected_keys = [
            "common_support_min",
            "common_support_max",
            "overlap_coefficient",
            "hist_treated",
            "hist_control",
            "bin_edges",
        ]
        for key in expected_keys:
            assert key in support_metrics

        # Check that the common support range is correct
        treated_ps = propensity_scores[treatment == 1]
        control_ps = propensity_scores[treatment == 0]

        min_treated, max_treated = np.min(treated_ps), np.max(treated_ps)
        min_control, max_control = np.min(control_ps), np.max(control_ps)

        expected_min = max(min_treated, min_control)
        expected_max = min(max_treated, max_control)

        assert support_metrics["common_support_min"] == expected_min
        assert support_metrics["common_support_max"] == expected_max

        # Check that overlap coefficient is between 0 and 1
        assert 0 <= support_metrics["overlap_coefficient"] <= 1

    def test_assess_propensity_overlap(self, sample_data):
        """Test assessing propensity score overlap between treatment and control groups."""
        # Create a copy with propensity scores
        data = sample_data.copy()
        data["propensity"] = np.random.uniform(0.1, 0.9, len(data))

        # Calculate overlap metrics
        overlap_metrics = assess_propensity_overlap(
            data=data, propensity_col="propensity", treatment_col="treatment"
        )

        # Check that the function returns the expected keys
        expected_keys = [
            "ks_statistic",
            "ks_pvalue",
            "overlap_coefficient",
            "common_support_range",
            "treated_range",
            "control_range",
            "prop_in_common_support",
            "prop_treated_in_cs",
            "prop_control_in_cs",
        ]
        for key in expected_keys:
            assert key in overlap_metrics

        # Check that the KS statistic is between 0 and 1
        assert 0 <= overlap_metrics["ks_statistic"] <= 1

        # Check that the p-value is between 0 and 1
        assert 0 <= overlap_metrics["ks_pvalue"] <= 1

        # Check that the overlap coefficient is between 0 and 1
        assert 0 <= overlap_metrics["overlap_coefficient"] <= 1

        # Check that the common support range is a tuple of length 2
        assert isinstance(overlap_metrics["common_support_range"], tuple)
        assert len(overlap_metrics["common_support_range"]) == 2

        # Check that the proportion in common support is between 0 and 1
        assert 0 <= overlap_metrics["prop_in_common_support"] <= 1

    def test_calculate_propensity_quality(self, sample_data):
        """Test calculating propensity score quality metrics."""
        # Create a copy with propensity scores
        data = sample_data.copy()
        data["propensity"] = np.random.uniform(0.1, 0.9, len(data))

        # Create some fake matched indices
        n_matched = len(data) // 2
        matched_indices = np.random.choice(data.index, size=n_matched, replace=False)

        # Calculate propensity quality metrics
        quality_metrics = calculate_propensity_quality(
            data=data,
            propensity_col="propensity",
            treatment_col="treatment",
            matched_indices=matched_indices,
        )

        # Check that the function returns the expected keys
        expected_keys = [
            "ks_statistic_before",
            "ks_pvalue_before",
            "mean_diff_before",
            "ks_statistic_after",
            "ks_pvalue_after",
            "mean_diff_after",
            "common_support_ratio",
        ]
        for key in expected_keys:
            assert key in quality_metrics

        # Check that the KS statistic before is between 0 and 1
        assert 0 <= quality_metrics["ks_statistic_before"] <= 1

        # Check that the KS p-value before is between 0 and 1
        assert 0 <= quality_metrics["ks_pvalue_before"] <= 1

        # Check that the KS statistic after is between 0 and 1 (or NaN if no matches)
        if not np.isnan(quality_metrics["ks_statistic_after"]):
            assert 0 <= quality_metrics["ks_statistic_after"] <= 1

        # Check that the KS p-value after is between 0 and 1 (or NaN if no matches)
        if not np.isnan(quality_metrics["ks_pvalue_after"]):
            assert 0 <= quality_metrics["ks_pvalue_after"] <= 1

        # Check that the common support ratio is between 0 and 1
        assert 0 <= quality_metrics["common_support_ratio"] <= 1

    @patch("cohortbalancer3.metrics.propensity.estimate_propensity_scores_with_cv")
    def test_estimate_propensity_scores(self, mock_cv_func, sample_data):
        """Test the main propensity score estimation function with mocked CV function."""
        # Create mock return value for CV function
        mock_cv_result = {
            "propensity_scores": np.random.uniform(0.1, 0.9, len(sample_data)),
            "final_model": MagicMock(),
            "cv_results": {
                "fold_aucs": [0.7, 0.75, 0.72],
                "mean_auc": 0.72,
                "std_auc": 0.02,
            },
            "fold_models": [MagicMock(), MagicMock(), MagicMock()],
            "auc": 0.72,
        }
        mock_cv_func.return_value = mock_cv_result

        # Call the function with mocked CV
        try:
            result = estimate_propensity_scores(
                data=sample_data, treatment_col="treatment", covariates=["X1", "X2"]
            )

            # Check that the function returns the expected keys
            expected_keys = [
                "propensity_scores",
                "model",
                "cv_results",
                "model_type",
                "auc",
                "calibration",
            ]
            for key in expected_keys:
                assert key in result

            # Check that the AUC is from the CV result
            assert result["auc"] == mock_cv_result["auc"]

            # Check that the function called the CV function with the right arguments
            mock_cv_func.assert_called_once()

        except ImportError:
            pytest.skip("scikit-learn not installed")
