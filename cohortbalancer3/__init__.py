"""
CohortBalancer3: A Python package for propensity score matching and treatment effect estimation.

This package provides tools for performing propensity score matching,
assessing covariate balance, and estimating treatment effects.
"""

__version__ = "0.1.0"

from cohortbalancer3.datatypes import MatcherConfig, MatchResults
from cohortbalancer3.matcher import Matcher
from cohortbalancer3.validation import validate_data, validate_matcher_config
from cohortbalancer3.metrics.utils import get_caliper_for_matching

__all__ = [
    "Matcher",
    "MatcherConfig",
    "MatchResults",
    "validate_data",
    "validate_matcher_config",
    "get_caliper_for_matching"
] 