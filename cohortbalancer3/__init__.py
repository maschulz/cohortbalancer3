"""
CohortBalancer3: A Python package for propensity score matching and treatment effect estimation.

This package provides tools for performing propensity score matching,
assessing covariate balance, and estimating treatment effects.
"""

__version__ = "0.1.0"

from cohortbalancer3.datatypes import MatcherConfig, MatchResults
from cohortbalancer3.matcher import Matcher

__all__ = [
    "Matcher",
    "MatcherConfig",
    "MatchResults"
] 