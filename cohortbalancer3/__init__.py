"""CohortBalancer3: A Python package for propensity score matching and treatment effect estimation.

This package provides tools for performing propensity score matching,
assessing covariate balance, and estimating treatment effects.
"""

__version__ = "0.2.0"

# Initialize logging
import logging

from cohortbalancer3.utils.logging import configure_logging

# Configure default logging (INFO level, to stdout)
logger = configure_logging()

# Import and expose key classes and functions
from cohortbalancer3.datatypes import MatcherConfig, MatchResults
from cohortbalancer3.matcher import Matcher
from cohortbalancer3.metrics.utils import get_caliper_for_matching
from cohortbalancer3.reporting import (
    create_report,
    create_visualizations,
    export_tables,
)
from cohortbalancer3.validation import validate_data, validate_matcher_config

__all__ = [
    "MatchResults",
    "Matcher",
    "MatcherConfig",
    "configure_logging",  # Add configure_logging to allow users to customize logging
    "create_report",
    "create_visualizations",
    "export_tables",
    "get_caliper_for_matching",
    "validate_data",
    "validate_matcher_config",
]
