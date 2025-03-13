"""
Logging utilities for CohortBalancer3.

This module provides functions for configuring and using logging
in a consistent way throughout the package.
"""

import logging
import sys
from typing import Optional, TextIO, Union


def get_logger(name: str = "cohortbalancer3") -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Name for the logger, defaults to 'cohortbalancer3'
        
    Returns:
        A named logger instance
    """
    return logging.getLogger(name)


def configure_logging(
    level: Union[int, str] = logging.INFO,
    format_string: Optional[str] = None,
    stream: Optional[TextIO] = sys.stdout,
    log_file: Optional[str] = None,
    name: str = "cohortbalancer3"
) -> logging.Logger:
    """Configure logging for CohortBalancer3.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Format string for log messages
        stream: Stream to output logs to (default: sys.stdout)
        log_file: Optional file path to write logs to
        name: Logger name, defaults to 'cohortbalancer3'
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set the logging level
    logger.setLevel(level)
    
    # Default format string if not provided
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Add console handler
    if stream:
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Don't propagate to root logger
    logger.propagate = False
    
    return logger


# Create a default logger instance
logger = get_logger() 