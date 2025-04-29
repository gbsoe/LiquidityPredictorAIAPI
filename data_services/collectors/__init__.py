"""
Collector modules for the data services package.

This package contains collectors for various data sources.
Each collector implements a standardized interface for
retrieving data from a specific source.
"""

# Import collector utilities
from .base_collector import BaseCollector, CollectorStatus

# Import collector implementations
from .defi_aggregation_collector import get_collector as get_defi_collector