"""
Data Collectors for SolPool Insight

This package provides collectors for different data sources.
"""

from .defi_aggregation_collector import get_collector as get_defi_collector

__all__ = ['get_defi_collector']