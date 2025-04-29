"""
Data Services for SolPool Insight

This package provides services for data collection, processing, and caching
to support the SolPool Insight application.
"""

from .data_service import get_data_service, initialize_data_service

__all__ = ['get_data_service', 'initialize_data_service']