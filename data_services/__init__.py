"""
Data Services Package for SolPool Insight.

This package provides data collection, processing, and storage 
services for liquidity pool data from various sources.

Key Components:
- Collectors: Standardized interfaces for data collection from APIs
- Cache: Memory and disk caching with TTL control
- Data Service: Central coordination of collection and processing
"""

from .initialize import init_services, get_stats
from .data_service import get_data_service