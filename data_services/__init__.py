"""
Data Services package for SolPool Insight.

This package provides data collection, processing, and caching services
for liquidity pool data across various Solana DEXes.
"""

# Import key functions for simplified access
from .initialize import init_services, get_stats
from .data_service import get_data_service, initialize_data_service