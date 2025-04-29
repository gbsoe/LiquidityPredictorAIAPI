"""
Initialization module for SolPool Insight data services.

This module provides a simple way to initialize the data services
and start the scheduled data collection.
"""

import logging
from typing import Dict, Any

from .data_service import initialize_data_service, get_data_service
from .cache import get_cache_manager

# Configure logging
logger = logging.getLogger(__name__)

def init_services() -> Dict[str, Any]:
    """
    Initialize all data services.
    
    Returns:
        Dictionary with service instances
    """
    logger.info("Initializing data services...")
    
    # Initialize data service (which starts scheduled collection)
    data_service = initialize_data_service()
    
    # Get cache manager
    cache_manager = get_cache_manager()
    
    # Return services
    services = {
        "data_service": data_service,
        "cache_manager": cache_manager
    }
    
    logger.info("Data services initialized successfully")
    return services

def get_stats() -> Dict[str, Any]:
    """
    Get statistics for all services.
    
    Returns:
        Dictionary with service statistics
    """
    # Get data service
    data_service = get_data_service()
    
    # Get cache manager
    cache_manager = get_cache_manager()
    
    # Get statistics
    stats = {
        "data_service": data_service.get_system_stats(),
        "cache": cache_manager.get_stats()
    }
    
    return stats