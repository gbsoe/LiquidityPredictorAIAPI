"""
Initialization module for the data services package.

This module provides functions to initialize and configure 
the data services package for use in the application.
"""

import logging
from typing import Any, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import data service
from .data_service import initialize_data_service, get_data_service

# Flag to track initialization
_initialized = False

def init_services() -> None:
    """
    Initialize all data services.
    
    This function sets up all required data services and 
    starts background collection if needed.
    """
    global _initialized
    
    if _initialized:
        logger.info("Data services already initialized")
        return
    
    logger.info("Initializing data services")
    
    # Initialize the data service
    data_service = initialize_data_service()
    
    # Mark as initialized
    _initialized = True
    logger.info("Data services initialized successfully")

def get_stats() -> Dict[str, Any]:
    """
    Get statistics for all data services.
    
    Returns:
        Dictionary with system statistics
    """
    # Make sure services are initialized
    if not _initialized:
        init_services()
    
    # Get the data service
    data_service = get_data_service()
    
    # Get the system stats
    return data_service.get_system_stats()