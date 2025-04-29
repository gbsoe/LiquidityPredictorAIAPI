"""
Configuration for data services.

This module provides configuration settings for the data collection,
processing, and caching services.
"""

import os
import logging
from datetime import timedelta
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Cache TTL settings (in seconds)
CACHE_SETTINGS = {
    # Short-lived cache for frequently changing data (5 minutes)
    "DEFAULT_TTL": 60 * 5,
    # Medium-lived cache for data that changes less frequently (1 hour)
    "MEDIUM_TTL": 60 * 60,
    # Long-lived cache for relatively static data (24 hours)
    "LONG_TTL": 60 * 60 * 24,
    # Cache directory location
    "CACHE_DIR": "data/cache",
    # Memory cache size limit (items)
    "MEMORY_CACHE_MAX_ITEMS": 1000,
}

# Collection schedule settings
SCHEDULE_SETTINGS = {
    # How often to collect new data (minutes)
    "COLLECTION_INTERVAL_MINUTES": 15,
    # Maximum collection threads
    "MAX_COLLECTION_THREADS": 3,
    # Collection timeout (seconds)
    "COLLECTION_TIMEOUT": 60,
}

# API settings
API_SETTINGS = {
    # Base URL for DeFi API
    "DEFI_API_URL": os.getenv("DEFI_API_URL", "https://filotdefiapi.replit.app/api/v1"),
    # API key for DeFi API
    "DEFI_API_KEY": os.getenv("DEFI_API_KEY"),
    # Request delay for rate limiting (seconds)
    "REQUEST_DELAY": 0.1,
    # Maximum retries for API requests
    "MAX_RETRIES": 3,
    # Retry delay (seconds)
    "RETRY_DELAY": 1.0,
}

# Database settings
DB_SETTINGS = {
    # Database URL
    "DB_URL": os.getenv("DATABASE_URL"),
    # Table names
    "POOL_TABLE": "liquidity_pools",
    "POOL_HISTORY_TABLE": "pool_history",
    "TOKEN_TABLE": "tokens",
    "TOKEN_PRICE_TABLE": "token_prices",
}

# File paths
FILE_PATHS = {
    # Backup directory
    "BACKUP_DIR": "data/backups",
    # Log directory
    "LOG_DIR": "logs",
}

def get_settings() -> Dict[str, Any]:
    """
    Get all settings as a dictionary.
    
    Returns:
        Dictionary with all settings
    """
    return {
        "cache": CACHE_SETTINGS,
        "schedule": SCHEDULE_SETTINGS,
        "api": API_SETTINGS,
        "db": DB_SETTINGS,
        "paths": FILE_PATHS,
    }

def validate_settings() -> bool:
    """
    Validate all settings and ensure required values are present.
    
    Returns:
        True if all settings are valid, False otherwise
    """
    # Check if API key is available
    if not API_SETTINGS["DEFI_API_KEY"]:
        logger.warning("DEFI_API_KEY is not set. DeFi API collector will not work.")
        return False
        
    # Check if database URL is available
    if not DB_SETTINGS["DB_URL"]:
        logger.warning("DATABASE_URL is not set. Database operations will not work.")
        
    # Ensure cache directory exists
    cache_dir = CACHE_SETTINGS["CACHE_DIR"]
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Created cache directory: {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {str(e)}")
            return False
            
    # Ensure backup directory exists
    backup_dir = FILE_PATHS["BACKUP_DIR"]
    if not os.path.exists(backup_dir):
        try:
            os.makedirs(backup_dir, exist_ok=True)
            logger.info(f"Created backup directory: {backup_dir}")
        except Exception as e:
            logger.error(f"Failed to create backup directory: {str(e)}")
            return False
            
    # Ensure log directory exists
    log_dir = FILE_PATHS["LOG_DIR"]
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"Created log directory: {log_dir}")
        except Exception as e:
            logger.error(f"Failed to create log directory: {str(e)}")
            return False
            
    return True