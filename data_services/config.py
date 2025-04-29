"""
Configuration module for the data services package.

This module provides configuration options and settings
for data services.
"""

import os
from typing import Dict, Any

# API keys and endpoints from environment variables
DEFI_API_KEY = os.getenv("DEFI_API_KEY")
DEFI_API_BASE_URL = os.getenv("DEFI_API_BASE_URL", "https://filotdefiapi.replit.app/api/v1")

# Default collection settings
DEFAULT_COLLECTION_INTERVAL = 15 * 60  # 15 minutes
MAX_POOLS_PER_COLLECTION = 100

# Default cache settings
DEFAULT_CACHE_TTL = 300  # 5 minutes
POOL_CACHE_TTL = 600  # 10 minutes
TOKEN_CACHE_TTL = 900  # 15 minutes

# Collection sources
ENABLE_DEFI_AGGREGATION_API = True

# Data storage configuration
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

def get_config() -> Dict[str, Any]:
    """
    Get the full configuration.
    
    Returns:
        Dictionary with configuration
    """
    return {
        "api": {
            "defi_api_key": DEFI_API_KEY,
            "defi_api_base_url": DEFI_API_BASE_URL,
        },
        "collection": {
            "interval": DEFAULT_COLLECTION_INTERVAL,
            "max_pools": MAX_POOLS_PER_COLLECTION,
            "enable_defi_aggregation_api": ENABLE_DEFI_AGGREGATION_API,
        },
        "cache": {
            "ttl": DEFAULT_CACHE_TTL,
            "pool_ttl": POOL_CACHE_TTL,
            "token_ttl": TOKEN_CACHE_TTL,
        },
        "storage": {
            "data_dir": DATA_DIR,
            "cache_dir": CACHE_DIR,
            "backup_dir": BACKUP_DIR,
        }
    }