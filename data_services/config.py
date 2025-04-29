"""
Configuration for the data services system.

This module contains all the configuration parameters for the data collection,
processing, and caching services.
"""

import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
BACKUP_DIR = DATA_DIR / "backups"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# API Configuration
API_CONFIG = {
    "defi_aggregation": {
        "base_url": "https://filotdefiapi.replit.app/api/v1",
        "api_key_env": "DEFI_API_KEY",
        "rate_limit": {
            "calls_per_minute": 25,
            "max_concurrent": 2
        },
        "timeout": 10,  # seconds
        "retries": 3,
        "backoff_factor": 1.5
    },
    "direct_dex": {
        "raydium": {
            "enabled": False,  # Will be enabled once implemented
            "priority": 7
        },
        "orca": {
            "enabled": False,  # Will be enabled once implemented
            "priority": 7
        },
        "meteora": {
            "enabled": False,  # Will be enabled once implemented
            "priority": 7
        }
    }
}

# Collection Schedule Configuration
COLLECTION_CONFIG = {
    "interval_minutes": 15,  # Run every 15 minutes
    "staggered_start": True,  # Stagger starts to avoid API hammering
    "initial_delay": 10,  # seconds before first collection
    "timeout": 120,  # Maximum time for collection job in seconds
    "skip_if_recent": 5  # Skip collection if data is less than X minutes old
}

# Database Configuration
DB_CONFIG = {
    "use_postgresql": True,
    "connection_string_env": "DATABASE_URL",
    "sqlite_fallback": "historical_pools.db",
    "pool_size": 5,
    "max_overflow": 10,
    "timeout": 30
}

# Cache Configuration
CACHE_CONFIG = {
    "default_ttl": 300,  # 5 minutes for most data
    "long_ttl": 1800,  # 30 minutes for rarely changing data
    "extended_ttl": 3600 * 4,  # 4 hours for static data
    "use_memory_cache": True,
    "use_disk_cache": True,
    "disk_cache_size": 1024 * 1024 * 100  # 100 MB
}

# Prediction Model Configuration
PREDICTION_CONFIG = {
    "enabled": True,
    "models_dir": ROOT_DIR / "models",
    "training_interval_hours": 24,  # Retrain models daily
    "min_history_points": 10,  # Minimum data points needed for prediction
    "prediction_horizon_days": [1, 7, 30],  # Make predictions for these horizons
    "ensemble_method": "weighted_average",
    "default_models": ["linear", "xgboost", "prophet"]
}

# Data Retention Policy
RETENTION_CONFIG = {
    "max_days": 90,  # Keep data for 90 days
    "pruning_schedule": "0 1 * * *",  # Run at 1 AM daily (cron format)
    "min_snapshots_per_pool": 50,  # Always keep at least this many snapshots
    "backup_before_pruning": True
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "file": ROOT_DIR / "logs" / "data_services.log",
    "max_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# Ensure logs directory exists
os.makedirs(ROOT_DIR / "logs", exist_ok=True)