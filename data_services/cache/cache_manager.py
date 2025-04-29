"""
Cache Manager for the data services package.

This module provides caching functionality for data services
to reduce API calls and improve performance.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Singleton instance
_instance = None

class CacheManager:
    """
    Cache manager for data services.
    
    This class provides memory and disk caching with TTL control
    for data services to reduce API calls and improve performance.
    """
    
    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize the cache manager.
        
        Args:
            ttl_seconds: Default time-to-live for cache entries (seconds)
        """
        # Memory cache
        self._cache = {}
        self._timestamps = {}
        self._ttl = ttl_seconds
        
        # Cache hit stats
        self._hits = 0
        self._misses = 0
        
        # Create a cache directory if it doesn't exist
        self._cache_dir = os.path.join("data", "cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        
        logger.info(f"Cache manager initialized with TTL {ttl_seconds}s")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key is not found or expired
            
        Returns:
            Cached value or default
        """
        # Check if key exists in memory cache
        if key in self._cache:
            # Check if the entry has expired
            timestamp = self._timestamps.get(key, 0)
            if time.time() - timestamp <= self._ttl:
                # Cache hit
                self._hits += 1
                logger.debug(f"Cache hit for key '{key}'")
                return self._cache[key]
            else:
                # Cache entry expired
                logger.debug(f"Cache entry expired for key '{key}'")
                self._misses += 1
                return default
        else:
            # Check if key exists in disk cache
            disk_value = self._load_from_disk(key)
            if disk_value is not None:
                # Cache hit from disk, store in memory cache for future use
                self._cache[key] = disk_value
                self._timestamps[key] = time.time()
                self._hits += 1
                logger.debug(f"Disk cache hit for key '{key}'")
                return disk_value
            else:
                # Cache miss
                self._misses += 1
                logger.debug(f"Cache miss for key '{key}'")
                return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL (defaults to class TTL)
        """
        # Store in memory cache
        self._cache[key] = value
        self._timestamps[key] = time.time()
        
        # Store on disk for persistence
        self._save_to_disk(key, value)
        
        logger.debug(f"Cached value for key '{key}'")
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was found and invalidated, False otherwise
        """
        # Remove from memory cache
        if key in self._cache:
            del self._cache[key]
            if key in self._timestamps:
                del self._timestamps[key]
            
            # Remove from disk cache
            self._remove_from_disk(key)
            
            logger.debug(f"Invalidated cache for key '{key}'")
            return True
        else:
            # Check if key exists in disk cache
            disk_path = self._get_disk_path(key)
            if os.path.exists(disk_path):
                # Remove from disk cache
                self._remove_from_disk(key)
                logger.debug(f"Invalidated disk cache for key '{key}'")
                return True
            else:
                logger.debug(f"Key '{key}' not found in cache")
                return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        # Clear memory cache
        self._cache = {}
        self._timestamps = {}
        
        # Clear disk cache
        for filename in os.listdir(self._cache_dir):
            if filename.endswith(".cache"):
                try:
                    os.remove(os.path.join(self._cache_dir, filename))
                except Exception as e:
                    logger.warning(f"Error removing cache file {filename}: {str(e)}")
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_ratio = self._hits / total_requests if total_requests > 0 else 0
        
        memory_entry_count = len(self._cache)
        disk_entry_count = sum(1 for f in os.listdir(self._cache_dir) if f.endswith(".cache"))
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_ratio": hit_ratio,
            "memory_entries": memory_entry_count,
            "disk_entries": disk_entry_count,
            "ttl": self._ttl
        }
    
    def _get_disk_path(self, key: str) -> str:
        """
        Get the disk path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the disk cache file
        """
        # Convert the key to a valid filename
        filename = "".join(c if c.isalnum() else "_" for c in key) + ".cache"
        return os.path.join(self._cache_dir, filename)
    
    def _save_to_disk(self, key: str, value: Any) -> bool:
        """
        Save a value to disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the disk path
            disk_path = self._get_disk_path(key)
            
            # Prepare the cache entry
            cache_entry = {
                "key": key,
                "timestamp": time.time(),
                "value": value
            }
            
            # Write to file
            with open(disk_path, "w") as f:
                json.dump(cache_entry, f)
            
            logger.debug(f"Saved cache to disk for key '{key}'")
            return True
        except Exception as e:
            logger.warning(f"Error saving cache to disk for key '{key}': {str(e)}")
            return False
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """
        Load a value from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        try:
            # Get the disk path
            disk_path = self._get_disk_path(key)
            
            # Check if file exists
            if not os.path.exists(disk_path):
                logger.debug(f"No disk cache found for key '{key}'")
                return None
            
            # Read from file
            with open(disk_path, "r") as f:
                cache_entry = json.load(f)
            
            # Check if the entry has expired
            timestamp = cache_entry.get("timestamp", 0)
            if time.time() - timestamp <= self._ttl:
                logger.debug(f"Loaded cache from disk for key '{key}'")
                return cache_entry.get("value")
            else:
                logger.debug(f"Disk cache expired for key '{key}'")
                # Clean up expired entry
                self._remove_from_disk(key)
                return None
        except Exception as e:
            logger.warning(f"Error loading cache from disk for key '{key}': {str(e)}")
            return None
    
    def _remove_from_disk(self, key: str) -> bool:
        """
        Remove a value from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the disk path
            disk_path = self._get_disk_path(key)
            
            # Remove file if it exists
            if os.path.exists(disk_path):
                os.remove(disk_path)
                logger.debug(f"Removed disk cache for key '{key}'")
                return True
            else:
                logger.debug(f"No disk cache found for key '{key}'")
                return False
        except Exception as e:
            logger.warning(f"Error removing disk cache for key '{key}': {str(e)}")
            return False

def get_cache_manager(ttl_seconds: int = 300) -> CacheManager:
    """
    Get the singleton cache manager instance.
    
    Args:
        ttl_seconds: Default time-to-live for cache entries (seconds)
        
    Returns:
        CacheManager instance
    """
    global _instance
    if _instance is None:
        _instance = CacheManager(ttl_seconds=ttl_seconds)
    return _instance