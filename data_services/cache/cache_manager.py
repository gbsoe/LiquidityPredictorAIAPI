"""
Cache Manager for SolPool Insight data.

This module implements a multi-level caching system with TTL support
for improved performance and reduced API load.
"""

import os
import time
import json
import logging
import threading
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta

from ..config import CACHE_SETTINGS

# Configure logging
logger = logging.getLogger(__name__)

class CacheEntry:
    """Represents a single cache entry with metadata."""
    
    def __init__(self, key: str, data: Any, ttl: int = 300):
        """
        Initialize a cache entry.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
        """
        self.key = key
        self.data = data
        self.created_at = time.time()
        self.accessed_at = time.time()
        self.ttl = ttl
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """Check if the entry is expired."""
        return time.time() > self.created_at + self.ttl
    
    def access(self) -> Any:
        """Mark as accessed and return data."""
        self.accessed_at = time.time()
        self.access_count += 1
        return self.data
    
    def remaining_ttl(self) -> float:
        """Get remaining TTL in seconds."""
        elapsed = time.time() - self.created_at
        return max(0, self.ttl - elapsed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "data": self.data,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "ttl": self.ttl,
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        entry = cls(data["key"], data["data"], data["ttl"])
        entry.created_at = data["created_at"]
        entry.accessed_at = data["accessed_at"]
        entry.access_count = data["access_count"]
        return entry

class CacheManager:
    """
    Multi-level cache manager with memory and disk caching.
    
    Uses TTL-based caching with different expiration periods for
    different types of data.
    """
    _instance = None  # Singleton instance
    
    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the cache manager."""
        # Skip initialization if already initialized
        if getattr(self, '_initialized', False):
            return
            
        # Memory cache
        self._memory_cache: Dict[str, CacheEntry] = {}
        
        # Cache settings
        self.cache_dir = Path(CACHE_SETTINGS["CACHE_DIR"])
        self.default_ttl = CACHE_SETTINGS["DEFAULT_TTL"]
        self.memory_cache_max_items = CACHE_SETTINGS["MEMORY_CACHE_MAX_ITEMS"]
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for thread safety
        self._lock = threading.RLock()
        
        # Stats
        self._stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "cleanups": 0
        }
        
        # Start background cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"Cache manager initialized with cache directory: {self.cache_dir}")
        self._initialized = True
    
    def _start_cleanup_thread(self):
        """Start a background thread to clean up expired cache entries."""
        def cleanup_worker():
            while True:
                # Sleep for 5 minutes before cleanup
                time.sleep(300)
                
                # Perform cleanup
                try:
                    removed = self.cleanup()
                    if removed > 0:
                        logger.info(f"Cache cleanup removed {removed} expired entries")
                except Exception as e:
                    logger.error(f"Error during cache cleanup: {str(e)}")
        
        # Start the thread
        thread = threading.Thread(
            target=cleanup_worker, 
            daemon=True,
            name="CacheCleanupThread"
        )
        thread.start()
    
    def _get_disk_path(self, key: str) -> Path:
        """Get the disk path for a cache key."""
        # Use MD5 hash of the key for the filename
        hash_obj = hashlib.md5(key.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Use the first 2 characters as a subdirectory for better organization
        subdir = hash_hex[:2]
        
        # Create subdirectory if it doesn't exist
        subdir_path = self.cache_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        
        # Return full path
        return subdir_path / f"{hash_hex}.json"
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            # Try memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check if expired
                if entry.is_expired():
                    # Remove from memory cache
                    del self._memory_cache[key]
                else:
                    # Update stats
                    self._stats["memory_hits"] += 1
                    
                    # Return data
                    return entry.access()
            
            # Try disk cache
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                try:
                    with open(disk_path, "r") as f:
                        entry_dict = json.load(f)
                        entry = CacheEntry.from_dict(entry_dict)
                        
                        # Check if expired
                        if entry.is_expired():
                            # Remove from disk
                            disk_path.unlink(missing_ok=True)
                        else:
                            # Update stats
                            self._stats["disk_hits"] += 1
                            
                            # Put in memory cache
                            self._memory_cache[key] = entry
                            
                            # Manage memory cache size
                            if len(self._memory_cache) > self.memory_cache_max_items:
                                # Remove least recently accessed item
                                lru_key = min(
                                    self._memory_cache.keys(),
                                    key=lambda k: self._memory_cache[k].accessed_at
                                )
                                del self._memory_cache[lru_key]
                            
                            # Return data
                            return entry.access()
                except Exception as e:
                    logger.error(f"Error reading cache from disk: {str(e)}")
            
            # Update stats
            self._stats["misses"] += 1
            
            # Not found
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl
            
        with self._lock:
            # Create entry
            entry = CacheEntry(key, value, ttl)
            
            # Update memory cache
            self._memory_cache[key] = entry
            
            # Manage memory cache size
            if len(self._memory_cache) > self.memory_cache_max_items:
                # Remove least recently accessed item
                lru_key = min(
                    self._memory_cache.keys(),
                    key=lambda k: self._memory_cache[k].accessed_at
                )
                del self._memory_cache[lru_key]
            
            # Update disk cache
            disk_path = self._get_disk_path(key)
            try:
                with open(disk_path, "w") as f:
                    json.dump(entry.to_dict(), f)
            except Exception as e:
                logger.error(f"Error writing cache to disk: {str(e)}")
            
            # Update stats
            self._stats["sets"] += 1
    
    def delete(self, key: str) -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            deleted = False
            
            # Remove from memory cache
            if key in self._memory_cache:
                del self._memory_cache[key]
                deleted = True
            
            # Remove from disk cache
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                try:
                    disk_path.unlink()
                    deleted = True
                except Exception as e:
                    logger.error(f"Error deleting cache from disk: {str(e)}")
            
            # Update stats if deleted
            if deleted:
                self._stats["deletes"] += 1
            
            return deleted
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache and is not expired.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists and not expired
        """
        with self._lock:
            # Check memory cache
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if not entry.is_expired():
                    return True
                    
            # Check disk cache
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                try:
                    with open(disk_path, "r") as f:
                        entry_dict = json.load(f)
                        entry = CacheEntry.from_dict(entry_dict)
                        
                        # Check if expired
                        if not entry.is_expired():
                            return True
                except Exception as e:
                    logger.error(f"Error checking cache on disk: {str(e)}")
            
            return False
    
    def set_multi(self, 
                 items: Dict[str, Any], 
                 ttl: Optional[int] = None) -> None:
        """
        Set multiple items in the cache.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds (uses default if None)
        """
        for key, value in items.items():
            self.set(key, value, ttl)
    
    def get_multi(self, 
                 keys: List[str], 
                 default: Any = None) -> Dict[str, Any]:
        """
        Get multiple items from the cache.
        
        Args:
            keys: List of cache keys
            default: Default value for missing keys
            
        Returns:
            Dictionary of key-value pairs
        """
        result = {}
        for key in keys:
            result[key] = self.get(key, default)
        return result
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            # Clear memory cache
            self._memory_cache.clear()
            
            # Clear disk cache
            try:
                # Only remove files, not directories
                for subdir in self.cache_dir.iterdir():
                    if subdir.is_dir():
                        for file in subdir.iterdir():
                            if file.is_file() and file.suffix == ".json":
                                file.unlink()
            except Exception as e:
                logger.error(f"Error clearing disk cache: {str(e)}")
    
    def cleanup(self) -> int:
        """
        Clean up expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            removed = 0
            
            # Clean memory cache
            expired_keys = []
            for key, entry in self._memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
                removed += 1
            
            # Clean disk cache
            try:
                # Check each file in the cache directory
                for subdir in self.cache_dir.iterdir():
                    if subdir.is_dir():
                        for file in subdir.iterdir():
                            if file.is_file() and file.suffix == ".json":
                                try:
                                    with open(file, "r") as f:
                                        entry_dict = json.load(f)
                                        entry = CacheEntry.from_dict(entry_dict)
                                        
                                        # Remove if expired
                                        if entry.is_expired():
                                            file.unlink()
                                            removed += 1
                                except Exception as e:
                                    logger.error(f"Error checking file {file}: {str(e)}")
                                    # Remove corrupted files
                                    file.unlink(missing_ok=True)
                                    removed += 1
            except Exception as e:
                logger.error(f"Error during disk cache cleanup: {str(e)}")
            
            # Update stats
            self._stats["cleanups"] += 1
            
            return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            # Add current size info
            stats = self._stats.copy()
            stats["memory_cache_size"] = len(self._memory_cache)
            
            # Count disk cache files
            disk_cache_size = 0
            try:
                for subdir in self.cache_dir.iterdir():
                    if subdir.is_dir():
                        for file in subdir.iterdir():
                            if file.is_file() and file.suffix == ".json":
                                disk_cache_size += 1
            except Exception as e:
                logger.error(f"Error counting disk cache files: {str(e)}")
            
            stats["disk_cache_size"] = disk_cache_size
            stats["total_size"] = stats["memory_cache_size"] + stats["disk_cache_size"]
            
            # Calculate hit ratio
            total_requests = stats["memory_hits"] + stats["disk_hits"] + stats["misses"]
            if total_requests > 0:
                stats["hit_ratio"] = (stats["memory_hits"] + stats["disk_hits"]) / total_requests
            else:
                stats["hit_ratio"] = 0.0
                
            return stats
    
    def cached(self, 
              ttl: Optional[int] = None, 
              key_fn: Optional[Callable] = None):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time to live in seconds (uses default if None)
            key_fn: Function to generate cache key from arguments
                    (uses str(args) + str(kwargs) if None)
        
        Returns:
            Decorated function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_fn:
                    key = key_fn(*args, **kwargs)
                else:
                    # Default key is function name + args + kwargs
                    key = f"{func.__module__}.{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                
                # Check cache
                cached_result = self.get(key)
                if cached_result is not None:
                    return cached_result
                
                # Call function
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(key, result, ttl)
                
                return result
            
            return wrapper
        
        return decorator


# Singleton instance getter
def get_cache_manager() -> CacheManager:
    """Get the singleton cache manager instance."""
    return CacheManager()