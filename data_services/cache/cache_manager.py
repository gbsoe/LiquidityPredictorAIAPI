"""
Cache Manager for SolPool Insight data.

This module implements a multi-level caching system with TTL support
for improved performance and reduced API load.
"""

import os
import json
import time
import logging
import threading
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

from ..config import CACHE_CONFIG, CACHE_DIR

# Configure logging
logger = logging.getLogger('cache_manager')

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
        self.ttl = ttl
        self.last_accessed = self.created_at
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """Check if the entry is expired."""
        return time.time() > (self.created_at + self.ttl)
    
    def access(self) -> Any:
        """Mark as accessed and return data."""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.data
    
    def remaining_ttl(self) -> float:
        """Get remaining TTL in seconds."""
        return max(0, (self.created_at + self.ttl) - time.time())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'data': self.data,
            'created_at': self.created_at,
            'ttl': self.ttl,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        entry = cls(
            key=data['key'],
            data=data['data'],
            ttl=data['ttl']
        )
        entry.created_at = data['created_at']
        entry.last_accessed = data['last_accessed']
        entry.access_count = data['access_count']
        return entry


class CacheManager:
    """
    Multi-level cache manager with memory and disk caching.
    
    Uses TTL-based caching with different expiration periods for
    different types of data.
    """
    
    def __init__(self):
        """Initialize the cache manager."""
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.disk_cache_dir = CACHE_DIR
        self.config = CACHE_CONFIG
        self._lock = threading.RLock()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.disk_cache_dir, exist_ok=True)
        
        # Stats
        self.memory_hits = 0
        self.disk_hits = 0
        self.misses = 0
        
        logger.info(f"Initialized cache manager with directory: {self.disk_cache_dir}")
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start a background thread to clean up expired cache entries."""
        def cleanup_worker():
            while True:
                try:
                    self.cleanup()
                except Exception as e:
                    logger.error(f"Error in cache cleanup: {str(e)}")
                
                # Sleep for 5 minutes between cleanups
                time.sleep(300)
        
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
        logger.info("Started cache cleanup thread")
    
    def _get_disk_path(self, key: str) -> Path:
        """Get the disk path for a cache key."""
        # Create a hash of the key for the filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return Path(self.disk_cache_dir) / f"{key_hash}.json"
    
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
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired():
                    self.memory_hits += 1
                    return entry.access()
                else:
                    # Expired entry
                    del self.memory_cache[key]
            
            # Check disk cache if memory_cache failed
            if self.config['use_disk_cache']:
                disk_path = self._get_disk_path(key)
                if disk_path.exists():
                    try:
                        with open(disk_path, 'r') as f:
                            entry_data = json.load(f)
                            entry = CacheEntry.from_dict(entry_data)
                            
                            if not entry.is_expired():
                                # Valid entry, add to memory cache
                                self.memory_cache[key] = entry
                                self.disk_hits += 1
                                return entry.access()
                            else:
                                # Expired entry, remove from disk
                                os.unlink(disk_path)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Error reading disk cache for {key}: {str(e)}")
                        # Remove corrupted entry
                        try:
                            os.unlink(disk_path)
                        except:
                            pass
            
            # Cache miss
            self.misses += 1
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
            ttl = self.config['default_ttl']
        
        with self._lock:
            # Create new entry
            entry = CacheEntry(key, value, ttl)
            
            # Add to memory cache
            self.memory_cache[key] = entry
            
            # Add to disk cache if enabled
            if self.config['use_disk_cache']:
                disk_path = self._get_disk_path(key)
                try:
                    with open(disk_path, 'w') as f:
                        json.dump(entry.to_dict(), f)
                except Exception as e:
                    logger.warning(f"Error writing to disk cache for {key}: {str(e)}")
    
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
            
            # Delete from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
                deleted = True
            
            # Delete from disk cache
            if self.config['use_disk_cache']:
                disk_path = self._get_disk_path(key)
                if disk_path.exists():
                    try:
                        os.unlink(disk_path)
                        deleted = True
                    except Exception as e:
                        logger.warning(f"Error deleting disk cache for {key}: {str(e)}")
            
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
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired():
                    return True
                else:
                    # Expired entry
                    del self.memory_cache[key]
            
            # Check disk cache if memory_cache failed
            if self.config['use_disk_cache']:
                disk_path = self._get_disk_path(key)
                if disk_path.exists():
                    try:
                        with open(disk_path, 'r') as f:
                            entry_data = json.load(f)
                            entry = CacheEntry.from_dict(entry_data)
                            
                            if not entry.is_expired():
                                # Valid entry, add to memory cache
                                self.memory_cache[key] = entry
                                return True
                            else:
                                # Expired entry, remove from disk
                                os.unlink(disk_path)
                    except:
                        pass
            
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
            self.memory_cache.clear()
            
            # Clear disk cache
            if self.config['use_disk_cache']:
                for path in Path(self.disk_cache_dir).glob('*.json'):
                    try:
                        os.unlink(path)
                    except Exception as e:
                        logger.warning(f"Error deleting disk cache file {path}: {str(e)}")
    
    def cleanup(self) -> int:
        """
        Clean up expired entries.
        
        Returns:
            Number of entries removed
        """
        count = 0
        with self._lock:
            # Clean memory cache
            expired_keys = []
            for key, entry in self.memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
                count += 1
            
            # Clean disk cache
            if self.config['use_disk_cache']:
                for path in Path(self.disk_cache_dir).glob('*.json'):
                    try:
                        with open(path, 'r') as f:
                            entry_data = json.load(f)
                            entry = CacheEntry.from_dict(entry_data)
                            
                            if entry.is_expired():
                                os.unlink(path)
                                count += 1
                    except:
                        # Delete corrupted files
                        try:
                            os.unlink(path)
                            count += 1
                        except:
                            pass
        
        if count > 0:
            logger.info(f"Cleaned up {count} expired cache entries")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            memory_size = len(self.memory_cache)
            
            # Count disk cache
            disk_size = 0
            disk_bytes = 0
            if self.config['use_disk_cache']:
                for path in Path(self.disk_cache_dir).glob('*.json'):
                    disk_size += 1
                    try:
                        disk_bytes += path.stat().st_size
                    except:
                        pass
            
            # Calculate hit rate
            total_requests = self.memory_hits + self.disk_hits + self.misses
            hit_rate = 0 if total_requests == 0 else (self.memory_hits + self.disk_hits) / total_requests
            
            return {
                'memory_cache_size': memory_size,
                'disk_cache_size': disk_size,
                'disk_cache_bytes': disk_bytes,
                'memory_hits': self.memory_hits,
                'disk_hits': self.disk_hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
    
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
                    # Default key generation
                    key = f"{func.__name__}:{args!r}:{kwargs!r}"
                
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

# Create singleton instance
_instance = None

def get_cache_manager() -> CacheManager:
    """Get the singleton cache manager instance."""
    global _instance
    if _instance is None:
        _instance = CacheManager()
    return _instance