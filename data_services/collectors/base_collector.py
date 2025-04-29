"""
Base Collector Interface for SolPool Insight.

This module provides the base interface and helpers for all data collectors.
"""

import os
import time
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from ..config import API_SETTINGS, FILE_PATHS

# Configure logging
logger = logging.getLogger(__name__)

class BaseCollector:
    """
    Base class for data collectors.
    
    Provides common functionality and interface for:
    - Rate limiting and backoff
    - Error handling and reporting
    - Stats collection
    - Data transformation
    """
    
    def __init__(self, 
                collector_id: str,
                collector_name: str,
                base_url: Optional[str] = None,
                request_delay: float = 0.1):
        """
        Initialize the collector.
        
        Args:
            collector_id: Unique identifier for this collector
            collector_name: Display name for this collector
            base_url: Base URL for API requests (if applicable)
            request_delay: Delay between requests (seconds) for rate limiting
        """
        self.collector_id = collector_id
        self.collector_name = collector_name
        self.base_url = base_url
        self.request_delay = request_delay
        
        # Last collection time
        self.last_collection_time = 0
        
        # Collection stats
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_items_collected": 0,
            "total_collection_time": 0,
            "last_collection_time": None,
            "last_collection_duration": 0,
            "last_collection_item_count": 0,
            "average_request_time": 0,
            "error_count": 0,
            "last_error": None,
            "last_error_time": None
        }
        
        # Backup directory
        self.backup_dir = Path(FILE_PATHS["BACKUP_DIR"]) / collector_id
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize
        logger.info(f"Initialized {collector_name} collector (ID: {collector_id})")
    
    def collect(self, force: bool = False) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Collect data from this source.
        
        Args:
            force: Force collection even if recently collected
            
        Returns:
            Tuple of (collected_data, success_flag)
        """
        # Check if we should collect (unless forced)
        current_time = time.time()
        time_since_last = current_time - self.last_collection_time
        
        # Skip if collected recently (within last minute) and not forced
        if not force and time_since_last < 60:
            logger.info(f"Skipping collection for {self.collector_name} - collected {time_since_last:.1f}s ago")
            return [], True
        
        # Start collection
        start_time = time.time()
        logger.info(f"Starting data collection for {self.collector_name}")
        
        try:
            # Perform the actual collection (implemented by subclasses)
            collected_data = self._collect_data()
            
            # Update stats
            end_time = time.time()
            duration = end_time - start_time
            
            self.stats["total_collection_time"] += duration
            self.stats["last_collection_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.stats["last_collection_duration"] = duration
            self.stats["last_collection_item_count"] = len(collected_data)
            self.stats["total_items_collected"] += len(collected_data)
            
            # Save successful collection as backup
            self._save_backup(collected_data)
            
            # Update last collection time
            self.last_collection_time = current_time
            
            logger.info(
                f"Completed collection for {self.collector_name} - " +
                f"collected {len(collected_data)} items in {duration:.2f}s"
            )
            
            return collected_data, True
            
        except Exception as e:
            # Log the error
            logger.error(f"Error during collection for {self.collector_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update error stats
            self.stats["error_count"] += 1
            self.stats["last_error"] = str(e)
            self.stats["last_error_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Try to load the most recent backup
            backup_data = self._load_latest_backup()
            if backup_data:
                logger.info(f"Loaded {len(backup_data)} items from backup for {self.collector_name}")
                return backup_data, False
            
            return [], False
    
    def _collect_data(self) -> List[Dict[str, Any]]:
        """
        Collect data from the source.
        
        This method should be implemented by subclasses.
        
        Returns:
            List of collected data items
        """
        raise NotImplementedError("Subclasses must implement _collect_data method")
    
    def _save_backup(self, data: List[Dict[str, Any]]) -> bool:
        """
        Save collected data as a backup.
        
        Args:
            data: The data to save
            
        Returns:
            True if successful, False otherwise
        """
        if not data:
            return False
            
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.collector_id}_{timestamp}.json"
            filepath = self.backup_dir / filename
            
            # Save to file
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
                
            # Remove old backups (keep last 10)
            self._cleanup_old_backups()
                
            return True
        except Exception as e:
            logger.error(f"Error saving backup for {self.collector_name}: {str(e)}")
            return False
    
    def _cleanup_old_backups(self, max_backups: int = 10) -> int:
        """
        Remove old backup files to avoid disk space issues.
        
        Args:
            max_backups: Maximum number of backup files to keep
            
        Returns:
            Number of files removed
        """
        try:
            # Get all backup files
            backup_files = list(self.backup_dir.glob(f"{self.collector_id}_*.json"))
            
            # Sort by modification time (oldest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x))
            
            # Remove oldest files if we have too many
            removed = 0
            while len(backup_files) > max_backups:
                # Remove oldest file
                oldest = backup_files.pop(0)
                oldest.unlink(missing_ok=True)
                removed += 1
                
            return removed
        except Exception as e:
            logger.error(f"Error cleaning up old backups for {self.collector_name}: {str(e)}")
            return 0
    
    def _load_latest_backup(self) -> List[Dict[str, Any]]:
        """
        Load the most recent backup file.
        
        Returns:
            The loaded data or an empty list if no backup found
        """
        try:
            # Get all backup files
            backup_files = list(self.backup_dir.glob(f"{self.collector_id}_*.json"))
            
            if not backup_files:
                return []
                
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Load newest file
            newest = backup_files[0]
            with open(newest, "r") as f:
                data = json.load(f)
                
            return data
        except Exception as e:
            logger.error(f"Error loading backup for {self.collector_name}: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collector statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        
        # Add collector identity
        stats["collector_id"] = self.collector_id
        stats["collector_name"] = self.collector_name
        
        # Add additional info
        stats["has_backups"] = self._has_backups()
        stats["backup_count"] = self._count_backups()
        
        return stats
    
    def _has_backups(self) -> bool:
        """Check if backups exist."""
        backup_files = list(self.backup_dir.glob(f"{self.collector_id}_*.json"))
        return len(backup_files) > 0
    
    def _count_backups(self) -> int:
        """Count backup files."""
        backup_files = list(self.backup_dir.glob(f"{self.collector_id}_*.json"))
        return len(backup_files)
    
    def _track_request_stats(self, success: bool, duration: float) -> None:
        """
        Track statistics for a request.
        
        Args:
            success: Whether the request was successful
            duration: Time taken for the request in seconds
        """
        # Update request counts
        self.stats["total_requests"] += 1
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Update average request time
        if self.stats["total_requests"] == 1:
            self.stats["average_request_time"] = duration
        else:
            # Running average
            prev_avg = self.stats["average_request_time"]
            count = self.stats["total_requests"]
            self.stats["average_request_time"] = prev_avg + (duration - prev_avg) / count
    
    def _normalize_pool_data(self, pool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize pool data to a standard format.
        
        Args:
            pool: Raw pool data
            
        Returns:
            Normalized pool data
        """
        # Create a normalized pool object
        normalized = {
            "pool_id": pool.get("id") or pool.get("pool_id") or pool.get("address") or "",
            "name": pool.get("name") or pool.get("displayName") or pool.get("symbol") or "",
            "dex": pool.get("dex") or pool.get("platform") or pool.get("amm") or "Unknown",
            "liquidity": pool.get("liquidity") or pool.get("tvl") or 0,
            "volume_24h": pool.get("volume_24h") or pool.get("volume") or 0,
            "fees_24h": pool.get("fees_24h") or pool.get("fees") or 0,
            "apr": pool.get("apr") or pool.get("apy") or 0,
            "price": pool.get("price") or 0,
            "token0": self._normalize_token(pool.get("token0") or pool.get("baseToken")),
            "token1": self._normalize_token(pool.get("token1") or pool.get("quoteToken")),
            "fee_tier": pool.get("fee_tier") or pool.get("fee") or 0,
            "data_source": pool.get("data_source") or f"Collected by {self.collector_name}",
            "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "collector_id": self.collector_id,
            # Track original data for reference
            "original_data": pool
        }
        
        return normalized
    
    def _normalize_token(self, token: Any) -> Dict[str, Any]:
        """
        Normalize token data to a standard format.
        
        Args:
            token: Raw token data (can be string or dict)
            
        Returns:
            Normalized token data
        """
        if token is None:
            return {
                "symbol": "UNKNOWN",
                "address": "",
                "name": "Unknown Token",
                "decimals": 9
            }
        
        # If token is already a dict, extract fields
        if isinstance(token, dict):
            return {
                "symbol": token.get("symbol") or "UNKNOWN",
                "address": token.get("address") or token.get("mint") or "",
                "name": token.get("name") or token.get("displayName") or token.get("symbol") or "Unknown Token",
                "decimals": token.get("decimals") or 9
            }
        
        # If token is a string, it's likely the symbol or address
        if isinstance(token, str):
            # Check if it looks like an address (base58 for Solana is ~32-44 chars)
            if len(token) > 30:
                return {
                    "symbol": "UNKNOWN",
                    "address": token,
                    "name": "Unknown Token",
                    "decimals": 9
                }
            else:
                # It's probably a symbol
                return {
                    "symbol": token,
                    "address": "",
                    "name": token,
                    "decimals": 9
                }
        
        # Fallback
        return {
            "symbol": "UNKNOWN",
            "address": "",
            "name": "Unknown Token",
            "decimals": 9
        }