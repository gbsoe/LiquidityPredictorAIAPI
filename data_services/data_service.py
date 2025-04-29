"""
Central Data Service for SolPool Insight.

This module provides a unified interface for data collection,
processing, and retrieval with caching and scheduling.
"""

import os
import time
import json
import logging
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
import schedule

from .config import SCHEDULE_SETTINGS, validate_settings
from .cache import get_cache_manager
from .collectors import get_defi_collector

# Configure logging
logger = logging.getLogger(__name__)

class DataService:
    """
    Central service for data collection and management.
    
    This service coordinates data collection from multiple sources,
    manages caching, and provides a unified API for the application.
    """
    _instance = None  # Singleton instance
    
    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(DataService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the data service."""
        # Skip initialization if already initialized
        if getattr(self, '_initialized', False):
            return
            
        # Validate configuration
        if not validate_settings():
            logger.warning("Some settings are invalid or missing")
            
        # Initialize cache manager
        self.cache_manager = get_cache_manager()
        
        # Initialize collectors
        self.collectors = {}
        self._init_collectors()
        
        # Scheduled collection
        self.scheduler = schedule.Scheduler()
        self.scheduler_thread = None
        self.scheduler_running = False
        
        # Collection stats
        self.stats = {
            "last_collection_time": None,
            "last_collection_duration": 0,
            "last_collection_pool_count": 0,
            "total_collections": 0,
            "total_pools_collected": 0,
            "collection_errors": 0
        }
        
        logger.info("Data service initialized")
        self._initialized = True
    
    def _init_collectors(self):
        """Initialize data collectors."""
        # Add DeFi API collector
        try:
            defi_collector = get_defi_collector()
            self.collectors["defi_api"] = defi_collector
            logger.info("Added DeFi API collector")
        except Exception as e:
            logger.error(f"Error initializing DeFi API collector: {str(e)}")
        
        # Add other collectors here as needed
        
        logger.info(f"Initialized {len(self.collectors)} data collectors")
    
    def get_collector_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all collectors.
        
        Returns:
            List of collector statistics
        """
        stats = []
        for collector_id, collector in self.collectors.items():
            try:
                collector_stats = collector.get_stats()
                stats.append(collector_stats)
            except Exception as e:
                logger.error(f"Error getting stats for collector {collector_id}: {str(e)}")
                stats.append({
                    "collector_id": collector_id,
                    "error": str(e)
                })
        
        return stats
    
    def collect_data(self, force: bool = False) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Collect data from all sources.
        
        Args:
            force: Force collection even if recent
            
        Returns:
            Tuple of (collected_data, success)
        """
        # Track collection time
        start_time = time.time()
        
        # Store all collected pools
        all_pools = []
        
        # Track success
        success = True
        
        # Track pool IDs to avoid duplicates
        pool_ids = set()
        
        # Update stats
        self.stats["last_collection_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats["total_collections"] += 1
        
        # Collect from each collector
        for collector_id, collector in self.collectors.items():
            try:
                logger.info(f"Collecting data from {collector_id}")
                pools, collector_success = collector.collect(force=force)
                
                # Update overall success
                success = success and collector_success
                
                if pools:
                    # Filter out duplicates
                    unique_pools = []
                    for pool in pools:
                        pool_id = pool.get("pool_id")
                        if pool_id and pool_id not in pool_ids:
                            pool_ids.add(pool_id)
                            unique_pools.append(pool)
                    
                    # Add unique pools to result
                    logger.info(f"Adding {len(unique_pools)} unique pools from {collector_id}")
                    all_pools.extend(unique_pools)
                else:
                    logger.warning(f"No pools collected from {collector_id}")
            except Exception as e:
                logger.error(f"Error collecting from {collector_id}: {str(e)}")
                logger.error(traceback.format_exc())
                success = False
                self.stats["collection_errors"] += 1
        
        # Update stats
        duration = time.time() - start_time
        self.stats["last_collection_duration"] = duration
        self.stats["last_collection_pool_count"] = len(all_pools)
        self.stats["total_pools_collected"] += len(all_pools)
        
        # Log stats
        logger.info(
            f"Collection complete in {duration:.2f}s: {len(all_pools)} pools " +
            f"({len(pool_ids)} unique) from {len(self.collectors)} collectors"
        )
        
        # Update cache
        if all_pools:
            try:
                # Cache the combined pool data
                cache_key = "all_pools"
                self.cache_manager.set(
                    key=cache_key,
                    value=all_pools,
                    ttl=300  # 5 minutes
                )
                logger.info(f"Cached {len(all_pools)} pools with key '{cache_key}'")
                
                # Also cache by DEX for more granular access
                dex_pools = {}
                for pool in all_pools:
                    dex = pool.get("dex", "unknown").lower()
                    if dex not in dex_pools:
                        dex_pools[dex] = []
                    dex_pools[dex].append(pool)
                
                # Cache each DEX's pools
                for dex, pools in dex_pools.items():
                    dex_cache_key = f"pools_by_dex_{dex}"
                    self.cache_manager.set(
                        key=dex_cache_key,
                        value=pools,
                        ttl=300  # 5 minutes
                    )
                    logger.info(f"Cached {len(pools)} pools for DEX '{dex}'")
            except Exception as e:
                logger.error(f"Error caching pools: {str(e)}")
                logger.error(traceback.format_exc())
        
        return all_pools, success
    
    def get_all_pools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get all pools with caching.
        
        Args:
            force_refresh: Force a fresh collection
            
        Returns:
            List of pool data
        """
        # Check cache first (unless forced)
        if not force_refresh:
            cached_pools = self.cache_manager.get("all_pools")
            if cached_pools:
                logger.info(f"Using {len(cached_pools)} cached pools")
                return cached_pools
        
        # Collect fresh data
        pools, _ = self.collect_data(force=force_refresh)
        return pools
    
    def get_pool_by_id(self, pool_id: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get detailed data for a specific pool.
        
        Args:
            pool_id: Pool ID
            force_refresh: Force a fresh collection
            
        Returns:
            Pool data or None if not found
        """
        # Check cache first (unless forced)
        cache_key = f"pool_{pool_id}"
        if not force_refresh:
            cached_pool = self.cache_manager.get(cache_key)
            if cached_pool:
                logger.info(f"Using cached data for pool {pool_id}")
                return cached_pool
        
        # Try to find in all pools
        pools = self.get_all_pools(force_refresh=force_refresh)
        
        # Find matching pool
        for pool in pools:
            if pool.get("pool_id") == pool_id:
                # Cache for future use
                self.cache_manager.set(cache_key, pool, ttl=300)  # 5 minutes
                return pool
        
        # Not found
        return None
    
    def get_pools_by_token(self, token: str) -> List[Dict[str, Any]]:
        """
        Get pools containing a specific token.
        
        Args:
            token: Token symbol or address
            
        Returns:
            List of matching pools
        """
        # Check cache first
        cache_key = f"pools_by_token_{token}"
        cached_pools = self.cache_manager.get(cache_key)
        if cached_pools:
            logger.info(f"Using {len(cached_pools)} cached pools for token {token}")
            return cached_pools
        
        # Get all pools
        all_pools = self.get_all_pools()
        
        # Find pools containing the token
        matching_pools = []
        token_lower = token.lower()
        
        for pool in all_pools:
            # Check token0
            token0 = pool.get("token0", {})
            token0_symbol = token0.get("symbol", "").lower()
            token0_address = token0.get("address", "").lower()
            
            # Check token1
            token1 = pool.get("token1", {})
            token1_symbol = token1.get("symbol", "").lower()
            token1_address = token1.get("address", "").lower()
            
            # Match on symbol or address
            if (token_lower in token0_symbol or token_lower == token0_address or
                token_lower in token1_symbol or token_lower == token1_address):
                matching_pools.append(pool)
        
        # Cache results
        if matching_pools:
            self.cache_manager.set(cache_key, matching_pools, ttl=300)  # 5 minutes
            
        return matching_pools
    
    def get_pool_history(self, pool_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get historical data for a pool.
        
        Args:
            pool_id: Pool ID
            days: Number of days of history
            
        Returns:
            Dictionary with historical data series
        """
        # TODO: Implement historical data retrieval from database
        # For now, return mock time series data
        
        # Get current pool data for reference
        pool = self.get_pool_by_id(pool_id)
        if not pool:
            return {"error": "Pool not found"}
            
        # Create result structure
        result = {
            "pool_id": pool_id,
            "name": pool.get("name", ""),
            "dates": [],
            "liquidity": [],
            "volume_24h": [],
            "apr": []
        }
            
        # This will be replaced with actual database lookups
        # when historical data collection is implemented
        return result
    
    def start_scheduled_collection(self) -> bool:
        """
        Start scheduled data collection.
        
        Returns:
            True if started, False if already running
        """
        if self.scheduler_running:
            logger.warning("Scheduled collection already running")
            return False
            
        # Define the collection job
        def run_collection():
            logger.info("Running scheduled data collection")
            try:
                self.collect_data()
            except Exception as e:
                logger.error(f"Error in scheduled collection: {str(e)}")
        
        # Schedule collection
        interval_minutes = SCHEDULE_SETTINGS["COLLECTION_INTERVAL_MINUTES"]
        self.scheduler.every(interval_minutes).minutes.do(run_collection)
        
        # Run collection immediately
        run_collection()
        
        # Start scheduler thread
        def run_scheduler():
            while self.scheduler_running:
                self.scheduler.run_pending()
                time.sleep(1)
        
        self.scheduler_thread = threading.Thread(
            target=run_scheduler,
            daemon=True,
            name="DataCollectionScheduler"
        )
        
        self.scheduler_running = True
        self.scheduler_thread.start()
        
        logger.info(f"Started scheduled collection every {interval_minutes} minutes")
        return True
    
    def stop_scheduled_collection(self) -> bool:
        """
        Stop scheduled data collection.
        
        Returns:
            True if stopped, False if not running
        """
        if not self.scheduler_running:
            logger.warning("Scheduled collection not running")
            return False
            
        # Clear the schedule
        self.scheduler.clear()
        
        # Stop the thread
        self.scheduler_running = False
        
        # Wait for thread to stop
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
            
        logger.info("Stopped scheduled collection")
        return True
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        # Service stats
        stats = self.stats.copy()
        
        # Add cache stats
        try:
            cache_stats = self.cache_manager.get_stats()
            stats["cache"] = cache_stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            stats["cache"] = {"error": str(e)}
            
        # Add collector stats
        try:
            collector_stats = self.get_collector_stats()
            stats["collectors"] = collector_stats
        except Exception as e:
            logger.error(f"Error getting collector stats: {str(e)}")
            stats["collectors"] = {"error": str(e)}
            
        # Add scheduler status
        stats["scheduler_running"] = self.scheduler_running
        if self.scheduler_running:
            stats["scheduled_jobs"] = len(self.scheduler.jobs)
            next_job = next(iter(self.scheduler.jobs), None)
            if next_job:
                stats["next_collection"] = str(next_job.next_run)
        
        return stats

# Singleton instance getter
def get_data_service() -> DataService:
    """Get the singleton data service instance."""
    return DataService()

def initialize_data_service() -> DataService:
    """
    Initialize the data service and start scheduled collection.
    
    Returns:
        The data service instance
    """
    # Initialize service
    service = get_data_service()
    
    # Start scheduled collection if not already running
    if not service.scheduler_running:
        service.start_scheduled_collection()
        
    return service