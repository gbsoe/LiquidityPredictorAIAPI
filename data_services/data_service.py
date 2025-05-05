"""
Central Data Service for SolPool Insight.

This module provides a unified interface for data collection,
processing, and retrieval with caching and scheduling.
"""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Import collectors
from .collectors import get_defi_collector
from .cache import get_cache_manager

# Import historical data service
import sys
sys.path.append(".")  # Add current directory to path for imports
from historical_data_service import get_historical_service

# Singleton instance
_instance = None

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
        # Only initialize once
        if self._initialized:
            return

        # Set up collectors
        self._collectors = []
        self._init_collectors()

        # Set up cache
        self._cache_manager = get_cache_manager()

        # Set up scheduling
        self.scheduler_running = False
        self.scheduler_thread = None
        self.collection_interval = 15 * 60  # 15 minutes

        # Set up stats
        self.stats = {
            "startup_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_collection_time": None,
            "last_collection_pool_count": 0,
            "total_collections": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "average_collection_time": 0,
            "average_pools_count": 0
        }

        # Mark as initialized
        self._initialized = True
        logger.info("Data service initialized")

    def _init_collectors(self):
        """Initialize data collectors."""
        # Check if we have an API key in streamlit session state
        api_key = None

        # Try to access streamlit's session state if it's available
        try:
            import streamlit as st
            if "defi_api_key" in st.session_state and st.session_state["defi_api_key"]:
                api_key = st.session_state["defi_api_key"]
                logger.info("Using API key from Streamlit session state")

                # Also set it in the environment for other components
                os.environ["DEFI_API_KEY"] = api_key
        except (ImportError, RuntimeError):
            # Either streamlit is not installed or we're not in a streamlit context
            logger.debug("Not running in Streamlit context or streamlit not installed")

        # Add DeFi Aggregation API collector with the API key if available
        defi_collector = get_defi_collector(api_key=api_key)
        self._collectors.append(defi_collector)

        # Log collectors
        logger.info(f"Initialized {len(self._collectors)} collectors")

    def get_collector_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all collectors.

        Returns:
            List of collector statistics
        """
        return [collector.get_stats() for collector in self._collectors]

    def collect_data(self, force: bool = False) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Collect data from all sources.

        Args:
            force: Force collection even if recent

        Returns:
            Tuple of (collected_data, success)
        """
        # Check if we need to collect
        if not force and self.stats["last_collection_time"]:
            # Check if we have recent data
            last_time = datetime.strptime(self.stats["last_collection_time"], "%Y-%m-%d %H:%M:%S")
            time_diff = (datetime.now() - last_time).total_seconds()

            if time_diff < self.collection_interval / 2:
                logger.info(f"Skipping collection, last collection was {time_diff:.1f}s ago")

                # Return cached data
                cached_pools = self._cache_manager.get("all_pools", [])
                return cached_pools, True

        # Collect from all sources
        logger.info("Starting data collection from all sources")
        start_time = time.time()

        all_data = []
        success = True

        # Use each collector
        for collector in self._collectors:
            try:
                # Collect data
                data = collector.collect()

                if data:
                    logger.info(f"Collected {len(data)} items from {collector.name}")
                    all_data.extend(data)
                else:
                    logger.warning(f"No data collected from {collector.name}")
            except Exception as e:
                logger.error(f"Error collecting from {collector.name}: {str(e)}")
                success = False

        # Update stats
        elapsed_time = time.time() - start_time
        self.stats["last_collection_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats["last_collection_pool_count"] = len(all_data)
        self.stats["total_collections"] += 1

        if success:
            self.stats["successful_collections"] += 1
        else:
            self.stats["failed_collections"] += 1

        # Update averages
        self.stats["average_collection_time"] = (
            (self.stats["average_collection_time"] * (self.stats["total_collections"] - 1) + elapsed_time) / 
            self.stats["total_collections"]
        )
        self.stats["average_pools_count"] = (
            (self.stats["average_pools_count"] * (self.stats["total_collections"] - 1) + len(all_data)) / 
            self.stats["total_collections"]
        )

        # Cache the results
        if all_data:
            self._cache_manager.set("all_pools", all_data)

            # Store historical data for later analysis
            try:
                historical_service = get_historical_service()
                historical_service.collect_pool_data(all_data)
                logger.info(f"Stored historical snapshot of {len(all_data)} pools")
            except Exception as e:
                logger.error(f"Error storing historical data: {str(e)}")

        # Log results
        logger.info(
            f"Data collection completed in {elapsed_time:.2f}s: " +
            f"{len(all_data)} items, success: {success}"
        )

        return all_data, success

    def get_all_pools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get all pools with caching.

        Args:
            force_refresh: Force a fresh collection

        Returns:
            List of pool data
        """
        # Check cache first
        if not force_refresh:
            cached_pools = self._cache_manager.get("all_pools", None)
            if cached_pools:
                logger.info(f"Retrieved {len(cached_pools)} pools from cache")
                return cached_pools

        # If not cached or forced refresh, collect
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
        # Check cache first using a pool-specific key
        cache_key = f"pool:{pool_id}"

        if not force_refresh:
            cached_pool = self._cache_manager.get(cache_key, None)
            if cached_pool:
                logger.info(f"Retrieved pool {pool_id} from cache")
                return cached_pool

        # Try to find in all pools first (cheaper than a specific request)
        all_pools = self.get_all_pools(force_refresh=force_refresh)

        for pool in all_pools:
            if pool.get("id") == pool_id or pool.get("poolId") == pool_id:
                # Cache for future use
                self._cache_manager.set(cache_key, pool)
                return pool

        # If not found in all pools, try a specific collector
        # DeFi Aggregation API has a specific endpoint for individual pools
        try:
            defi_collector = get_defi_collector()
            pool = defi_collector.get_pool_by_id(pool_id)

            if pool:
                # Cache for future use
                self._cache_manager.set(cache_key, pool)
                return pool
        except Exception as e:
            logger.error(f"Error getting pool {pool_id}: {str(e)}")

        # Not found
        logger.warning(f"Pool {pool_id} not found")
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
        cache_key = f"token_pools:{token}"
        cached_pools = self._cache_manager.get(cache_key, None)

        if cached_pools:
            logger.info(f"Retrieved {len(cached_pools)} pools for token {token} from cache")
            return cached_pools

        # First try to filter the existing pools
        all_pools = self.get_all_pools()
        matching_pools = []

        for pool in all_pools:
            # Check both tokens in the pool
            token1_symbol = pool.get("token1_symbol", "").lower()
            token2_symbol = pool.get("token2_symbol", "").lower()
            token1_address = pool.get("token1_address", "").lower()
            token2_address = pool.get("token2_address", "").lower()

            # Match against the token
            token_lower = token.lower()
            if (
                token_lower == token1_symbol or
                token_lower == token2_symbol or
                token_lower == token1_address or
                token_lower == token2_address
            ):
                matching_pools.append(pool)

        # If we found matches, cache and return
        if matching_pools:
            self._cache_manager.set(cache_key, matching_pools)
            return matching_pools

        # If no matches in existing pools, try specific API
        try:
            defi_collector = get_defi_collector()
            token_pools = defi_collector.get_pools_by_token(token)

            if token_pools:
                # Cache for future use
                self._cache_manager.set(cache_key, token_pools)
                return token_pools
        except Exception as e:
            logger.error(f"Error getting pools for token {token}: {str(e)}")

        # Return empty list if nothing found
        return []

    def get_pool_history(self, pool_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get historical data for a pool.

        Args:
            pool_id: Pool ID
            days: Number of days of history

        Returns:
            Dictionary with historical data series
        """
        # Use the historical data service
        historical_service = get_historical_service()
        history = historical_service.get_pool_history(pool_id, days)

        # Extract metric histories
        liquidity_history = historical_service.get_metric_history(pool_id, "liquidity", days)
        volume_history = historical_service.get_metric_history(pool_id, "volume_24h", days)
        apr_history = historical_service.get_metric_history(pool_id, "apr_24h", days)

        # Format for presentation
        result = {
            "pool_id": pool_id,
            "days": days,
            "data_points": len(history),
            "has_history": len(history) > 0,
            "history": history,
            "metrics": {
                "liquidity": liquidity_history,
                "volume_24h": volume_history,
                "apr_24h": apr_history
            },
            "first_snapshot": history[0]["snapshot_timestamp"] if history else None,
            "last_snapshot": history[-1]["snapshot_timestamp"] if history else None
        }

        return result

    def start_scheduled_collection(self) -> bool:
        """
        Start scheduled data collection.

        Returns:
            True if started, False if already running
        """
        if self.scheduler_running:
            logger.warning("Scheduler already running")
            return False

        # Set up collection function
        def run_collection():
            logger.info("Running scheduled data collection")
            self.collect_data(force=True)

        # Set up scheduler thread
        def run_scheduler():
            self.scheduler_running = True
            logger.info(f"Starting scheduled collection every {self.collection_interval}s")

            while self.scheduler_running:
                try:
                    # Run collection
                    run_collection()
                except Exception as e:
                    logger.error(f"Error in scheduled collection: {str(e)}")

                # Sleep until next collection
                for _ in range(self.collection_interval):
                    # Check if still running every second
                    if not self.scheduler_running:
                        break
                    time.sleep(1)

        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

        logger.info("Scheduled collection started")
        return True

    def stop_scheduled_collection(self) -> bool:
        """
        Stop scheduled data collection.

        Returns:
            True if stopped, False if not running
        """
        if not self.scheduler_running:
            logger.warning("Scheduler not running")
            return False

        # Stop scheduler
        self.scheduler_running = False

        # Wait for thread to exit
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(1.0)

        logger.info("Scheduled collection stopped")
        return True

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary with system statistics
        """
        # Get data service stats
        stats = self.stats.copy()

        # Add collector stats
        stats["collectors"] = self.get_collector_stats()

        # Add cache stats
        stats["cache"] = self._cache_manager.get_stats()

        # Add system info
        stats["scheduler_running"] = self.scheduler_running
        stats["collection_interval"] = self.collection_interval

        return stats

def get_data_service() -> DataService:
    """Get the singleton data service instance."""
    global _instance
    if _instance is None:
        _instance = DataService()
    return _instance

def initialize_data_service() -> DataService:
    """
    Initialize the data service and start scheduled collection.

    Returns:
        The data service instance
    """
    # Get data service
    data_service = get_data_service()

    # Start scheduled collection
    data_service.start_scheduled_collection()

    return data_service