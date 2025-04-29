"""
Central Data Service for SolPool Insight.

This module provides a unified interface for data collection,
processing, and retrieval with caching and scheduling.
"""

import os
import json
import time
import logging
import threading
import schedule
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path

# Import collectors
from .collectors.defi_aggregation_collector import get_collector as get_defi_collector

# Import cache manager
from .cache.cache_manager import get_cache_manager

# Import database storage
from database.historical_pool_storage import get_storage as get_db_storage

# Import configuration
from .config import COLLECTION_CONFIG, CACHE_CONFIG, API_CONFIG

# Configure logging
logger = logging.getLogger('data_service')

class DataService:
    """
    Central service for data collection and management.
    
    This service coordinates data collection from multiple sources,
    manages caching, and provides a unified API for the application.
    """
    
    def __init__(self):
        """Initialize the data service."""
        self.collectors = []
        self.cache = get_cache_manager()
        self.db_storage = get_db_storage()
        self.config = COLLECTION_CONFIG
        
        # Collection state
        self.last_collection_time = None
        self.collection_running = False
        self.collection_thread = None
        self.scheduler_thread = None
        self.scheduler_running = False
        
        # Initialize collectors
        self._init_collectors()
        
        logger.info("Initialized data service")
    
    def _init_collectors(self):
        """Initialize data collectors."""
        # Add DefiAggregation collector
        self.collectors.append(get_defi_collector())
        
        # Add additional collectors here as they are implemented
        
        logger.info(f"Initialized {len(self.collectors)} data collectors")
    
    def get_collector_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all collectors.
        
        Returns:
            List of collector statistics
        """
        return [collector.get_stats() for collector in self.collectors]
    
    def collect_data(self, force: bool = False) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Collect data from all sources.
        
        Args:
            force: Force collection even if recent
            
        Returns:
            Tuple of (collected_data, success)
        """
        # Prevent concurrent collection
        if self.collection_running:
            logger.warning("Data collection already in progress")
            return [], False
        
        try:
            self.collection_running = True
            
            # Check if we recently collected
            if not force and self.last_collection_time:
                min_interval = timedelta(minutes=self.config['skip_if_recent'])
                elapsed = datetime.now() - self.last_collection_time
                
                if elapsed < min_interval:
                    remaining = (min_interval - elapsed).total_seconds()
                    logger.info(f"Skipping collection, last run was {elapsed.total_seconds():.1f}s ago (wait {remaining:.1f}s more)")
                    return [], True
            
            # Start collection
            start_time = time.time()
            logger.info("Starting data collection from all sources")
            
            all_pools = []
            success = False
            
            # Collect from each source
            for collector in self.collectors:
                try:
                    logger.info(f"Collecting from {collector.name}")
                    pools, source_success = collector.collect()
                    
                    if source_success and pools:
                        logger.info(f"Collected {len(pools)} pools from {collector.name}")
                        all_pools.extend(pools)
                except Exception as e:
                    logger.error(f"Error collecting from {collector.name}: {str(e)}")
            
            # Process results
            if all_pools:
                # Deduplicate by pool ID
                unique_pools = {}
                for pool in all_pools:
                    pool_id = pool.get('poolId') or pool.get('id')
                    if not pool_id:
                        continue
                    
                    # If pool already exists, keep the one from the higher priority source
                    if pool_id in unique_pools:
                        existing = unique_pools[pool_id]
                        existing_source = existing.get('data_source')
                        new_source = pool.get('data_source')
                        
                        # Find collector priorities
                        existing_priority = 0
                        new_priority = 0
                        
                        for collector in self.collectors:
                            if collector.name == existing_source:
                                existing_priority = collector.priority
                            if collector.name == new_source:
                                new_priority = collector.priority
                        
                        # Only replace if new source has higher priority
                        if new_priority > existing_priority:
                            unique_pools[pool_id] = pool
                    else:
                        unique_pools[pool_id] = pool
                
                # Convert back to list
                all_pools = list(unique_pools.values())
                
                # Store in database
                try:
                    if self.db_storage:
                        stored, total = self.db_storage.store_multiple_snapshots(all_pools)
                        logger.info(f"Stored {stored}/{total} pool snapshots in database")
                except Exception as e:
                    logger.error(f"Error storing pools in database: {str(e)}")
                
                # Update cache
                cache_ttl = CACHE_CONFIG['default_ttl']
                pool_list_key = 'all_pools'
                self.cache.set(pool_list_key, all_pools, ttl=cache_ttl)
                
                # Cache individual pools
                for pool in all_pools:
                    pool_id = pool.get('poolId') or pool.get('id')
                    if pool_id:
                        pool_key = f"pool:{pool_id}"
                        self.cache.set(pool_key, pool, ttl=cache_ttl)
                
                # Cache successful
                success = True
            
            # Update collection time
            self.last_collection_time = datetime.now()
            
            # Finish
            duration = time.time() - start_time
            logger.info(f"Data collection completed in {duration:.2f}s, collected {len(all_pools)} pools")
            
            return all_pools, success
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            return [], False
        finally:
            self.collection_running = False
    
    def get_all_pools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get all pools with caching.
        
        Args:
            force_refresh: Force a fresh collection
            
        Returns:
            List of pool data
        """
        # Check cache first if not forcing refresh
        if not force_refresh:
            cached = self.cache.get('all_pools')
            if cached:
                logger.debug(f"Returning {len(cached)} pools from cache")
                return cached
        
        # Collect fresh data
        pools, success = self.collect_data(force=force_refresh)
        if success and pools:
            return pools
        
        # Fall back to database if collection failed
        try:
            if self.db_storage:
                # Get the most recent snapshot for each pool
                pool_ids = self.db_storage.get_all_pools()
                pools = []
                
                for pool_id in pool_ids:
                    snapshot = self.db_storage.get_latest_snapshot(pool_id)
                    if snapshot:
                        pools.append(snapshot)
                
                if pools:
                    logger.info(f"Returning {len(pools)} pools from database")
                    return pools
        except Exception as e:
            logger.error(f"Error retrieving pools from database: {str(e)}")
        
        # No data available
        return []
    
    def get_pool_by_id(self, pool_id: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get detailed data for a specific pool.
        
        Args:
            pool_id: Pool ID
            force_refresh: Force a fresh collection
            
        Returns:
            Pool data or None if not found
        """
        # Check cache first if not forcing refresh
        cache_key = f"pool:{pool_id}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Returning pool {pool_id} from cache")
                return cached
        
        # Try to get from collectors
        for collector in self.collectors:
            if hasattr(collector, 'get_pool_by_id'):
                try:
                    pool = collector.get_pool_by_id(pool_id)
                    if pool:
                        # Cache the result
                        self.cache.set(cache_key, pool, ttl=CACHE_CONFIG['default_ttl'])
                        return pool
                except:
                    pass
        
        # Try to get from database
        try:
            if self.db_storage:
                snapshot = self.db_storage.get_latest_snapshot(pool_id)
                if snapshot:
                    # Cache the result
                    self.cache.set(cache_key, snapshot, ttl=CACHE_CONFIG['default_ttl'])
                    return snapshot
        except Exception as e:
            logger.error(f"Error retrieving pool {pool_id} from database: {str(e)}")
        
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
        cache_key = f"token_pools:{token}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug(f"Returning {len(cached)} pools for token {token} from cache")
            return cached
        
        # Get all pools and filter
        all_pools = self.get_all_pools()
        matching_pools = []
        
        for pool in all_pools:
            # Check token1 and token2
            token1 = pool.get('token1', '')
            token2 = pool.get('token2', '')
            
            # Also check tokens array
            tokens = pool.get('tokens', [])
            token_symbols = [t.get('symbol', '') for t in tokens]
            
            if (token1 and token1.lower() == token.lower()) or \
               (token2 and token2.lower() == token.lower()) or \
               any(t.lower() == token.lower() for t in token_symbols):
                matching_pools.append(pool)
        
        # Cache the result with longer TTL since token-pool mappings change less frequently
        self.cache.set(cache_key, matching_pools, ttl=CACHE_CONFIG['long_ttl'])
        
        logger.info(f"Found {len(matching_pools)} pools containing token {token}")
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
        # Check cache first for shorter lookbacks
        if days <= 7:
            cache_key = f"pool_history:{pool_id}:{days}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Returning {days}-day history for pool {pool_id} from cache")
                return cached
        
        # Get from database
        try:
            if self.db_storage:
                df = self.db_storage.get_pool_history(pool_id, days=days)
                
                if not df.empty:
                    # Convert to serializable format
                    result = {
                        'pool_id': pool_id,
                        'days': days,
                        'timestamps': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                        'tvl': df['tvl'].tolist() if 'tvl' in df else [],
                        'apy': df['apy'].tolist() if 'apy' in df else [],
                        'volume_24h': df['volume_24h'].tolist() if 'volume_24h' in df else [],
                        'token1_price': df['token1_price'].tolist() if 'token1_price' in df else [],
                        'token2_price': df['token2_price'].tolist() if 'token2_price' in df else [],
                        'price_ratio': df['price_ratio'].tolist() if 'price_ratio' in df else []
                    }
                    
                    # Cache shorter lookbacks
                    if days <= 7:
                        self.cache.set(cache_key, result, ttl=CACHE_CONFIG['long_ttl'])
                    
                    return result
        except Exception as e:
            logger.error(f"Error retrieving history for pool {pool_id}: {str(e)}")
        
        # No data
        return {
            'pool_id': pool_id,
            'days': days,
            'timestamps': [],
            'tvl': [],
            'apy': [],
            'volume_24h': [],
            'token1_price': [],
            'token2_price': [],
            'price_ratio': []
        }
    
    def start_scheduled_collection(self) -> bool:
        """
        Start scheduled data collection.
        
        Returns:
            True if started, False if already running
        """
        if self.scheduler_running:
            logger.warning("Scheduled collection already running")
            return False
        
        # Configure the scheduler
        interval_minutes = self.config['interval_minutes']
        
        # Clear any existing jobs
        schedule.clear()
        
        # Schedule the collection job
        schedule.every(interval_minutes).minutes.do(self.collect_data)
        
        logger.info(f"Scheduled data collection every {interval_minutes} minutes")
        
        # Start the scheduler thread
        def run_scheduler():
            self.scheduler_running = True
            
            # Run initial collection after a brief delay
            time.sleep(self.config['initial_delay'])
            self.collect_data()
            
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(1)
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        return True
    
    def stop_scheduled_collection(self) -> bool:
        """
        Stop scheduled data collection.
        
        Returns:
            True if stopped, False if not running
        """
        if not self.scheduler_running:
            return False
        
        # Stop the scheduler
        self.scheduler_running = False
        schedule.clear()
        
        # Wait for thread to finish if it's running
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=2)
        
        logger.info("Stopped scheduled data collection")
        return True
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            'collection': {
                'last_collection': self.last_collection_time.isoformat() if self.last_collection_time else None,
                'collection_running': self.collection_running,
                'scheduler_running': self.scheduler_running,
                'collectors': self.get_collector_stats()
            },
            'cache': self.cache.get_stats()
        }
        
        # Add database stats if available
        try:
            if self.db_storage:
                stats['database'] = self.db_storage.get_stats()
        except Exception as e:
            logger.error(f"Error retrieving database stats: {str(e)}")
            stats['database'] = {'error': str(e)}
        
        return stats

# Create singleton instance
_instance = None

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
    service = get_data_service()
    service.start_scheduled_collection()
    return service