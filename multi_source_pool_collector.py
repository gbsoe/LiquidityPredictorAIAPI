"""
Multi-Source Pool Collector for SolPool Insight

This module integrates data from multiple sources to provide more comprehensive
liquidity pool coverage while maintaining data quality and consistency.
"""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from defi_aggregation_api import DefiAggregationAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='multi_source_collector.log'
)
logger = logging.getLogger('multi_source')

class PoolDataSource:
    """Base class for pool data sources"""
    
    def __init__(self, name: str, priority: int = 5):
        """
        Initialize a data source.
        
        Args:
            name: Name of the data source
            priority: Priority level (1-10, higher means more authoritative)
        """
        self.name = name
        self.priority = priority
        self.enable_rate_limiting = True
        self.rate_limit_calls = 10
        self.rate_limit_period = 1  # seconds
        self.last_call_time = 0
        self.success_rate = 1.0  # Start optimistic
        
    def collect_pools(self) -> List[Dict[str, Any]]:
        """Collect pool data from this source"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def apply_rate_limiting(self):
        """Apply rate limiting if enabled"""
        if not self.enable_rate_limiting:
            return
            
        # Calculate minimum interval between calls
        min_interval = self.rate_limit_period / self.rate_limit_calls
        
        # Check if we need to wait
        elapsed = time.time() - self.last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            
        # Update last call time
        self.last_call_time = time.time()
    
    def update_success_rate(self, success: bool):
        """Update the success rate tracking"""
        # Simple exponential moving average
        alpha = 0.1  # Weight for most recent result
        self.success_rate = (alpha * (1.0 if success else 0.0)) + ((1 - alpha) * self.success_rate)


class DefiAggregationSource(PoolDataSource):
    """Data source using the DefiAggregation API"""
    
    def __init__(self):
        super().__init__(name="DefiAggregation", priority=8)
        self.api = DefiAggregationAPI()
        self.rate_limit_calls = 10
        self.rate_limit_period = 1  # seconds
        
    def collect_pools(self) -> List[Dict[str, Any]]:
        """Collect pool data from the DefiAggregation API"""
        try:
            self.apply_rate_limiting()
            
            # Get pools with pagination
            pools = self.api.get_all_pools(max_pools=500)
            
            # Mark the source
            for pool in pools:
                pool['data_source'] = self.name
                
            self.update_success_rate(True)
            return pools
            
        except Exception as e:
            logger.error(f"Error collecting from {self.name}: {str(e)}")
            self.update_success_rate(False)
            return []


class RaydiumDirectSource(PoolDataSource):
    """Direct connection to Raydium API (simulated for now)"""
    
    def __init__(self):
        super().__init__(name="RaydiumDirect", priority=7)
        # In a real implementation, this would initialize the Raydium client
        
    def collect_pools(self) -> List[Dict[str, Any]]:
        """Collect pool data directly from Raydium"""
        try:
            self.apply_rate_limiting()
            
            # Simulated for this implementation
            # In reality, this would make direct API calls to Raydium
            logger.info("Would connect directly to Raydium API here")
            
            # For now, return empty list - in production would return actual pools
            self.update_success_rate(True)
            return []
            
        except Exception as e:
            logger.error(f"Error collecting from {self.name}: {str(e)}")
            self.update_success_rate(False)
            return []


class MultiSourcePoolCollector:
    """
    Collects and merges pool data from multiple sources with 
    conflict resolution and quality checks.
    """
    
    def __init__(self, backup_dir: str = "./data/multi_source"):
        """
        Initialize the multi-source collector.
        
        Args:
            backup_dir: Directory for backing up collected data
        """
        self.sources = []
        self.backup_dir = backup_dir
        self.pool_deduplication_keys = ['poolId']  # Fields used to identify unique pools
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Add default sources
        self.add_source(DefiAggregationSource())
        # Don't enable this until implemented
        # self.add_source(RaydiumDirectSource())
    
    def add_source(self, source: PoolDataSource):
        """Add a data source to the collector"""
        self.sources.append(source)
        # Sort sources by priority (highest first)
        self.sources.sort(key=lambda s: s.priority, reverse=True)
        
    def collect_from_all_sources(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect data from all sources in parallel.
        
        Returns:
            Dict mapping source names to lists of pools
        """
        results = {}
        
        # Using ThreadPoolExecutor for parallel collection
        with ThreadPoolExecutor(max_workers=min(len(self.sources), 3)) as executor:
            # Submit collection jobs
            future_to_source = {
                executor.submit(source.collect_pools): source
                for source in self.sources
            }
            
            # Gather results as they complete
            for future in future_to_source:
                source = future_to_source[future]
                try:
                    pools = future.result()
                    logger.info(f"Collected {len(pools)} pools from {source.name}")
                    results[source.name] = pools
                except Exception as e:
                    logger.error(f"Failed to collect from {source.name}: {str(e)}")
                    results[source.name] = []
        
        return results
    
    def merge_pools(self, source_pools: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Merge pools from different sources with conflict resolution.
        
        Args:
            source_pools: Dict mapping source names to lists of pools
            
        Returns:
            List of merged pools with conflicts resolved
        """
        merged_pools = {}  # poolId -> pool data
        
        # Process each source in priority order
        for source in self.sources:
            if source.name not in source_pools:
                continue
                
            pools = source_pools[source.name]
            for pool in pools:
                # Generate a unique key for this pool
                pool_key = self._get_pool_key(pool)
                if not pool_key:
                    continue
                
                if pool_key not in merged_pools:
                    # This is a new pool, add it
                    merged_pools[pool_key] = pool.copy()
                    merged_pools[pool_key]['sources'] = [source.name]
                else:
                    # This pool exists, handle conflicts based on source priority
                    existing_pool = merged_pools[pool_key]
                    
                    # Update the sources list
                    if 'sources' in existing_pool:
                        existing_pool['sources'].append(source.name)
                    
                    # Only update if the current source has higher priority
                    # than sources we've already seen
                    if all(self._get_source_priority(s) < source.priority 
                          for s in existing_pool.get('sources', [])):
                        
                        # Selectively update fields
                        self._update_pool_fields(existing_pool, pool)
        
        # Convert back to list
        return list(merged_pools.values())
    
    def _get_pool_key(self, pool: Dict[str, Any]) -> Optional[str]:
        """
        Generate a unique key for a pool.
        
        Args:
            pool: Pool data
            
        Returns:
            Unique key or None if required fields missing
        """
        key_parts = []
        for key in self.pool_deduplication_keys:
            if key not in pool or not pool[key]:
                return None
            key_parts.append(str(pool[key]))
        
        return "::".join(key_parts)
    
    def _get_source_priority(self, source_name: str) -> int:
        """Get the priority level of a source by name"""
        for source in self.sources:
            if source.name == source_name:
                return source.priority
        return 0
    
    def _update_pool_fields(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        Update fields in the target pool with data from source pool.
        Only updates fields that exist in both pools.
        """
        # Fields to exclude from updates
        excluded_fields = {'id', 'data_source', 'sources'}
        
        for key, value in source.items():
            if key in excluded_fields:
                continue
                
            if key in target:
                target[key] = value
    
    def collect_and_merge(self) -> List[Dict[str, Any]]:
        """
        Collect from all sources and merge the results.
        
        Returns:
            List of merged pool data
        """
        start_time = time.time()
        logger.info("Starting multi-source pool collection")
        
        try:
            # Collect from all sources
            source_pools = self.collect_from_all_sources()
            
            # Count pools from each source
            for source_name, pools in source_pools.items():
                logger.info(f"Source {source_name}: {len(pools)} pools")
            
            # Merge pools
            merged_pools = self.merge_pools(source_pools)
            logger.info(f"Merged into {len(merged_pools)} unique pools")
            
            # Add timestamp
            timestamp = datetime.now()
            collection_id = timestamp.strftime("%Y%m%d_%H%M%S")
            
            for pool in merged_pools:
                pool['collection_timestamp'] = timestamp.isoformat()
            
            # Create a backup
            if merged_pools:
                backup_file = os.path.join(self.backup_dir, f"multi_source_{collection_id}.json")
                try:
                    with open(backup_file, 'w') as f:
                        json.dump(merged_pools, f)
                    logger.info(f"Backed up {len(merged_pools)} pools to {backup_file}")
                except Exception as e:
                    logger.error(f"Failed to create backup: {str(e)}")
            
            duration = time.time() - start_time
            logger.info(f"Multi-source collection completed in {duration:.2f} seconds")
            
            return merged_pools
            
        except Exception as e:
            logger.error(f"Error in multi-source collection: {str(e)}")
            return []
            
    def get_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about each source.
        
        Returns:
            Dict mapping source names to stats
        """
        stats = {}
        for source in self.sources:
            stats[source.name] = {
                "priority": source.priority,
                "success_rate": source.success_rate,
                "enabled": True
            }
        return stats

# Singleton instance
collector = None

def get_collector() -> MultiSourcePoolCollector:
    """Get the singleton collector instance"""
    global collector
    if collector is None:
        collector = MultiSourcePoolCollector()
    return collector

if __name__ == "__main__":
    # Simple test
    collector = get_collector()
    pools = collector.collect_and_merge()
    print(f"Collected {len(pools)} unique pools")