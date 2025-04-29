"""
DefiAggregation API Collector for SolPool Insight.

This collector fetches data from the DefiAggregation API with
advanced rate limiting and error handling.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base_collector import BaseCollector
from ..config import API_CONFIG

# Configure logging
logger = logging.getLogger('defi_aggregation_collector')

class DefiAggregationCollector(BaseCollector):
    """
    Collector for the DefiAggregation API.
    
    Handles pagination, rate limiting, and data standardization.
    """
    
    def __init__(self):
        """Initialize the collector with DefiAggregation API specific settings."""
        # Get API config
        config = API_CONFIG['defi_aggregation']
        
        # Set up rate limiting
        rate_limit = config['rate_limit']
        
        # Initialize base class
        super().__init__(
            name="DefiAggregation",
            priority=8,  # High priority source
            rate_limit_calls=rate_limit['calls_per_minute'],
            rate_limit_period=60,  # 1 minute period
            max_retries=config['retries'],
            backoff_factor=config['backoff_factor'],
            timeout=config['timeout']
        )
        
        # API specific config
        self.base_url = config['base_url']
        self.api_key = os.getenv(config['api_key_env'])
        
        if not self.api_key:
            logger.warning("No API key found for DefiAggregation API")
        
        logger.info(f"Initialized DefiAggregation collector with base URL: {self.base_url}")
    
    def collect(self) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Collect data from the DefiAggregation API.
        
        Returns:
            Tuple of (pool_list, success)
        """
        # Initialize results
        all_pools = []
        success = False
        
        try:
            # Get all pools from API with pagination
            pools = self._get_all_pools()
            
            if not pools:
                logger.warning("No pools returned from DefiAggregation API")
                return [], False
            
            # Standardize data format
            for pool in pools:
                try:
                    standardized = self.standardize_pool_data(pool, source=self.name)
                    all_pools.append(standardized)
                except Exception as e:
                    logger.warning(f"Error standardizing pool data: {str(e)}")
            
            # Update stats
            self.total_data_points += len(all_pools)
            
            # Save backup if we have data
            if all_pools:
                self.save_backup(all_pools)
                success = True
            
            logger.info(f"Collected {len(all_pools)} pools from DefiAggregation API")
            return all_pools, success
            
        except Exception as e:
            logger.error(f"Error collecting from DefiAggregation API: {str(e)}")
            return all_pools, False
    
    def _get_all_pools(self, max_pools: int = 200) -> List[Dict[str, Any]]:
        """
        Get all pools with pagination.
        
        Args:
            max_pools: Maximum number of pools to retrieve
            
        Returns:
            List of pool data
        """
        all_pools = []
        offset = 0
        limit = 20  # Max per request
        unique_pool_ids = set()
        
        while len(all_pools) < max_pools:
            # Prepare headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            # Make request
            url = f"{self.base_url}/pools"
            params = {
                'limit': limit,
                'offset': offset
            }
            
            response = self.make_http_request(
                url=url,
                method='GET',
                params=params,
                headers=headers
            )
            
            if not response:
                logger.warning(f"Failed to get pools at offset {offset}")
                break
            
            # Handle response
            if isinstance(response, list):
                pools = response
            elif isinstance(response, dict) and 'data' in response:
                pools = response.get('data', [])
            else:
                logger.warning(f"Unexpected response format: {type(response)}")
                break
            
            if not pools:
                # No more pools
                break
            
            # Process pools and handle duplicates
            new_pools = 0
            for pool in pools:
                pool_id = pool.get('poolId') or pool.get('id')
                if not pool_id or pool_id in unique_pool_ids:
                    continue
                    
                unique_pool_ids.add(pool_id)
                all_pools.append(pool)
                new_pools += 1
            
            logger.info(f"Retrieved {len(pools)} pools from API, {new_pools} new pools")
            
            if new_pools == 0:
                # We're only getting duplicates now, stop
                logger.info("Only duplicates returned, stopping pagination")
                break
            
            # Update offset for next page
            offset += limit
            
            # Brief pause to avoid hammering the API
            time.sleep(0.2)
        
        logger.info(f"Retrieved a total of {len(all_pools)} unique pools")
        return all_pools
    
    def get_pool_by_id(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed data for a specific pool.
        
        Args:
            pool_id: Pool ID
            
        Returns:
            Pool data or None on failure
        """
        # Prepare headers
        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        
        # Make request
        url = f"{self.base_url}/pools/{pool_id}"
        
        response = self.make_http_request(
            url=url,
            method='GET',
            headers=headers
        )
        
        if not response:
            logger.warning(f"Failed to get pool {pool_id}")
            return None
        
        # Standardize response
        try:
            return self.standardize_pool_data(response, source=self.name)
        except Exception as e:
            logger.error(f"Error standardizing pool {pool_id}: {str(e)}")
            return None

# Create singleton instance
_instance = None

def get_collector() -> DefiAggregationCollector:
    """Get the singleton collector instance."""
    global _instance
    if _instance is None:
        _instance = DefiAggregationCollector()
    return _instance