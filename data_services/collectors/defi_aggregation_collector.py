"""
DeFi Aggregation API Collector for SolPool Insight.

This collector retrieves data from the DeFi Aggregation API.
"""

import logging
import os
import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from defi_aggregation_api import DefiAggregationAPI
from ..collectors.base_collector import BaseCollector

# Configure logging
logger = logging.getLogger(__name__)

# Singleton instance
_instance = None

class DefiAggregationCollector(BaseCollector):
    """
    Collector for DeFi Aggregation API data.
    
    This collector handles:
    - Authentication with API key
    - Rate limiting for API requests
    - Data transformation to unified format
    - Backoff strategy for errors
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the DeFi API collector.
        
        Args:
            api_key: API key for authentication (defaults to config)
            base_url: Base URL for the API (defaults to config)
        """
        # Initialize the base collector with a name
        super().__init__(name="DeFi Aggregation API")
        
        # Get API key from parameters or environment
        self.api_key = api_key or os.getenv("DEFI_API_KEY")
        if not self.api_key:
            logger.warning("No DEFI_API_KEY found in environment or parameters")
        
        # Create API client
        self.api_client = DefiAggregationAPI(api_key=self.api_key, base_url=base_url)
        
        # Collection configuration
        self.max_pools_per_collection = 100  # Increased maximum pools per collection
        self.delay_between_requests = 0.1     # 100ms delay for rate limiting
        self.max_pages_per_dex = 3           # Maximum pages to fetch per DEX
        self.page_size = 16                  # Standard page size for pagination
        
        # Cache DEX list
        self.supported_dexes = []
        
        # Historical data tracking
        self.last_collection_time = None
        self.historical_collections = []
        
        logger.info("DefiAggregationCollector initialized")
    
    def get_supported_dexes(self) -> List[str]:
        """
        Get a list of supported DEXes.
        
        Returns:
            List of DEX names
        """
        if not self.supported_dexes:
            try:
                self.supported_dexes = self.api_client.get_supported_dexes()
                logger.info(f"Retrieved {len(self.supported_dexes)} supported DEXes")
            except Exception as e:
                logger.error(f"Error getting supported DEXes: {str(e)}")
                # Fall back to default list
                self.supported_dexes = ["Raydium", "Orca", "Meteora"]
        
        return self.supported_dexes
    
    def get_pools_by_dex(self, dex: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get pools for a specific DEX.
        
        Args:
            dex: DEX name
            limit: Maximum number of pools to retrieve
            
        Returns:
            List of pool data
        """
        try:
            # Use the API client to get pools by DEX
            pools = self.api_client.get_pools_by_dex(dex=dex, limit=limit)
            logger.info(f"Retrieved {len(pools)} pools for DEX {dex}")
            return pools
        except Exception as e:
            logger.error(f"Error getting pools for DEX {dex}: {str(e)}")
            return []
    
    def get_pool_by_id(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific pool by ID.
        
        Args:
            pool_id: Pool ID
            
        Returns:
            Pool data or None if not found
        """
        try:
            # Use the API client to get a specific pool
            pool = self.api_client.get_pool_by_id(pool_id=pool_id)
            return pool
        except Exception as e:
            logger.error(f"Error getting pool {pool_id}: {str(e)}")
            return None
    
    def get_pools_by_token(self, token: str, limit: int = 20, max_pages: int = 5) -> List[Dict[str, Any]]:
        """
        Get pools containing a specific token with pagination support.
        
        Args:
            token: Token symbol or address
            limit: Maximum number of pools to retrieve per page
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of pool data
        """
        try:
            all_token_pools = []
            collection_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Fetch pools with pagination
            for page in range(max_pages):
                # Add a small delay between requests
                if page > 0:
                    time.sleep(self.delay_between_requests)
                
                # Calculate offset for pagination
                offset = page * limit
                
                # Log pagination information
                if page > 0:
                    logger.info(f"Fetching page {page+1} for token {token} (offset: {offset})")
                
                # Get pools for this token and page
                page_pools = self.api_client.get_pools_by_token(
                    token=token, 
                    limit=limit, 
                    offset=offset
                )
                
                if page_pools:
                    logger.info(f"Retrieved {len(page_pools)} pools for token {token} (page {page+1})")
                    all_token_pools.extend(page_pools)
                    
                    # Add collection metadata to each pool for historical tracking
                    for pool in page_pools:
                        pool['collection_timestamp'] = collection_timestamp
                        pool['collection_source'] = f"token-{token}-API"
                    
                    # If we got fewer results than requested, we've reached the end
                    if len(page_pools) < limit:
                        logger.info(f"Reached end of results for token {token} at page {page+1}")
                        break
                else:
                    # No more results for this token
                    if page == 0:
                        logger.warning(f"No pools found for token {token}")
                    else:
                        logger.info(f"No more pools available for token {token} after page {page}")
                    break
            
            logger.info(f"Total: Retrieved {len(all_token_pools)} pools for token {token}")
            return all_token_pools
        except Exception as e:
            logger.error(f"Error getting pools for token {token}: {str(e)}")
            return []
    
    def _collect_data(self) -> List[Dict[str, Any]]:
        """
        Collect pool data from the DeFi Aggregation API with pagination support.
        
        Returns:
            List of collected pool data
        """
        # Check if we have an API key
        if not self.api_key:
            logger.error("Cannot collect data: No API key provided")
            raise ValueError("No API key provided for DeFi Aggregation API")
        
        # Start collection
        logger.info(f"Starting pool data collection from DeFi Aggregation API")
        start_time = time.time()
        collection_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Try to get supported DEXes
        try:
            dexes = self.get_supported_dexes()
            logger.info(f"Will collect pools from {len(dexes)} DEXes: {', '.join(dexes)}")
        except Exception as e:
            # Just log and continue with all pools
            logger.warning(f"Could not get supported DEXes: {str(e)}")
            dexes = []
        
        # Collect pools using the optimal API collection strategy with pagination
        all_pools = []
        
        # If we have a list of DEXes, collect pools for each DEX with pagination
        if dexes:
            for dex in dexes:
                dex_pools = []
                try:
                    # Implement pagination: collect multiple pages of results for each DEX
                    for page in range(self.max_pages_per_dex):
                        # Add a small delay between requests
                        time.sleep(self.delay_between_requests)
                        
                        # Calculate offset for pagination
                        offset = page * self.page_size
                        
                        # Log pagination information
                        if page > 0:
                            logger.info(f"Fetching page {page+1} for DEX {dex} (offset: {offset})")
                        
                        # Get pools for this DEX and page
                        page_pools = self.api_client.get_pools_by_dex(
                            dex=dex, 
                            limit=self.page_size, 
                            offset=offset
                        )
                        
                        if page_pools:
                            logger.info(f"Retrieved {len(page_pools)} pools for DEX {dex} (page {page+1})")
                            dex_pools.extend(page_pools)
                            
                            # Add collection metadata to each pool for historical tracking
                            for pool in page_pools:
                                pool['collection_timestamp'] = collection_timestamp
                                pool['collection_source'] = f"{dex}-API"
                            
                            # If we got fewer results than requested, we've reached the end
                            if len(page_pools) < self.page_size:
                                logger.info(f"Reached end of results for DEX {dex} at page {page+1}")
                                break
                        else:
                            # No more results for this DEX
                            if page == 0:
                                logger.warning(f"No pools found for DEX {dex}")
                            else:
                                logger.info(f"No more pools available for DEX {dex} after page {page}")
                            break
                    
                    # Add the collected pools from this DEX to the overall pool list
                    if dex_pools:
                        logger.info(f"Total: Retrieved {len(dex_pools)} pools for DEX {dex}")
                        all_pools.extend(dex_pools)
                
                except Exception as e:
                    logger.error(f"Error collecting pools for DEX {dex}: {str(e)}")
        
        # If we didn't get any pools from DEX-specific requests, try the general endpoint
        if not all_pools:
            try:
                logger.info("Falling back to general pools endpoint")
                all_pools = self.api_client.get_all_pools(max_pools=self.max_pools_per_collection)
                
                # Add collection metadata to each pool
                for pool in all_pools:
                    pool['collection_timestamp'] = collection_timestamp
                    pool['collection_source'] = "general-API"
                
                logger.info(f"Retrieved {len(all_pools)} pools from general endpoint")
            except Exception as e:
                logger.error(f"Error collecting pools from general endpoint: {str(e)}")
                raise  # Re-raise the exception
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Update historical data tracking
        self.last_collection_time = collection_timestamp
        self._update_historical_data(all_pools, collection_timestamp, elapsed_time)
        
        # Log collection results
        logger.info(
            f"Pool data collection completed: {len(all_pools)} pools in {elapsed_time:.2f}s "
            f"({len(all_pools) / elapsed_time:.2f} pools/s)"
        )
        
        return all_pools
        
    def _update_historical_data(self, pools: List[Dict[str, Any]], timestamp: str, collection_time: float = 0) -> None:
        """
        Update historical data tracking.
        
        Args:
            pools: List of collected pools
            timestamp: Collection timestamp
            collection_time: Time taken to collect the pools in seconds
        """
        # Create a historical snapshot record
        snapshot = {
            'timestamp': timestamp,
            'pool_count': len(pools),
            'dexes': {},
            'collection_rate': len(pools) / collection_time if collection_time > 0 else 0
        }
        
        # Aggregate DEX statistics
        for pool in pools:
            dex = pool.get('source', 'unknown')
            if dex not in snapshot['dexes']:
                snapshot['dexes'][dex] = 0
            snapshot['dexes'][dex] += 1
        
        # Add to historical collections (limit to last 100 for memory usage)
        self.historical_collections.append(snapshot)
        if len(self.historical_collections) > 100:
            self.historical_collections = self.historical_collections[-100:]
            
    def get_historical_stats(self) -> List[Dict[str, Any]]:
        """
        Get historical collection statistics.
        
        Returns:
            List of historical collection records
        """
        return self.historical_collections

def get_collector(api_key: Optional[str] = None, 
                 base_url: Optional[str] = None) -> DefiAggregationCollector:
    """
    Get the singleton collector instance.
    
    Args:
        api_key: Optional API key
        base_url: Optional base URL
        
    Returns:
        DefiAggregationCollector instance
    """
    global _instance
    if _instance is None:
        _instance = DefiAggregationCollector(api_key=api_key, base_url=base_url)
    return _instance