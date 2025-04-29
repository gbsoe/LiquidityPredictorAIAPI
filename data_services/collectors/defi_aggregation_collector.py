"""
DeFi Aggregation API Collector for SolPool Insight.

This collector retrieves data from the DeFi Aggregation API.
"""

import logging
import os
import time
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
        self.max_pools_per_collection = 50
        self.delay_between_requests = 0.1  # 100ms delay for rate limiting
        
        # Cache DEX list
        self.supported_dexes = []
        
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
    
    def get_pools_by_token(self, token: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get pools containing a specific token.
        
        Args:
            token: Token symbol or address
            limit: Maximum number of pools to retrieve
            
        Returns:
            List of pool data
        """
        try:
            # Use the API client to get pools by token
            pools = self.api_client.get_pools_by_token(token=token, limit=limit)
            logger.info(f"Retrieved {len(pools)} pools for token {token}")
            return pools
        except Exception as e:
            logger.error(f"Error getting pools for token {token}: {str(e)}")
            return []
    
    def _collect_data(self) -> List[Dict[str, Any]]:
        """
        Collect pool data from the DeFi Aggregation API.
        
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
        
        # Try to get supported DEXes
        try:
            dexes = self.get_supported_dexes()
            logger.info(f"Will collect pools from {len(dexes)} DEXes: {', '.join(dexes)}")
        except Exception as e:
            # Just log and continue with all pools
            logger.warning(f"Could not get supported DEXes: {str(e)}")
            dexes = []
        
        # Collect pools using the optimal API collection strategy
        all_pools = []
        
        # If we have a list of DEXes, collect pools for each DEX
        if dexes:
            for dex in dexes:
                try:
                    # Add a small delay between requests
                    time.sleep(self.delay_between_requests)
                    
                    # Get pools for this DEX
                    dex_pools = self.api_client.get_pools_by_dex(
                        dex=dex, 
                        limit=min(20, self.max_pools_per_collection // len(dexes))
                    )
                    
                    if dex_pools:
                        logger.info(f"Retrieved {len(dex_pools)} pools for DEX {dex}")
                        all_pools.extend(dex_pools)
                    else:
                        logger.warning(f"No pools found for DEX {dex}")
                except Exception as e:
                    logger.error(f"Error collecting pools for DEX {dex}: {str(e)}")
        
        # If we didn't get any pools from DEX-specific requests, try the general endpoint
        if not all_pools:
            try:
                logger.info("Falling back to general pools endpoint")
                all_pools = self.api_client.get_all_pools(max_pools=self.max_pools_per_collection)
                logger.info(f"Retrieved {len(all_pools)} pools from general endpoint")
            except Exception as e:
                logger.error(f"Error collecting pools from general endpoint: {str(e)}")
                raise  # Re-raise the exception
        
        # Log collection results
        elapsed_time = time.time() - start_time
        logger.info(
            f"Pool data collection completed: {len(all_pools)} pools in {elapsed_time:.2f}s "
            f"({len(all_pools) / elapsed_time:.2f} pools/s)"
        )
        
        return all_pools

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