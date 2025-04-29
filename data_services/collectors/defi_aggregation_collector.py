"""
DeFi Aggregation API Collector for SolPool Insight.

This collector retrieves data from the DeFi Aggregation API.
"""

import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple

from .base_collector import BaseCollector
from ..config import API_SETTINGS
from defi_aggregation_api import DefiAggregationAPI

# Configure logging
logger = logging.getLogger(__name__)

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
        # Initialize base class
        super().__init__(
            collector_id="defi_aggregation_api",
            collector_name="DeFi Aggregation API",
            base_url=base_url or API_SETTINGS["DEFI_API_URL"],
            request_delay=API_SETTINGS["REQUEST_DELAY"]
        )
        
        # API key from args or config
        self.api_key = api_key or API_SETTINGS["DEFI_API_KEY"]
        
        # Initialize API client
        try:
            self.api_client = DefiAggregationAPI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"Initialized DeFi Aggregation API client with base URL: {self.base_url}")
        except Exception as e:
            logger.error(f"Error initializing DeFi Aggregation API client: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _collect_data(self) -> List[Dict[str, Any]]:
        """
        Collect pool data from the DeFi Aggregation API.
        
        Returns:
            List of collected pool data
        """
        # Validate API client
        if not hasattr(self, 'api_client') or self.api_client is None:
            raise ValueError("API client not initialized")
            
        # List to store all collected pools
        all_pools = []
        
        # Track collection start time
        start_time = time.time()
        
        # Get available DEXes
        logger.info("Retrieving supported DEXes from API")
        try:
            dexes = self.api_client.get_supported_dexes()
            logger.info(f"Found {len(dexes)} supported DEXes: {', '.join(dexes)}")
        except Exception as e:
            logger.warning(f"Error retrieving DEXes, using default list: {str(e)}")
            # Fallback to known DEXes
            dexes = ["Raydium", "Orca", "Meteora"]
        
        # Track maximum pools per DEX
        max_pools_per_dex = 30  # Adjust based on API limitations
        
        # Track API request stats
        total_requests = 0
        total_pools = 0
        
        # Process each DEX
        for dex in dexes:
            try:
                logger.info(f"Collecting pools for DEX: {dex}")
                
                # Get pools for this DEX
                request_start = time.time()
                pools = self.api_client.get_pools_by_dex(dex=dex, limit=max_pools_per_dex)
                request_duration = time.time() - request_start
                
                # Track request stats
                self._track_request_stats(success=True, duration=request_duration)
                total_requests += 1
                
                # Process pools
                if pools:
                    logger.info(f"Retrieved {len(pools)} pools from {dex}")
                    
                    # Normalize each pool
                    for pool in pools:
                        # Add DEX info if not present
                        if not pool.get("dex"):
                            pool["dex"] = dex
                            
                        # Add data source info
                        pool["data_source"] = f"Real-time data from {dex} via DeFi API"
                        
                        # Normalize and add to results
                        normalized_pool = self._normalize_pool_data(pool)
                        all_pools.append(normalized_pool)
                        
                    total_pools += len(pools)
                else:
                    logger.warning(f"No pools retrieved for DEX: {dex}")
            except Exception as e:
                logger.error(f"Error retrieving pools for DEX {dex}: {str(e)}")
                logger.error(traceback.format_exc())
                self._track_request_stats(success=False, duration=0)
                total_requests += 1
        
        # Try to get pools by token if we have fewer than expected
        if total_pools < 30:
            try:
                # Common tokens to check
                common_tokens = ["SOL", "USDC", "ETH", "BTC", "mSOL", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"]
                
                for token in common_tokens[:3]:  # Limit to 3 to avoid too many requests
                    logger.info(f"Collecting pools for token: {token}")
                    
                    # Get pools for this token
                    request_start = time.time()
                    token_pools = self.api_client.get_pools_by_token(token=token, limit=10)
                    request_duration = time.time() - request_start
                    
                    # Track request stats
                    self._track_request_stats(success=True, duration=request_duration)
                    total_requests += 1
                    
                    # Process pools
                    if token_pools:
                        logger.info(f"Retrieved {len(token_pools)} pools for token {token}")
                        
                        # Keep track of pools we've already added
                        existing_pool_ids = {p.get("pool_id") for p in all_pools}
                        
                        # Add new pools
                        for pool in token_pools:
                            pool_id = pool.get("id") or pool.get("pool_id") or pool.get("address") or ""
                            
                            # Skip if already added
                            if pool_id in existing_pool_ids:
                                continue
                                
                            # Add data source info
                            pool["data_source"] = f"Real-time data via DeFi API (token: {token})"
                            
                            # Normalize and add to results
                            normalized_pool = self._normalize_pool_data(pool)
                            all_pools.append(normalized_pool)
                            existing_pool_ids.add(pool_id)
                            
                        total_pools = len(all_pools)
                    else:
                        logger.warning(f"No pools retrieved for token: {token}")
            except Exception as e:
                logger.error(f"Error retrieving pools by token: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Calculate collection duration
        duration = time.time() - start_time
        
        # Log collection stats
        logger.info(
            f"DeFi API collection complete: {total_pools} pools from {len(dexes)} DEXes " +
            f"in {duration:.2f}s ({total_requests} API requests)"
        )
        
        return all_pools

# Singleton instance
_collector_instance = None

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
    global _collector_instance
    
    if _collector_instance is None:
        _collector_instance = DefiAggregationCollector(
            api_key=api_key,
            base_url=base_url
        )
        
    return _collector_instance