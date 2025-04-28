"""
DeFi Pool Fetcher

This module fetches liquidity pool data from the DeFi Aggregation API,
providing authentic on-chain data from Solana DEXes.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from defi_api_client import DefiApiClient, transform_pool_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DefiPoolFetcher:
    """
    Fetches real liquidity pool data from the DeFi Aggregation API.
    Provides authentic on-chain data from multiple Solana DEXes.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the fetcher
        
        Args:
            api_key: Optional API key to use (defaults to environment variable)
        """
        # Initialize the API client
        self.api_key = api_key or os.getenv("DEFI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set the DEFI_API_KEY environment variable or pass it explicitly.")
        
        self.client = DefiApiClient(api_key=self.api_key)
        logger.info("Initialized DeFi Pool Fetcher")
    
    def fetch_pools(self, limit: int = 50, dex: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch liquidity pools from the DeFi Aggregation API
        
        Args:
            limit: Maximum number of pools to return
            dex: Optional DEX name to filter by (Raydium, Meteora, Orca)
            
        Returns:
            List of pool data dictionaries in the application's expected format
        """
        logger.info(f"Fetching up to {limit} pools from DeFi API" + (f" for {dex}" if dex else ""))
        
        try:
            # Get pools from API
            response = self.client.get_all_pools(
                source=dex,
                limit=limit,
                sort="apr24h",  # Sort by APR for potentially more interesting pools
                order="desc"     # Highest APR first
            )
            
            # Extract pools list from response
            if "pools" not in response or not response["pools"]:
                logger.warning("No pools returned from API")
                return []
            
            api_pools = response["pools"]
            logger.info(f"Retrieved {len(api_pools)} pools from DeFi API")
            
            # Transform API data to our application format
            transformed_pools = [transform_pool_data(pool) for pool in api_pools]
            
            # Add data source to all pools
            for pool in transformed_pools:
                pool["data_source"] = "Real-time DeFi API"
            
            return transformed_pools
            
        except Exception as e:
            logger.error(f"Error fetching pools from DeFi API: {str(e)}")
            raise
    
    def fetch_pool_by_id(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch details for a specific pool
        
        Args:
            pool_id: Pool ID (base58-encoded address)
            
        Returns:
            Pool data dictionary or None if not found
        """
        logger.info(f"Fetching pool with ID {pool_id}")
        
        try:
            # Get pool from API
            pool_data = self.client.get_pool_by_id(pool_id)
            
            # Check if pool data is valid
            if not pool_data or "error" in pool_data:
                logger.warning(f"Pool {pool_id} not found or error in response")
                return None
            
            # Transform API data to our application format
            transformed_pool = transform_pool_data(pool_data)
            transformed_pool["data_source"] = "Real-time DeFi API"
            
            return transformed_pool
            
        except Exception as e:
            logger.error(f"Error fetching pool {pool_id}: {str(e)}")
            return None
    
    def fetch_top_pools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch top-performing pools ordered by APR
        
        Args:
            limit: Maximum number of pools to return
            
        Returns:
            List of pool data dictionaries
        """
        logger.info(f"Fetching top {limit} pools by APR")
        
        try:
            # Get top pools by APR
            response = self.client.get_top_pools_by_apr(limit=limit)
            
            # Extract pools list from response
            if "pools" not in response or not response["pools"]:
                logger.warning("No top pools returned from API")
                return []
            
            api_pools = response["pools"]
            logger.info(f"Retrieved {len(api_pools)} top pools from DeFi API")
            
            # Transform API data to our application format
            transformed_pools = [transform_pool_data(pool) for pool in api_pools]
            
            # Add data source to all pools
            for pool in transformed_pools:
                pool["data_source"] = "Real-time DeFi API (Top Performers)"
            
            return transformed_pools
            
        except Exception as e:
            logger.error(f"Error fetching top pools: {str(e)}")
            return []


def main():
    """Test the fetcher with current environment"""
    try:
        fetcher = DefiPoolFetcher()
        pools = fetcher.fetch_pools(limit=5)
        
        print(f"Successfully fetched {len(pools)} pools:")
        for i, pool in enumerate(pools):
            print(f"  {i+1}. {pool['name']} (APR: {pool['apr']:.2f}%, TVL: ${pool['liquidity']:,.2f})")
        
        # Save to file for inspection
        with open("fetched_pools.json", "w") as f:
            json.dump(pools, f, indent=2)
            print(f"Saved pool data to fetched_pools.json")
    
    except Exception as e:
        print(f"Error testing fetcher: {str(e)}")


if __name__ == "__main__":
    main()