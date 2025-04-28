"""
DeFi Pool Fetcher

This module fetches liquidity pool data from the DeFi Aggregation API,
providing authentic on-chain data from Solana DEXes.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import our DeFi API client
from defi_api_client import DefiApiClient, transform_pool_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('defi_pool_fetcher')

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
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("DEFI_API_KEY")
        
        # Initialize the API client
        self.api_client = DefiApiClient(api_key=self.api_key)
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
        try:
            logger.info(f"Fetching {limit} pools from DeFi Aggregation API" + 
                      (f" for {dex}" if dex else ""))
            
            # Make API request
            if dex:
                response = self.api_client.get_all_pools(source=dex, limit=limit)
            else:
                response = self.api_client.get_all_pools(limit=limit)
            
            # Check for errors
            if "error" in response:
                logger.error(f"API error: {response['error']}")
                return []
            
            # Extract pools data
            pools = response.get("pools", [])
            logger.info(f"Retrieved {len(pools)} pools from API")
            
            # Transform pools to application format
            transformed_pools = []
            for pool in pools:
                try:
                    transformed_pool = transform_pool_data(pool)
                    transformed_pools.append(transformed_pool)
                except Exception as e:
                    logger.error(f"Error transforming pool {pool.get('poolId', 'unknown')}: {e}")
            
            logger.info(f"Successfully transformed {len(transformed_pools)} pools")
            
            # Save to file for inspection
            with open("fetched_defi_pools.json", "w") as f:
                json.dump(transformed_pools, f, indent=2)
                logger.info(f"Saved pool data to fetched_defi_pools.json")
            
            return transformed_pools
            
        except Exception as e:
            logger.error(f"Error fetching pools from DeFi Aggregation API: {e}")
            return []
    
    def fetch_pool_by_id(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch details for a specific pool
        
        Args:
            pool_id: Pool ID (base58-encoded address)
            
        Returns:
            Pool data dictionary or None if not found
        """
        try:
            logger.info(f"Fetching details for pool {pool_id}")
            
            # Make API request
            response = self.api_client.get_pool_by_id(pool_id)
            
            # Check for errors
            if "error" in response:
                logger.error(f"API error: {response['error']}")
                return None
            
            # Extract pool data
            pool = response.get("pool")
            if not pool:
                logger.error(f"No pool data found for {pool_id}")
                return None
            
            # Transform to application format
            transformed_pool = transform_pool_data(pool)
            logger.info(f"Successfully retrieved pool {pool_id} ({transformed_pool['name']})")
            
            return transformed_pool
            
        except Exception as e:
            logger.error(f"Error fetching pool {pool_id}: {e}")
            return None
    
    def fetch_top_pools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch top-performing pools ordered by APR
        
        Args:
            limit: Maximum number of pools to return
            
        Returns:
            List of pool data dictionaries
        """
        try:
            logger.info(f"Fetching top {limit} pools by APR")
            
            # Make API request
            response = self.api_client.get_top_pools_by_apr(limit=limit)
            
            # Check for errors
            if "error" in response:
                logger.error(f"API error: {response['error']}")
                return []
            
            # Extract pools data
            pools = response.get("pools", [])
            logger.info(f"Retrieved {len(pools)} top pools from API")
            
            # Transform pools to application format
            transformed_pools = []
            for pool in pools:
                try:
                    transformed_pool = transform_pool_data(pool)
                    transformed_pools.append(transformed_pool)
                except Exception as e:
                    logger.error(f"Error transforming pool {pool.get('poolId', 'unknown')}: {e}")
            
            logger.info(f"Successfully transformed {len(transformed_pools)} top pools")
            return transformed_pools
            
        except Exception as e:
            logger.error(f"Error fetching top pools: {e}")
            return []

def main():
    """Test the fetcher with current environment"""
    try:
        # Create a fetcher
        fetcher = DefiPoolFetcher()
        
        # Fetch top pools
        pools = fetcher.fetch_top_pools(limit=5)
        
        print(f"Fetched {len(pools)} top pools:")
        for pool in pools:
            print(f"  {pool['name']} ({pool['dex']}): ${pool['liquidity']:,.0f} liquidity, {pool['apr']:.2f}% APR")
            
    except Exception as e:
        print(f"Error testing DeFi pool fetcher: {e}")

if __name__ == "__main__":
    main()