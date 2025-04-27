import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pool_retrieval.log')
    ]
)
logger = logging.getLogger('raydium_pool_fetcher')

# Import our Raydium API client
from data_ingestion.raydium_api_client import RaydiumAPIClient

class RaydiumPoolFetcher:
    """
    Fetches real liquidity pool data from Raydium's API.
    This class provides genuine on-chain data for liquidity pools.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize the RaydiumPoolFetcher.
        
        Args:
            api_key: Optional Raydium API key to use.
            api_url: Optional Raydium API URL to use.
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("RAYDIUM_API_KEY")
            if api_key == "YOUR_API_KEY_HERE":
                logger.warning("Using default Raydium API key - this may not work correctly")
        
        # Get API URL from environment if not provided
        if api_url is None:
            api_url = os.getenv("RAYDIUM_API_URL", "https://api.raydium.io")
        
        # Initialize the API client
        self.api_client = RaydiumAPIClient(api_key=api_key, base_url=api_url)
        logger.info(f"Initialized RaydiumPoolFetcher with API URL: {api_url}")
        
    def fetch_pools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch liquidity pools from Raydium's API.
        
        Args:
            limit: Maximum number of pools to return.
            
        Returns:
            List of pool data dictionaries with real metrics.
        """
        try:
            logger.info(f"Fetching {limit} pools from Raydium API...")
            
            # For Raydium v2 API, use a direct approach to get the liquidity pools
            try:
                # Direct URL for Raydium v2 API SDK endpoint
                url = f"{self.api_client.base_url}/sdk/liquidity/mainnet.json"
                
                # Make request
                response = self.api_client.session.get(url, timeout=15)
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                # Extract official pools from the response
                pools_data = data.get("official", [])
                logger.info(f"Retrieved {len(pools_data)} official pools from Raydium v2 API")
                
                # Limit the number of pools
                pools_data = pools_data[:limit]
            except Exception as e:
                logger.error(f"Error fetching from Raydium v2 API: {str(e)}")
                # Fallback to the original method
                pools_data = self.api_client.get_all_pools()
                pools_data = pools_data[:limit]
            
            # Transform the data to match our application's expected format
            transformed_pools = []
            
            for pool in pools_data:
                try:
                    # Extract essential information
                    pool_id = pool.get("id", "unknown")
                    
                    # Get detailed information for this pool
                    pool_details = self.api_client.get_pool_by_id(pool_id)
                    
                    # Get metrics for the pool
                    pool_metrics = self.api_client.get_pool_metrics(pool_id)
                    
                    # Extract tokens information
                    token1_symbol = pool.get("token1", {}).get("symbol", "Unknown")
                    token2_symbol = pool.get("token2", {}).get("symbol", "Unknown")
                    token1_address = pool.get("token1", {}).get("address", "")
                    token2_address = pool.get("token2", {}).get("address", "")
                    
                    # Extract financial metrics
                    liquidity = pool.get("liquidity", 0.0)
                    volume_24h = pool.get("volume24h", 0.0)
                    apr = pool.get("apr", 0.0)
                    fee = pool.get("fee", 0.003)  # Default to 0.3% if not specified
                    
                    # Create a standardized pool object
                    transformed_pool = {
                        "id": pool_id,
                        "dex": "Raydium",
                        "name": f"{token1_symbol}/{token2_symbol}",
                        "token1_symbol": token1_symbol,
                        "token2_symbol": token2_symbol,
                        "token1_address": token1_address,
                        "token2_address": token2_address,
                        "liquidity": liquidity,
                        "volume_24h": volume_24h,
                        "apr": apr,
                        "fee": fee,
                        "version": pool.get("version", "v4"),
                        "category": self._determine_category(token1_symbol, token2_symbol),
                        "data_source": "Real-time data from Raydium API"
                    }
                    
                    transformed_pools.append(transformed_pool)
                    logger.info(f"Added pool: {token1_symbol}/{token2_symbol} with APR: {apr:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Error processing pool {pool.get('id', 'unknown')}: {str(e)}")
            
            logger.info(f"Successfully fetched and transformed {len(transformed_pools)} pools")
            
            # Save to file for inspection
            with open("fetched_pools.json", "w") as f:
                json.dump(transformed_pools, f, indent=2)
                logger.info(f"Saved pool data to fetched_pools.json")
                
            return transformed_pools
            
        except Exception as e:
            logger.error(f"Error fetching pools from Raydium API: {str(e)}")
            return []
    
    def _determine_category(self, token1_symbol: str, token2_symbol: str) -> str:
        """
        Determine the category of a pool based on token symbols.
        
        Args:
            token1_symbol: Symbol of the first token
            token2_symbol: Symbol of the second token
            
        Returns:
            Category name as a string
        """
        # Define token sets for categorization
        stablecoins = {"USDC", "USDT", "DAI", "BUSD", "USDH"}
        major_tokens = {"SOL", "BTC", "ETH", "WSOL"}
        meme_tokens = {"BONK", "SAMO", "WIF", "DOGWIFHAT", "CAT", "POPCAT"}
        defi_tokens = {"RAY", "ORCA", "JUP", "MER", "SRM"}
        
        # Check token combinations for categorization
        token1_upper = token1_symbol.upper()
        token2_upper = token2_symbol.upper()
        
        # Stablecoin pairs
        if token1_upper in stablecoins and token2_upper in stablecoins:
            return "Stablecoin"
            
        # Major token pairs
        if token1_upper in major_tokens and token2_upper in major_tokens:
            return "Major"
            
        # Major/Stable pairs
        if (token1_upper in major_tokens and token2_upper in stablecoins) or \
           (token2_upper in major_tokens and token1_upper in stablecoins):
            return "Major"
            
        # Meme token pairs
        if token1_upper in meme_tokens or token2_upper in meme_tokens:
            return "Meme"
            
        # DeFi token pairs
        if token1_upper in defi_tokens or token2_upper in defi_tokens:
            return "DeFi"
            
        # Default category
        return "Other"

def main():
    """Test the fetcher with the current environment"""
    try:
        # Create a fetcher using environment variables or defaults
        fetcher = RaydiumPoolFetcher()
        
        # Fetch pools
        pools = fetcher.fetch_pools(limit=10)
        
        print(f"Fetched {len(pools)} pools:")
        for pool in pools:
            print(f"  {pool['name']} (Raydium): {pool['liquidity']:,.0f} USD, APR: {pool['apr']:.2f}%")
            
    except Exception as e:
        print(f"Error testing Raydium pool fetcher: {e}")

if __name__ == "__main__":
    main()