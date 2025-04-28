"""
DeFi Aggregation API Client

This module implements a client for accessing the DeFi Aggregation API,
which provides authentic on-chain data from Solana DEXes including
Raydium, Meteora, and Orca.
"""

import os
import time
import requests
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DefiApiClient:
    """Client for interacting with the DeFi Aggregation API"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the DeFi API client
        
        Args:
            api_key: API key for authentication (defaults to environment variable)
            base_url: Base URL for the API (defaults to standard endpoint)
        """
        self.api_key = api_key or os.getenv("DEFI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set the DEFI_API_KEY environment variable or pass it explicitly.")
        
        self.base_url = base_url or "https://defi-aggregation-api.solana.com/v1"
        
        # Create a session for better performance
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     max_retries: int = 3) -> Dict[str, Any]:
        """
        Make a request to the API with retry logic
        
        Args:
            endpoint: API endpoint (relative to base URL)
            params: Query parameters for the request
            max_retries: Maximum number of retries for failed requests
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.base_url}{endpoint}"
        retries = 0
        
        while retries < max_retries:
            try:
                response = self.session.get(url, params=params, timeout=15)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if (
                    hasattr(e, 'response') and 
                    e.response is not None and
                    (e.response.status_code == 429 or e.response.status_code >= 500) and
                    retries < max_retries - 1
                ):
                    # Retry on rate limit (429) or server errors (5xx)
                    retries += 1
                    logger.warning(f"Retry attempt {retries} for {endpoint}")
                    
                    # Exponential backoff with jitter
                    delay = (2 ** retries) + (0.1 * retries)
                    time.sleep(delay)
                else:
                    # Log the error details
                    if hasattr(e, 'response') and e.response is not None:
                        logger.error(f"API request failed: {e.response.status_code} - {e.response.text}")
                    else:
                        logger.error(f"API request failed: {str(e)}")
                    
                    # Don't retry for other errors
                    raise
    
    def get_all_pools(self, 
                     source: Optional[str] = None, 
                     token: Optional[str] = None,
                     sort: Optional[str] = None,
                     order: Optional[str] = "desc",
                     limit: Optional[int] = 50,
                     page: Optional[int] = 1) -> Dict[str, Any]:
        """
        Get all available liquidity pools
        
        Args:
            source: Filter by DEX (Raydium, Meteora, or Orca)
            token: Filter by token symbol
            sort: Sort field (e.g., 'apr24h', 'tvl')
            order: Sort direction ('asc' or 'desc')
            limit: Maximum number of results
            page: Pagination page number
            
        Returns:
            Dictionary with pools data
        """
        params = {
            "limit": limit,
            "page": page
        }
        
        if source:
            params["source"] = source
        if token:
            params["token"] = token
        if sort:
            params["sort"] = sort
        if order:
            params["order"] = order
        
        return self._make_request("/pools", params)
    
    def get_pool_by_id(self, pool_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific pool
        
        Args:
            pool_id: Pool ID (base58-encoded address)
            
        Returns:
            Dictionary with pool details
        """
        return self._make_request(f"/pools/{pool_id}")
    
    def get_token_information(self, token_symbol: str) -> Dict[str, Any]:
        """
        Get information about a specific token
        
        Args:
            token_symbol: Token symbol (e.g., 'SOL', 'USDC')
            
        Returns:
            Dictionary with token details
        """
        return self._make_request(f"/tokens/{token_symbol}")
    
    def get_top_pools_by_apr(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get top pools ordered by APR
        
        Args:
            limit: Maximum number of results
            
        Returns:
            Dictionary with top pools data
        """
        return self.get_all_pools(sort="apr24h", order="desc", limit=limit)


def transform_pool_data(api_pool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform API pool data to the format expected by the application
    
    Args:
        api_pool: Pool data from the API
        
    Returns:
        Transformed pool data
    """
    # Extract token symbols
    token1_symbol = api_pool.get("token1", {}).get("symbol", "Unknown")
    token2_symbol = api_pool.get("token2", {}).get("symbol", "Unknown")
    
    # Determine category based on token symbols
    category = determine_category(token1_symbol, token2_symbol)
    
    # Create a unified pool data structure compatible with our application
    return {
        "id": api_pool.get("address", ""),
        "name": f"{token1_symbol}-{token2_symbol}",
        "dex": api_pool.get("source", "Unknown"),
        "liquidity": api_pool.get("tvl", 0),
        "volume_24h": api_pool.get("volume24h", 0),
        "apr": api_pool.get("apr24h", 0),
        "apr_change_24h": api_pool.get("aprChange24h", 0),
        "apr_change_7d": api_pool.get("aprChange7d", 0),
        "tvl_change_24h": api_pool.get("tvlChange24h", 0),
        "fee_rate": api_pool.get("fee", 0) * 100 if "fee" in api_pool else 0,  # Convert to percentage
        "token1_symbol": token1_symbol,
        "token2_symbol": token2_symbol,
        "token1_reserve": api_pool.get("token1Reserve", 0),
        "token2_reserve": api_pool.get("token2Reserve", 0),
        "token1_price": api_pool.get("token1", {}).get("price", 0),
        "token2_price": api_pool.get("token2", {}).get("price", 0),
        "token1_decimals": api_pool.get("token1", {}).get("decimals", 9),
        "token2_decimals": api_pool.get("token2", {}).get("decimals", 9),
        "token1_address": api_pool.get("token1", {}).get("address", ""),
        "token2_address": api_pool.get("token2", {}).get("address", ""),
        "category": category,
        "data_source": "Real-time DeFi API",
        "prediction_score": calculate_prediction_score(api_pool),
        "last_updated": api_pool.get("lastUpdated", ""),
    }


def determine_category(token1_symbol: str, token2_symbol: str) -> str:
    """
    Determine the category of a pool based on token symbols
    
    Args:
        token1_symbol: Symbol of the first token
        token2_symbol: Symbol of the second token
        
    Returns:
        Category name as a string
    """
    stablecoins = ["USDC", "USDT", "DAI", "BUSD", "USDH", "USDR", "UXD"]
    
    # Both tokens are stablecoins
    if token1_symbol in stablecoins and token2_symbol in stablecoins:
        return "Stablecoin"
    
    # One token is a stablecoin
    if token1_symbol in stablecoins or token2_symbol in stablecoins:
        return "Stable-based"
    
    # Either token is SOL
    if token1_symbol == "SOL" or token2_symbol == "SOL":
        return "SOL-based"
    
    # Default category for other token pairs
    return "Other"


def calculate_prediction_score(pool_data: Dict[str, Any]) -> float:
    """
    Calculate a prediction score for a pool based on various metrics
    
    Args:
        pool_data: Pool data from the API
        
    Returns:
        Prediction score between 0 and 100
    """
    # Base score starts at 50
    score = 50.0
    
    # Add points for higher APR (up to 15 points)
    apr = pool_data.get("apr24h", 0)
    if apr > 100:
        score += 15
    elif apr > 50:
        score += 10
    elif apr > 20:
        score += 5
    
    # Add points for higher TVL (up to 10 points)
    tvl = pool_data.get("tvl", 0)
    if tvl > 1000000:  # $1M+
        score += 10
    elif tvl > 500000:  # $500K+
        score += 7
    elif tvl > 100000:  # $100K+
        score += 5
    
    # Add points for higher volume (up to 10 points)
    volume = pool_data.get("volume24h", 0)
    if volume > 500000:  # $500K+
        score += 10
    elif volume > 100000:  # $100K+
        score += 7
    elif volume > 50000:  # $50K+
        score += 5
    
    # Add points for positive APR change (up to 10 points)
    apr_change_24h = pool_data.get("aprChange24h", 0)
    if apr_change_24h > 5:
        score += 10
    elif apr_change_24h > 2:
        score += 7
    elif apr_change_24h > 0:
        score += 5
    
    # Add points for positive TVL change (up to 5 points)
    tvl_change_24h = pool_data.get("tvlChange24h", 0)
    if tvl_change_24h > 5:
        score += 5
    elif tvl_change_24h > 2:
        score += 3
    elif tvl_change_24h > 0:
        score += 1
    
    # Cap the score at 100
    return min(score, 100.0)


if __name__ == "__main__":
    # Example usage
    try:
        client = DefiApiClient()
        pools = client.get_all_pools(limit=5)
        print(f"Retrieved {len(pools.get('pools', []))} pools")
    except Exception as e:
        print(f"Error: {str(e)}")