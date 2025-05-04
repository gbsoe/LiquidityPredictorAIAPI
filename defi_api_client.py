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
        
        self.base_url = base_url or "https://raydium-trader-filot.replit.app"
        
        # Create a session for better performance
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",  # Use Bearer token format (confirmed by testing)
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
                
                # Parse the JSON response
                result = response.json()
                
                # Rate-limit ourselves to respect the 10 req/sec API limit
                time.sleep(0.1)
                
                return result
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
        
        # This code should never be reached, but we need a return to satisfy the type checker
        # If we've exhausted all retries with no success, we'll have raised an exception
        raise RuntimeError(f"Failed to make request to {endpoint} after {max_retries} retries")
    
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
        # Create a properly typed dictionary with all parameters
        params: Dict[str, Any] = {
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
        
        return self._make_request("/api/pools", params)
    
    def get_pool_by_id(self, pool_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific pool
        
        Args:
            pool_id: Pool ID (base58-encoded address)
            
        Returns:
            Dictionary with pool details
        """
        return self._make_request(f"/api/pools/{pool_id}")
    
    def get_token_information(self, token_symbol: str) -> Dict[str, Any]:
        """
        Get information about a specific token
        
        Args:
            token_symbol: Token symbol (e.g., 'SOL', 'USDC')
            
        Returns:
            Dictionary with token details
        """
        return self._make_request(f"/api/tokens/{token_symbol}")
    
    def get_top_pools_by_apr(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get top pools ordered by APR
        
        Args:
            limit: Maximum number of results
            
        Returns:
            Dictionary with top pools data
        """
        return self.get_all_pools(sort="apr24h", order="desc", limit=limit)


def transform_pool_data(api_pool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Transform API pool data to the format expected by the application
    
    Args:
        api_pool: Pool data from the API
        
    Returns:
        Transformed pool data
    """
    # According to the API docs, poolId is the authentic base58-encoded Solana address
    # Never create synthetic IDs, always use authentic ones from the API
    pool_id = api_pool.get("poolId", "")
    if not pool_id:
        # If there's no poolId, we cannot reliably identify this pool
        logging.error(f"Pool missing ID: {api_pool}")
        # This pool cannot be used for reliable predictions
        return None
    
    # Get the name directly from the API
    pool_name = api_pool.get("name", "")
    # Remove ' LP' suffix if present (as seen in the API example: "mSOL-USDC LP")
    if pool_name.endswith(" LP"):
        pool_name = pool_name[:-3]
    
    # Extract token data from the tokens array
    tokens = api_pool.get("tokens", [])
    token1 = tokens[0] if len(tokens) > 0 else {}
    token2 = tokens[1] if len(tokens) > 1 else {}
    
    # Extract token symbols - these should always be present in valid API responses
    token1_symbol = token1.get("symbol", "")
    token2_symbol = token2.get("symbol", "")
    
    # If either token symbol is "UNKNOWN" or empty, this pool is not usable for predictions
    if not token1_symbol or not token2_symbol or token1_symbol == "UNKNOWN" or token2_symbol == "UNKNOWN":
        logging.warning(f"Pool with ID {pool_id} has unknown or missing token symbols: {token1_symbol}-{token2_symbol}")
        return None
    
    # If we don't have proper token symbols from the API, construct the name from them
    if not pool_name and token1_symbol and token2_symbol:
        pool_name = f"{token1_symbol}-{token2_symbol}"
    elif not pool_name:
        # If we still don't have a name, this is not a valid pool for prediction
        logging.error(f"Pool with ID {pool_id} has no name or token symbols")
        return None
    
    # Determine category based on token symbols
    category = determine_category(token1_symbol, token2_symbol)
    
    # Get metrics from API - this is where TVL and APY are actually stored
    metrics = api_pool.get("metrics", {})
    
    # Get TVL from metrics
    tvl = metrics.get("tvl", 0)
    
    # Get APR/APY values from metrics (API uses APY nomenclature)
    # Convert APY to APR for consistency with our app
    apy_24h = metrics.get("apy24h", 0)
    apy_7d = metrics.get("apy7d", 0)
    apy_30d = metrics.get("apy30d", 0)
    
    # Simple conversion for display purposes
    apr_24h = apy_24h
    apr_7d = apy_7d
    apr_30d = apy_30d
    
    # Calculate APR changes
    apr_change_24h = 0
    if apr_24h != 0 and apr_7d != 0:
        # Estimate 24h change as a fraction of the 7d change
        apr_change_24h = (apr_24h - apr_7d) / 7
    
    apr_change_7d = 0
    if apr_7d != 0 and apr_30d != 0:
        # Use difference between 7d and 30d APR for weekly change
        apr_change_7d = apr_7d - apr_30d
    
    # Get volume from API metrics
    volume_24h = metrics.get("volumeUsd", 0)
    
    # Get fee percentage from metrics
    fee_rate = metrics.get("fee", 0) * 100  # Convert to percentage
    
    # Get additional token info
    token1_price = token1.get("price", 0)
    token2_price = token2.get("price", 0)
    token1_address = token1.get("address", "")
    token2_address = token2.get("address", "")
    
    # Set token reserves when available
    token1_reserve = 0
    token2_reserve = 0
    reserves = api_pool.get("reserves", {})
    if reserves and token1_symbol in reserves:
        token1_reserve = reserves[token1_symbol]
    if reserves and token2_symbol in reserves:
        token2_reserve = reserves[token2_symbol]
    
    # Create a unified pool data structure compatible with our application
    return {
        "id": pool_id,  # Use authentic base58-encoded ID directly from API
        "name": pool_name,  # Use name from API with LP suffix removed
        "dex": api_pool.get("source", "Unknown"),
        "liquidity": tvl,
        "volume_24h": volume_24h,
        "apr": apr_24h,
        "apr_change_24h": apr_change_24h,
        "apr_change_7d": apr_change_7d,
        "fee_rate": fee_rate,
        "token1_symbol": token1_symbol,
        "token2_symbol": token2_symbol,
        "token1_reserve": token1_reserve,
        "token2_reserve": token2_reserve,
        "token1_price": token1_price,
        "token2_price": token2_price,
        "token1_decimals": token1.get("decimals", 9),
        "token2_decimals": token2.get("decimals", 9),
        "token1_address": token1_address,
        "token2_address": token2_address,
        "category": category,
        "data_source": "Real-time DeFi API",
        "prediction_score": calculate_prediction_score(api_pool),
        "last_updated": api_pool.get("updatedAt", ""),
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
    
    # Extract metrics which contain the APY/TVL/volume data
    metrics = pool_data.get("metrics", {})
    
    # Add points for higher APR/APY (up to 15 points)
    apy = metrics.get("apy24h", 0)  # Use the correct field name from metrics
    if apy > 100:
        score += 15
    elif apy > 50:
        score += 10
    elif apy > 20:
        score += 5
    
    # Add points for higher TVL (up to 10 points)
    tvl = metrics.get("tvl", 0)
    if tvl > 1000000:  # $1M+
        score += 10
    elif tvl > 500000:  # $500K+
        score += 7
    elif tvl > 100000:  # $100K+
        score += 5
    
    # Add points for higher volume (up to 10 points)
    volume = metrics.get("volumeUsd", 0)
    if volume > 500000:  # $500K+
        score += 10
    elif volume > 100000:  # $100K+
        score += 7
    elif volume > 50000:  # $50K+
        score += 5
    
    # Add points for APR stability or growth (up to 10 points)
    # Compare apy24h and apy7d from metrics
    apy_24h = metrics.get("apy24h", 0)
    apy_7d = metrics.get("apy7d", 0)
    
    if apy_24h and apy_7d:  # Only compare if both values exist
        if apy_24h > apy_7d * 1.05:  # 5% increase
            score += 10  # Strong uptrend
        elif apy_24h > apy_7d:
            score += 7   # Moderate uptrend
        elif apy_24h > apy_7d * 0.95:
            score += 5   # Stable
    
    # Add points for having a popular token (up to 5 points)
    tokens = pool_data.get("tokens", [])
    popular_tokens = ["SOL", "USDC", "ETH", "BTC", "RAY", "BONK", "mSOL"]
    for token in tokens:
        if token.get("symbol") in popular_tokens:
            score += 2.5
            break
    
    # Add points for DEX reputation (up to 5 points)
    dex = pool_data.get("source", "").lower()
    if dex == "raydium":
        score += 5
    elif dex == "orca":
        score += 4
    elif dex == "meteora":
        score += 3
    
    # Add points for fee structure (lower fees are better for traders, up to 5 points)
    fee = pool_data.get("fee", 0)
    if fee > 0:  # Only if fee information is available
        if fee <= 0.001:  # 0.1% or less
            score += 5
        elif fee <= 0.002:  # 0.2% or less
            score += 3
        elif fee <= 0.003:  # 0.3% or less
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