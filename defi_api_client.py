"""
DeFi Aggregation API Client

This module implements a client for accessing the DeFi Aggregation API,
which provides authentic on-chain data from Solana DEXes including
Raydium, Meteora, and Orca.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('defi_api_client')

class DefiApiClient:
    """Client for interacting with the DeFi Aggregation API"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the DeFi API client
        
        Args:
            api_key: API key for authentication (defaults to environment variable)
            base_url: Base URL for the API (defaults to standard endpoint)
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("DEFI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Provide it as a parameter or set DEFI_API_KEY environment variable.")
            
        # Set base URL
        self.base_url = base_url or "https://filotdefiapi.replit.app/api/v1"
        
        # Create a session for better performance
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        logger.info(f"Initialized DeFi API client with base URL: {self.base_url}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the API
        
        Args:
            endpoint: API endpoint (relative to base URL)
            params: Query parameters for the request
            
        Returns:
            API response as a dictionary
        """
        # Construct full URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            # Make the request
            response = self.session.get(url, params=params, timeout=15)
            
            # Raise for HTTP errors
            response.raise_for_status()
            
            # Parse and return JSON response
            return response.json()
            
        except requests.exceptions.RequestException as e:
            # Handle request errors
            logger.error(f"API request failed: {str(e)}")
            
            # Return error information
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                try:
                    error_data = e.response.json()
                    error_message = error_data.get('error', str(e))
                except:
                    error_message = str(e)
                
                logger.error(f"API error ({status_code}): {error_message}")
                return {"error": error_message, "status": status_code}
            
            return {"error": str(e), "status": 500}
    
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
        # Prepare parameters
        params = {}
        if source:
            params['source'] = source
        if token:
            params['token'] = token
        if sort:
            params['sort'] = sort
        if order:
            params['order'] = order
        if limit:
            params['limit'] = limit
        if page:
            params['page'] = page
        
        # Make request
        return self._make_request('pools', params)
    
    def get_pool_by_id(self, pool_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific pool
        
        Args:
            pool_id: Pool ID (base58-encoded address)
            
        Returns:
            Dictionary with pool details
        """
        return self._make_request(f'pools/{pool_id}')
    
    def get_token_information(self, token_symbol: str) -> Dict[str, Any]:
        """
        Get information about a specific token
        
        Args:
            token_symbol: Token symbol (e.g., 'SOL', 'USDC')
            
        Returns:
            Dictionary with token details
        """
        return self._make_request(f'tokens/{token_symbol}')
    
    def get_top_pools_by_apr(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get top pools ordered by APR
        
        Args:
            limit: Maximum number of results
            
        Returns:
            Dictionary with top pools data
        """
        return self.get_all_pools(sort='apr24h', order='desc', limit=limit)

def transform_pool_data(api_pool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform API pool data to the format expected by the application
    
    Args:
        api_pool: Pool data from the API
        
    Returns:
        Transformed pool data
    """
    # Extract token information
    tokens = api_pool.get('tokens', [])
    token1 = tokens[0] if len(tokens) > 0 else {"symbol": "Unknown", "address": "", "price": 0}
    token2 = tokens[1] if len(tokens) > 1 else {"symbol": "Unknown", "address": "", "price": 0}
    
    # Get volume data (could be a number or an object with timeframes)
    volume_data = api_pool.get('volumeUsd', 0)
    if isinstance(volume_data, dict):
        volume_24h = volume_data.get('24h', 0)
    else:
        volume_24h = volume_data
    
    # Calculate changes based on APR values
    apr24h = api_pool.get('apr24h', 0)
    apr7d = api_pool.get('apr7d', 0)
    apr30d = api_pool.get('apr30d', 0)
    
    # Calculate approximate changes
    apr_change_24h = 0  # Can't calculate without previous value
    apr_change_7d = ((apr24h / apr7d) - 1) * 100 if apr7d > 0 else 0
    apr_change_30d = ((apr24h / apr30d) - 1) * 100 if apr30d > 0 else 0
    
    # Create standardized pool object
    transformed_pool = {
        "id": api_pool.get('poolId', ''),
        "name": api_pool.get('name', ''),
        "dex": api_pool.get('source', ''),
        "category": determine_category(token1.get('symbol', ''), token2.get('symbol', '')),
        "token1_symbol": token1.get('symbol', ''),
        "token2_symbol": token2.get('symbol', ''),
        "token1_address": token1.get('address', ''),
        "token2_address": token2.get('address', ''),
        "token1_price": token1.get('price', 0),
        "token2_price": token2.get('price', 0),
        "liquidity": api_pool.get('tvl', 0),
        "volume_24h": volume_24h,
        "apr": apr24h,  # Use 24h APR as the default
        "fee": api_pool.get('fee', 0) * 100,  # Convert from decimal to percentage
        "version": "",  # Not provided by the API
        "apr_change_24h": apr_change_24h,
        "apr_change_7d": apr_change_7d,
        "apr_change_30d": apr_change_30d,
        "tvl_change_24h": 0,  # Not provided directly by the API
        "tvl_change_7d": 0,   # Not provided directly by the API
        "tvl_change_30d": 0,  # Not provided directly by the API
        "prediction_score": 0,  # Not available from this API
        "risk_score": 0,       # Not available from this API
        "data_source": "Real-time data from DeFi Aggregation API"
    }
    
    return transformed_pool

def determine_category(token1_symbol: str, token2_symbol: str) -> str:
    """
    Determine the category of a pool based on token symbols
    
    Args:
        token1_symbol: Symbol of the first token
        token2_symbol: Symbol of the second token
        
    Returns:
        Category name as a string
    """
    # Normalize token symbols
    token1_upper = token1_symbol.upper()
    token2_upper = token2_symbol.upper()
    
    # Define token sets for categorization
    stablecoins = {"USDC", "USDT", "DAI", "BUSD", "USDH"}
    major_tokens = {"SOL", "BTC", "ETH", "WSOL"}
    meme_tokens = {"BONK", "SAMO", "WIF", "DOGWIFHAT", "CAT", "POPCAT"}
    defi_tokens = {"RAY", "ORCA", "JUP", "MER", "SRM"}
    
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

if __name__ == "__main__":
    # Simple test code for the client
    try:
        # Initialize client with API key from environment
        client = DefiApiClient()
        
        # Get top pools
        result = client.get_top_pools_by_apr(limit=5)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            pools = result.get("pools", [])
            print(f"Got {len(pools)} top pools:")
            
            for pool in pools:
                transformed = transform_pool_data(pool)
                print(f"  {transformed['name']} ({transformed['dex']}): ${transformed['liquidity']:,.0f} liquidity, {transformed['apr']:.2f}% APR")
                
    except Exception as e:
        print(f"Error testing DeFi API client: {e}")