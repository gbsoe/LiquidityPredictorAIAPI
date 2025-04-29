"""
Token Data Service for SolPool Insight

This module provides a service for retrieving and managing token data
using the DeFi API's token endpoints as specified in GET Token Docs.docx.

Key features:
- Fetch token data from the /tokens endpoint
- Get specific token details by symbol
- Cache token data for better performance
- DEX categorization for tokens
- Token metadata handling
"""

import os
import json
import logging
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('token_data_service')

class TokenDataService:
    """
    Service for handling token data operations using the DeFi API.
    Implements the GET Token endpoint functionality as described in the API docs.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the token data service.
        
        Args:
            api_key: API key for authentication (defaults to DEFI_API_KEY env var)
            base_url: Base URL for the API (defaults to standard URL)
        """
        self.api_key = api_key or os.getenv("DEFI_API_KEY")
        self.base_url = base_url or "https://filotdefiapi.replit.app/api/v1"
        self.token_cache = {}
        self.token_cache_timestamp = None
        self.token_cache_file = "token_cache.json"
        self.dex_categories = {}
        
        # Load token cache from file if available
        self._load_token_cache()
    
    def _load_token_cache(self) -> None:
        """Load token cache from file if it exists"""
        try:
            if os.path.exists(self.token_cache_file):
                with open(self.token_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.token_cache = cache_data.get('tokens', {})
                    self.token_cache_timestamp = cache_data.get('timestamp')
                    logger.info(f"Loaded token cache with {len(self.token_cache)} tokens from {self.token_cache_timestamp}")
        except Exception as e:
            logger.warning(f"Failed to load token cache: {str(e)}")
    
    def _save_token_cache(self) -> None:
        """Save token cache to file"""
        try:
            with open(self.token_cache_file, 'w') as f:
                json.dump({
                    'tokens': self.token_cache,
                    'timestamp': self.token_cache_timestamp
                }, f, indent=2)
            logger.info(f"Saved token cache with {len(self.token_cache)} tokens")
        except Exception as e:
            logger.warning(f"Failed to save token cache: {str(e)}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Make a rate-limited request to the API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            API response data (can be Dict or List depending on endpoint)
        
        Raises:
            ValueError: For various API errors with specific messages
        """
        if not self.api_key:
            logger.warning("No API key provided for token data service")
        
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
        try:
            logger.info(f"Making API request to URL: {url}")
            if params:
                logger.info(f"With params: {params}")
                
            response = requests.get(
                url, 
                headers=headers, 
                params=params,
                timeout=10  # 10 second timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Received successful response from API: {url}")
                data = response.json()
                
                # Log the structure of the received data
                if isinstance(data, list):
                    logger.info(f"Retrieved {len(data)} tokens")
                    if data and isinstance(data[0], dict):
                        logger.info(f"First token sample keys: {list(data[0].keys())}")
                elif isinstance(data, dict):
                    logger.info(f"Retrieved token data with keys: {list(data.keys())}")
                
                return data
            elif response.status_code == 401:
                raise ValueError("API authentication failed. Please check your API key.")
            elif response.status_code == 403:
                raise ValueError("API access forbidden. Your key may not have sufficient permissions.")
            elif response.status_code == 404:
                raise ValueError("API endpoint not found. Please check the documentation.")
            elif response.status_code == 429:
                raise ValueError("API rate limit exceeded. Please wait and try again.")
            else:
                raise ValueError(f"API error with status code {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise ValueError(f"API request failed: {str(e)}")
    
    def get_all_tokens(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get all tokens from the API or cache.
        
        Args:
            force_refresh: Force refresh from API even if cache is fresh
            
        Returns:
            List of token data
        """
        # Check if we need to refresh the cache
        # Refresh if cache is empty, older than 1 hour, or forced
        current_time = datetime.now()
        cache_age_seconds = (current_time - datetime.fromisoformat(self.token_cache_timestamp)).total_seconds() if self.token_cache_timestamp else float('inf')
        
        if force_refresh or not self.token_cache or cache_age_seconds > 3600:  # 1 hour cache lifetime
            try:
                # Fetch tokens from API
                tokens = self._make_request("tokens")
                
                if tokens and isinstance(tokens, list):
                    # Reset and rebuild the cache
                    self.token_cache = {}
                    for token in tokens:
                        if 'symbol' in token and token['symbol']:
                            self.token_cache[token['symbol'].upper()] = token
                    
                    # Update timestamp and save cache
                    self.token_cache_timestamp = current_time.isoformat()
                    logger.info(f"Updated token cache with {len(self.token_cache)} tokens at {self.token_cache_timestamp}")
                    self._save_token_cache()
                    
                    # Build DEX categories based on tokens
                    self._build_dex_categories()
                    
                    return tokens
                else:
                    logger.warning("Received empty or invalid token data from API")
                    return list(self.token_cache.values()) if self.token_cache else []
            except Exception as e:
                logger.error(f"Error fetching tokens: {str(e)}")
                # Fall back to cached tokens if available
                return list(self.token_cache.values()) if self.token_cache else []
        else:
            # Return cached tokens
            logger.info(f"Using cached token data ({len(self.token_cache)} tokens, {int(cache_age_seconds)} seconds old)")
            return list(self.token_cache.values())
    
    def get_token_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get token data for a specific symbol.
        Implements the GET Token endpoint as specified in the docs.
        
        Args:
            symbol: Token symbol (e.g., 'SOL', 'RAY')
            
        Returns:
            Token data or None if not found
        """
        symbol = symbol.upper()  # Standardize to uppercase
        
        # First check the cache
        if symbol in self.token_cache:
            logger.info(f"Token {symbol} found in cache")
            return self.token_cache[symbol]
        
        # If not in cache, try to fetch from API
        try:
            # Endpoint format: /tokens/{symbol}
            token_data = self._make_request(f"tokens/{symbol}")
            
            if token_data:
                # Handle both list and single object responses
                if isinstance(token_data, list) and token_data:
                    token = token_data[0]
                else:
                    token = token_data
                
                # Update cache
                self.token_cache[symbol] = token
                logger.info(f"Added {symbol} to token cache")
                
                # Save updated cache
                self._save_token_cache()
                
                return token
            else:
                logger.warning(f"Token {symbol} not found in API")
                return None
        except Exception as e:
            logger.error(f"Error fetching token {symbol}: {str(e)}")
            return None
    
    def _build_dex_categories(self) -> None:
        """
        Build a mapping of DEXes to their associated tokens.
        This categorizes tokens by the DEXes they're commonly used with.
        """
        # Define commonly used tokens by DEX
        self.dex_categories = {
            "raydium": [
                "RAY", "SOL", "USDC", "USDT", "FIDA", "SRM", 
                "MNGO", "SAMO", "BONK", "DUST", "ORCA"
            ],
            "meteora": [
                "mSOL", "BTC", "ETH", "USDC", "USDT", "SOL",
                "JTO", "PYTH", "WIF", "JUP"
            ],
            "orca": [
                "SOL", "USDC", "USDT", "ETH", "BTC", "ORCA",
                "mSOL", "SAMO", "BONK", "whETH"
            ]
        }
        
        # Ensure all token symbols are uppercase
        for dex, tokens in self.dex_categories.items():
            self.dex_categories[dex] = [t.upper() for t in tokens]
        
        logger.info(f"Built DEX token categories for {len(self.dex_categories)} DEXes")
    
    def get_token_categories(self) -> Dict[str, List[str]]:
        """Get the mapping of DEXes to their associated tokens"""
        if not self.dex_categories:
            self._build_dex_categories()
        return self.dex_categories
    
    def get_tokens_by_dex(self, dex: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all tokens for a specific DEX.
        
        Args:
            dex: DEX name (e.g., 'raydium', 'meteora', 'orca')
            
        Returns:
            Dictionary of token symbols to token data
        """
        dex = dex.lower()  # Standardize to lowercase
        
        # Ensure we have the latest token data
        self.get_all_tokens()
        
        # Get tokens for this DEX
        tokens = {}
        if dex in self.dex_categories:
            for symbol in self.dex_categories[dex]:
                if symbol in self.token_cache:
                    tokens[symbol] = self.token_cache[symbol]
        
        logger.info(f"Retrieved {len(tokens)} tokens for {dex}")
        return tokens
    
    def get_token_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive token metadata for a specific symbol.
        Combines token data from the API with additional informational metadata.
        
        Args:
            symbol: Token symbol (e.g., 'SOL', 'RAY')
            
        Returns:
            Enhanced token metadata or None if token not found
        """
        symbol = symbol.upper()  # Standardize to uppercase
        
        # Get base token data
        token = self.get_token_by_symbol(symbol)
        if not token:
            return None
        
        # Determine which DEXes use this token
        dexes = []
        for dex, tokens in self.dex_categories.items():
            if symbol in tokens:
                dexes.append(dex)
        
        # Add dex information to the token metadata
        enhanced_metadata = token.copy()
        enhanced_metadata['dexes'] = dexes
        
        return enhanced_metadata
    
    def refresh_all_token_data(self) -> bool:
        """
        Force refresh all token data from the API.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            tokens = self.get_all_tokens(force_refresh=True)
            return bool(tokens)
        except Exception as e:
            logger.error(f"Failed to refresh token data: {str(e)}")
            return False
    
    def search_tokens(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for tokens by name or symbol.
        
        Args:
            query: Search query
            
        Returns:
            List of matching tokens
        """
        query = query.lower()  # Case-insensitive search
        
        # Ensure we have the latest token data
        tokens = self.get_all_tokens()
        
        # Filter tokens that match the query
        matching_tokens = []
        for token in tokens:
            symbol = token.get('symbol', '').lower()
            name = token.get('name', '').lower()
            
            if query in symbol or query in name:
                matching_tokens.append(token)
        
        logger.info(f"Found {len(matching_tokens)} tokens matching '{query}'")
        return matching_tokens
    
    def get_token_pools(self, symbol: str, pool_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find all pools that contain a specific token.
        
        Args:
            symbol: Token symbol
            pool_data: List of pool data to search
            
        Returns:
            List of pools containing the token
        """
        symbol = symbol.upper()  # Standardize to uppercase
        
        matching_pools = []
        for pool in pool_data:
            # Check in the tokens array first
            tokens = pool.get('tokens', [])
            has_token = False
            
            for token in tokens:
                if isinstance(token, dict) and token.get('symbol', '').upper() == symbol:
                    has_token = True
                    break
            
            # Fallback to token1/token2 fields if necessary
            if not has_token:
                token1_symbol = pool.get('token1_symbol', '').upper()
                token2_symbol = pool.get('token2_symbol', '').upper()
                has_token = token1_symbol == symbol or token2_symbol == symbol
            
            if has_token:
                matching_pools.append(pool)
        
        logger.info(f"Found {len(matching_pools)} pools containing {symbol}")
        return matching_pools

# Singleton instance for the token service
_token_service_instance = None

def get_token_service(api_key: Optional[str] = None, base_url: Optional[str] = None) -> TokenDataService:
    """
    Get the singleton instance of the token service.
    
    Args:
        api_key: Optional API key
        base_url: Optional base URL
        
    Returns:
        TokenDataService instance
    """
    global _token_service_instance
    if _token_service_instance is None:
        _token_service_instance = TokenDataService(api_key, base_url)
    return _token_service_instance