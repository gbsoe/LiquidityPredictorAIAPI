"""
Token Data Service for SolPool Insight

This module provides functions for fetching and managing token data from the DeFi API,
specifically using the /tokens endpoint described in the documentation.

It provides access to comprehensive token information including:
- Token symbols and names
- Token addresses and decimals
- Token prices
- Token categorization by DEX
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token mappings by DEX
DEX_TOKEN_MAPPINGS = {
    "raydium": [
        "RAY", "SOL", "USDC"
    ],
    "meteora": [
        "mSOL", "BTC", "USDC"
    ],
    "orca": [
        "SOL", "ETH", "USDC", "Es9v", "DezX"
    ]
}

class TokenDataService:
    """
    Service for retrieving and managing token data from the DeFi API.
    Provides methods for fetching tokens, caching them, and organizing by DEX.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the token data service.
        
        Args:
            api_key: API key for authentication (defaults to DEFI_API_KEY env var)
            base_url: Base URL for the API (defaults to standard URL)
        """
        self.api_key = api_key or os.environ.get("DEFI_API_KEY")
        self.base_url = base_url or "https://filotdefiapi.replit.app/api/v1"
        self.token_cache = {}
        self.token_cache_by_address = {}
        self.token_cache_by_dex = {
            "raydium": {},
            "meteora": {},
            "orca": {}
        }
        self.last_updated = None
    
    def _make_request(self, endpoint: str) -> List[Dict[str, Any]]:
        """
        Make an authenticated request to the API.
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            API response data as a list of token objects
        
        Raises:
            ValueError: For API errors with specific messages
        """
        url = f"{self.base_url}/{endpoint}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add authorization header if API key is provided
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            logger.info(f"Making API request to URL: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            
            # Check if the request was successful
            if response.status_code == 200:
                logger.info(f"Received successful response from API: {url}")
                data = response.json()
                
                # Check data structure
                if isinstance(data, list):
                    logger.info(f"Retrieved {len(data)} tokens")
                    if data and isinstance(data[0], dict):
                        sample_keys = list(data[0].keys())
                        logger.info(f"First token sample keys: {sample_keys}")
                    return data
                else:
                    raise ValueError(f"Unexpected response format: {type(data)}")
            else:
                error_msg = f"API request failed with status code {response.status_code}"
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict) and "message" in error_data:
                        error_msg += f": {error_data['message']}"
                except:
                    pass
                raise ValueError(error_msg)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request failed: {str(e)}")
    
    def get_all_tokens(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve all tokens from the API.
        
        Args:
            force_refresh: Force a refresh even if cache exists
            
        Returns:
            List of token objects
        """
        # Check if we need to refresh the cache
        cache_expired = self.last_updated is None
        
        if cache_expired or force_refresh or not self.token_cache:
            try:
                # Fetch tokens from the API
                tokens = self._make_request("tokens")
                
                # Update the cache
                self._update_cache(tokens)
                
                return tokens
            except Exception as e:
                logger.error(f"Error fetching tokens: {str(e)}")
                # Return cached tokens if available, otherwise empty list
                return list(self.token_cache.values()) if self.token_cache else []
        else:
            # Return cached tokens
            return list(self.token_cache.values())
    
    def _update_cache(self, tokens: List[Dict[str, Any]]) -> None:
        """
        Update the token cache with fresh data.
        
        Args:
            tokens: List of token objects from the API
        """
        # Reset caches
        self.token_cache = {}
        self.token_cache_by_address = {}
        self.token_cache_by_dex = {
            "raydium": {},
            "meteora": {},
            "orca": {}
        }
        
        # Populate caches
        for token in tokens:
            if "symbol" in token and token["symbol"]:
                symbol = token["symbol"].upper()
                address = token.get("address", "")
                
                # Main token cache by symbol
                self.token_cache[symbol] = token
                
                # Address cache
                if address:
                    self.token_cache_by_address[address] = token
                
                # DEX-specific caches
                for dex, dex_tokens in DEX_TOKEN_MAPPINGS.items():
                    if symbol in dex_tokens:
                        self.token_cache_by_dex[dex][symbol] = token
        
        # Update last updated timestamp
        self.last_updated = datetime.now()
        logger.info(f"Updated token cache with {len(tokens)} tokens at {self.last_updated}")
    
    def get_token_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get token data by symbol.
        
        Args:
            symbol: Token symbol (case insensitive)
            
        Returns:
            Token data or None if not found
        """
        symbol = symbol.upper()
        if symbol in self.token_cache:
            return self.token_cache[symbol]
        
        # If not in cache, try to fetch and update
        self.get_all_tokens(force_refresh=True)
        
        # Check again
        return self.token_cache.get(symbol)
    
    def get_token_by_address(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Get token data by address.
        
        Args:
            address: Token address
            
        Returns:
            Token data or None if not found
        """
        if address in self.token_cache_by_address:
            return self.token_cache_by_address[address]
        
        # If not in cache, try to fetch and update
        self.get_all_tokens(force_refresh=True)
        
        # Check again
        return self.token_cache_by_address.get(address)
    
    def get_tokens_by_dex(self, dex: str) -> Dict[str, Dict[str, Any]]:
        """
        Get tokens used by a specific DEX.
        
        Args:
            dex: DEX name (raydium, meteora, orca)
            
        Returns:
            Dictionary of token symbols to token data
        """
        dex = dex.lower()
        if dex not in self.token_cache_by_dex:
            return {}
        
        # If we have no tokens for this DEX, try to fetch and update
        if not self.token_cache_by_dex[dex]:
            self.get_all_tokens(force_refresh=True)
        
        return self.token_cache_by_dex[dex]
    
    def save_tokens_to_cache_file(self, filename: str = "token_cache.json") -> bool:
        """
        Save token data to a cache file.
        
        Args:
            filename: Cache file name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'w') as f:
                json.dump(list(self.token_cache.values()), f, indent=2)
            
            logger.info(f"Saved {len(self.token_cache)} tokens to cache file: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save tokens to cache: {str(e)}")
            return False
    
    def load_tokens_from_cache_file(self, filename: str = "token_cache.json") -> bool:
        """
        Load token data from a cache file.
        
        Args:
            filename: Cache file name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    tokens = json.load(f)
                
                if isinstance(tokens, list) and tokens:
                    self._update_cache(tokens)
                    logger.info(f"Loaded {len(tokens)} tokens from cache file: {filename}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to load tokens from cache: {str(e)}")
            return False
    
    def get_token_price(self, symbol: str) -> float:
        """
        Get the price of a token by symbol.
        
        Args:
            symbol: Token symbol
            
        Returns:
            Token price or 0 if not found
        """
        token = self.get_token_by_symbol(symbol)
        if token and "price" in token:
            return float(token["price"])
        return 0
    
    def get_token_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive metadata for a token.
        
        Args:
            symbol: Token symbol
            
        Returns:
            Dictionary with token metadata or empty dict if not found
        """
        token = self.get_token_by_symbol(symbol)
        if not token:
            return {}
        
        # Determine which DEXes use this token
        dexes = []
        for dex, dex_tokens in DEX_TOKEN_MAPPINGS.items():
            if symbol.upper() in dex_tokens:
                dexes.append(dex)
        
        return {
            "symbol": token.get("symbol", ""),
            "name": token.get("name", ""),
            "address": token.get("address", ""),
            "decimals": token.get("decimals", 0),
            "price": token.get("price", 0),
            "active": token.get("active", False),
            "dexes": dexes,
            "id": token.get("id", 0)
        }
    
    def get_active_tokens(self) -> List[Dict[str, Any]]:
        """
        Get all active tokens.
        
        Returns:
            List of active token objects
        """
        tokens = self.get_all_tokens()
        return [token for token in tokens if token.get("active", False)]
    
    def get_token_categories(self) -> Dict[str, List[str]]:
        """
        Get token categories by DEX.
        
        Returns:
            Dictionary of DEX names to lists of token symbols
        """
        return {dex: list(self.get_tokens_by_dex(dex).keys()) for dex in DEX_TOKEN_MAPPINGS.keys()}


# Singleton instance for reuse
_token_service_instance = None

def get_token_service() -> TokenDataService:
    """
    Get or create a singleton instance of TokenDataService.
    
    Returns:
        TokenDataService instance
    """
    global _token_service_instance
    if _token_service_instance is None:
        _token_service_instance = TokenDataService()
        # Try to load from cache file first
        if not _token_service_instance.load_tokens_from_cache_file():
            # If loading fails, fetch fresh data
            _token_service_instance.get_all_tokens()
    
    return _token_service_instance

if __name__ == "__main__":
    # Example usage
    token_service = get_token_service()
    tokens = token_service.get_all_tokens()
    print(f"Retrieved {len(tokens)} tokens")
    
    # Save to cache file
    token_service.save_tokens_to_cache_file()