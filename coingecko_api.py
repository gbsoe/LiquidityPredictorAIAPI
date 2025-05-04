"""
CoinGecko API Client

This module provides a client for retrieving cryptocurrency price data from CoinGecko,
which is a reliable source for token price information.
"""

import os
import time
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoinGeckoAPI:
    """Client for retrieving cryptocurrency data from CoinGecko"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the CoinGecko API client
        
        Args:
            api_key: Optional CoinGecko API key for higher rate limits
        """
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        
        # Token symbol to CoinGecko ID mapping cache
        self.token_id_mapping = {}
        
        # Token address to CoinGecko ID mapping cache
        self.address_id_mapping = {}
        
        # Last request timestamp for rate limiting
        self.last_request_time = 0
        
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a rate-limited request to the CoinGecko API
        
        Args:
            endpoint: API endpoint (relative to base URL)
            params: Query parameters
            
        Returns:
            Response data as a dictionary
        """
        url = f"{self.base_url}{endpoint}"
        
        # Ensure we don't exceed rate limits
        # CoinGecko free tier allows ~50 calls per minute
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < 1.2:  # Limit to ~50 req/min
            time.sleep(1.2 - time_since_last_request)
        
        # Add API key if available
        if params is None:
            params = {}
            
        if self.api_key:
            params["x_cg_demo_api_key"] = self.api_key
        
        try:
            logger.info(f"Making CoinGecko API request to: {url}")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            self.last_request_time = time.time()
            
            result = response.json()
            logger.info(f"Received successful response from CoinGecko API: {url}?{response.request.path_url.split('?')[1] if '?' in response.request.path_url else ''}")
            return result
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"CoinGecko API request failed: {e.response.status_code} - {e.response.text}")
            else:
                logger.error(f"CoinGecko API request failed: {str(e)}")
            
            # Return empty dict on failure
            return {}
    
    def search_token(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for tokens by name or symbol
        
        Args:
            query: Search query (name or symbol)
            
        Returns:
            List of matching tokens
        """
        result = self._make_request("/search", {"query": query})
        coins = result.get("coins", [])
        
        # Cache mappings for found coins
        for coin in coins:
            symbol = coin.get("symbol", "").upper()  # Normalize to uppercase
            coin_id = coin.get("id")
            
            if symbol and coin_id:
                self.token_id_mapping[symbol] = coin_id
                logger.info(f"Added token mapping: {symbol} -> {coin_id}")
            
            # Cache token address mappings
            platforms = coin.get("platforms", {})
            for platform, address in platforms.items():
                if address and coin_id:
                    self.address_id_mapping[address] = coin_id
                    logger.info(f"Added address mapping: {address} -> {coin_id}")
        
        return coins
    
    def get_token_id(self, symbol: str) -> Optional[str]:
        """
        Get the CoinGecko token ID for a given symbol
        
        Args:
            symbol: Token symbol (e.g., 'SOL', 'BTC')
            
        Returns:
            CoinGecko token ID or None if not found
        """
        # Normalize to uppercase
        symbol = symbol.upper()
        
        # Return from cache if available
        if symbol in self.token_id_mapping:
            return self.token_id_mapping[symbol]
        
        # Search for the token
        coins = self.search_token(symbol)
        
        # Find exact match by symbol
        for coin in coins:
            if coin.get("symbol", "").upper() == symbol:
                coin_id = coin.get("id")
                if coin_id:
                    self.token_id_mapping[symbol] = coin_id
                    return coin_id
        
        # No exact match found
        return None
    
    def get_token_id_by_address(self, address: str) -> Optional[str]:
        """
        Get the CoinGecko token ID for a given contract address
        
        Args:
            address: Token contract address
            
        Returns:
            CoinGecko token ID or None if not found
        """
        # Return from cache if available
        if address in self.address_id_mapping:
            return self.address_id_mapping[address]
        
        # Search for the token - no direct API for this, so we need to use a general search
        # This is less reliable than using a symbol
        coins = self.search_token(address)
        
        # Check if any returned coin has this address
        for coin in coins:
            platforms = coin.get("platforms", {})
            for platform, coin_address in platforms.items():
                if coin_address == address:
                    coin_id = coin.get("id")
                    if coin_id:
                        self.address_id_mapping[address] = coin_id
                        return coin_id
        
        # No match found
        return None
    
    def get_price(self, token_ids: Union[str, List[str]], vs_currencies: Union[str, List[str]] = "usd") -> Dict[str, Dict[str, float]]:
        """
        Get current prices for tokens
        
        Args:
            token_ids: Single token ID or list of token IDs
            vs_currencies: Single currency or list of currencies to convert to
            
        Returns:
            Dictionary of token prices by ID and currency
        """
        # Convert single values to lists
        if isinstance(token_ids, str):
            token_ids = [token_ids]
        
        if isinstance(vs_currencies, str):
            vs_currencies = [vs_currencies]
        
        # Join lists into comma-separated strings
        ids = ",".join(token_ids)
        currencies = ",".join(vs_currencies)
        
        # Make the request
        result = self._make_request("/simple/price", {
            "ids": ids,
            "vs_currencies": currencies
        })
        
        return result
    
    def get_price_by_symbol(self, symbol: str, vs_currency: str = "usd") -> Optional[float]:
        """
        Get the current price for a token by its symbol
        
        Args:
            symbol: Token symbol (e.g., 'SOL', 'BTC')
            vs_currency: Currency to convert to
            
        Returns:
            Token price or None if not found
        """
        token_id = self.get_token_id(symbol)
        if not token_id:
            logger.warning(f"Could not find CoinGecko ID for symbol: {symbol}")
            return None
        
        prices = self.get_price(token_id, vs_currency)
        if token_id in prices and vs_currency in prices[token_id]:
            price = prices[token_id][vs_currency]
            logger.info(f"Updated price for {symbol} from CoinGecko: {price}")
            return price
        
        logger.warning(f"Could not get price for token: {symbol}")
        return None
    
    def get_price_by_address(self, address: str, vs_currency: str = "usd") -> Optional[float]:
        """
        Get the current price for a token by its contract address
        
        Args:
            address: Token contract address
            vs_currency: Currency to convert to
            
        Returns:
            Token price or None if not found
        """
        token_id = self.get_token_id_by_address(address)
        if not token_id:
            logger.warning(f"Could not find CoinGecko ID for address: {address}")
            return None
        
        prices = self.get_price(token_id, vs_currency)
        if token_id in prices and vs_currency in prices[token_id]:
            price = prices[token_id][vs_currency]
            logger.info(f"Updated price for token address {address} from CoinGecko: {price}")
            return price
        
        logger.warning(f"Could not get price for token address: {address}")
        return None
    
    def get_token_details(self, token_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a token
        
        Args:
            token_id: CoinGecko token ID
            
        Returns:
            Dictionary with token details
        """
        return self._make_request(f"/coins/{token_id}")
    
    def get_token_details_by_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a token by its symbol
        
        Args:
            symbol: Token symbol (e.g., 'SOL', 'BTC')
            
        Returns:
            Dictionary with token details
        """
        token_id = self.get_token_id(symbol)
        if not token_id:
            logger.warning(f"Could not find CoinGecko ID for symbol: {symbol}")
            return {}
        
        return self.get_token_details(token_id)

# Singleton instance
_instance = None

def get_coingecko_api() -> CoinGeckoAPI:
    """
    Get a singleton instance of the CoinGecko API client
    
    Returns:
        CoinGecko API client instance
    """
    global _instance
    if _instance is None:
        api_key = os.getenv("COINGECKO_API_KEY", None)
        _instance = CoinGeckoAPI(api_key=api_key)
    return _instance


if __name__ == "__main__":
    # Example usage
    try:
        api = get_coingecko_api()
        
        # Get SOL price
        sol_price = api.get_price_by_symbol("SOL")
        print(f"SOL price: ${sol_price}")
        
        # Get multiple token prices
        prices = api.get_price(["bitcoin", "ethereum", "solana"], ["usd", "eur"])
        print(json.dumps(prices, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")