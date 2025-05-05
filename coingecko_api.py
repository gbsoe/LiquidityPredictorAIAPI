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
        
        # Set base URL - always use the free API endpoint regardless of API key
        # This is a safer approach and works with demo keys as well
        logger.info("Initializing CoinGecko API client with API key" if api_key else "Initializing CoinGecko API client without API key")
        self.base_url = "https://api.coingecko.com/api/v3"
            
        self.session = requests.Session()
        
        # Token symbol to CoinGecko ID mapping cache
        self.token_id_mapping = {}
        
        # Token address to CoinGecko ID mapping cache
        self.address_id_mapping = {}
        
        # Price cache with timestamps
        self.price_cache = {}
        
        # Token details cache with timestamps
        self.token_details_cache = {}
        
        # Last request timestamp for rate limiting
        self.last_request_time = 0
        
        # Load cached mappings from file if available
        self._load_cache()
        
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
        # With API key we get higher limits but still need to be cautious
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # If we have API key, we can make more requests, but still be cautious
        if self.api_key:
            if time_since_last_request < 0.5:  # ~120 req/min with API key
                time.sleep(0.5 - time_since_last_request)
        else:
            if time_since_last_request < 1.2:  # ~50 req/min without key
                time.sleep(1.2 - time_since_last_request)
        
        # Add API key if available
        if params is None:
            params = {}
            
        if self.api_key:
            if "demo" in self.api_key.lower() or self.base_url == "https://api.coingecko.com/api/v3":
                # For demo or free API keys, add as query parameter
                params["x_cg_demo_api_key"] = self.api_key
                headers = {}
            else:
                # For Pro API keys, add to headers as recommended in CoinGecko docs
                headers = {"x-cg-pro-api-key": self.api_key}
        else:
            headers = {}
        
        try:
            logger.info(f"Making CoinGecko API request to: {url}")
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            self.last_request_time = time.time()
            
            result = response.json()
            logger.info(f"Received successful response from CoinGecko API: {url}?{response.request.path_url.split('?')[1] if '?' in response.request.path_url else ''}")
            return result
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"CoinGecko API request failed: {e.response.status_code} - {e.response.text}")
                # If rate limited, add a longer delay
                if hasattr(e, 'response') and e.response.status_code == 429:
                    logger.warning("Rate limited by CoinGecko API, adding delay before next request")
                    time.sleep(2.0)  # Add extra delay after rate limit
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
    
    def get_price_by_symbol(self, symbol: str, vs_currency: str = "usd", max_age_seconds: int = 300) -> Optional[float]:
        """
        Get the current price for a token by its symbol with caching
        
        Args:
            symbol: Token symbol (e.g., 'SOL', 'BTC')
            vs_currency: Currency to convert to
            max_age_seconds: Maximum age of cached price in seconds (default: 5 minutes)
            
        Returns:
            Token price or None if not found
        """
        # Normalize symbol to uppercase
        symbol = symbol.upper()
        cache_key = f"{symbol}_{vs_currency}"
        
        # Check cache first
        if cache_key in self.price_cache:
            cache_entry = self.price_cache[cache_key]
            cache_time = cache_entry.get("timestamp", 0)
            current_time = time.time()
            
            # If cache is fresh, return cached price
            if current_time - cache_time < max_age_seconds:
                logger.info(f"Using cached price for {symbol}: {cache_entry.get('price')}")
                return cache_entry.get("price")
        
        # Get token ID
        token_id = self.get_token_id(symbol)
        if not token_id:
            logger.warning(f"Could not find CoinGecko ID for symbol: {symbol}")
            return None
        
        # Get fresh price from API
        prices = self.get_price(token_id, vs_currency)
        if token_id in prices and vs_currency in prices[token_id]:
            price = prices[token_id][vs_currency]
            logger.info(f"Updated price for {symbol} from CoinGecko: {price}")
            
            # Update cache
            self.price_cache[cache_key] = {
                "timestamp": time.time(),
                "price": price,
                "token_id": token_id
            }
            
            # Save cache periodically
            if len(self.price_cache) % 10 == 0:  # Save after every 10 new prices
                self._save_cache()
                
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
    
    def get_token_details(self, token_id: str, max_age_seconds: int = 86400) -> Dict[str, Any]:
        """
        Get detailed information about a token with caching
        
        Args:
            token_id: CoinGecko token ID
            max_age_seconds: Maximum age of cached data in seconds (default: 24 hours)
            
        Returns:
            Dictionary with token details
        """
        # Check cache first
        if token_id in self.token_details_cache:
            cache_entry = self.token_details_cache[token_id]
            cache_time = cache_entry.get("timestamp", 0)
            current_time = time.time()
            
            # If cache is fresh, return cached data
            if current_time - cache_time < max_age_seconds:
                logger.info(f"Using cached details for token {token_id}")
                return cache_entry.get("data", {})
        
        # Get fresh data from API
        details = self._make_request(f"/coins/{token_id}")
        
        # Update cache if we got data
        if details:
            self.token_details_cache[token_id] = {
                "timestamp": time.time(),
                "data": details
            }
            # Save cache periodically
            self._save_cache()
        
        return details
    
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

    def _load_cache(self):
        """
        Load cached token data from files
        """
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load token mappings cache
        mapping_cache_file = os.path.join(cache_dir, "coingecko_mappings.json")
        if os.path.exists(mapping_cache_file):
            try:
                with open(mapping_cache_file, "r") as f:
                    cache_data = json.load(f)
                    self.token_id_mapping = cache_data.get("token_id_mapping", {})
                    self.address_id_mapping = cache_data.get("address_id_mapping", {})
                    logger.info(f"Loaded {len(self.token_id_mapping)} token ID mappings from cache")
                    logger.info(f"Loaded {len(self.address_id_mapping)} address mappings from cache")
            except Exception as e:
                logger.error(f"Error loading CoinGecko cache: {e}")
        
        # Load token details cache
        details_cache_file = os.path.join(cache_dir, "coingecko_details.json")
        if os.path.exists(details_cache_file):
            try:
                with open(details_cache_file, "r") as f:
                    self.token_details_cache = json.load(f)
                    logger.info(f"Loaded {len(self.token_details_cache)} token details from cache")
            except Exception as e:
                logger.error(f"Error loading CoinGecko token details cache: {e}")
        
        # Load price cache
        price_cache_file = os.path.join(cache_dir, "coingecko_prices.json")
        if os.path.exists(price_cache_file):
            try:
                with open(price_cache_file, "r") as f:
                    self.price_cache = json.load(f)
                    logger.info(f"Loaded {len(self.price_cache)} token prices from cache")
            except Exception as e:
                logger.error(f"Error loading CoinGecko price cache: {e}")
    
    def _save_cache(self):
        """
        Save cached token data to files
        """
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save token mappings cache
        mapping_cache_file = os.path.join(cache_dir, "coingecko_mappings.json")
        try:
            with open(mapping_cache_file, "w") as f:
                json.dump({
                    "token_id_mapping": self.token_id_mapping,
                    "address_id_mapping": self.address_id_mapping
                }, f, indent=2)
            logger.info(f"Saved {len(self.token_id_mapping)} token ID mappings to cache")
        except Exception as e:
            logger.error(f"Error saving CoinGecko mappings cache: {e}")
        
        # Save token details cache
        details_cache_file = os.path.join(cache_dir, "coingecko_details.json")
        try:
            with open(details_cache_file, "w") as f:
                json.dump(self.token_details_cache, f, indent=2)
            logger.info(f"Saved {len(self.token_details_cache)} token details to cache")
        except Exception as e:
            logger.error(f"Error saving CoinGecko token details cache: {e}")
        
        # Save price cache
        price_cache_file = os.path.join(cache_dir, "coingecko_prices.json")
        try:
            with open(price_cache_file, "w") as f:
                json.dump(self.price_cache, f, indent=2)
            logger.info(f"Saved {len(self.price_cache)} token prices to cache")
        except Exception as e:
            logger.error(f"Error saving CoinGecko price cache: {e}")

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