"""
CoinGecko API Client for SolPool Insight.

This module provides a client for fetching token prices from the CoinGecko API.
It includes caching, rate limiting, and error handling to ensure reliable price data.
"""

import os
import time
import json
import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

# Constants for URLs
DEMO_API_URL = "https://api.coingecko.com/api/v3"
PRO_API_URL = "https://pro-api.coingecko.com/api/v3"

class CoinGeckoAPI:
    """
    Client for the CoinGecko API which provides token price data.
    
    Features:
    - Token price lookup by ID or symbol
    - Smart caching with TTL
    - Rate limiting
    - Error handling with backoff
    """
    
    def __init__(self):
        """Initialize the CoinGecko API client."""
        # Get the API key from environment variable
        self.api_key = os.getenv("COINGECKO_API_KEY", "")
        
        # Configure API based on key type
        if not self.api_key:
            # No API key - use standard API with no auth
            self.base_url = DEMO_API_URL
            self.api_key_type = "none"
            logger.info("Initialized CoinGecko API client with no API key")
        elif self.api_key.startswith("CG-"):
            # Demo API key - use standard API
            self.base_url = DEMO_API_URL
            self.api_key_type = "demo"
            logger.info("Initialized CoinGecko API client with Demo API key")
        else:
            # Pro API key - use Pro API
            self.base_url = PRO_API_URL
            self.api_key_type = "pro"
            logger.info("Initialized CoinGecko API client with Pro API key")
        
        # Initialize token cache for prices
        self.price_cache = {}
        self.price_cache_ttl = 900  # 15 minutes TTL for price data
        
        # Initialize token ID cache (maps symbols to IDs) - type annotation to ensure only strings are stored
        self.token_id_cache: Dict[str, str] = {}
        
        # Rate limiting configuration
        self.last_request_time = 0
        self.request_delay = 4.0  # 4.0 seconds between requests to avoid rate limits
        # Set an initial cooldown to prevent immediate API calls on startup
        self.rate_limited_until = time.time() + 5
        
        # Initialize common token mappings
        self._init_common_token_mappings()
        
        logger.info("Initialized CoinGecko API client")
    
    def _init_common_token_mappings(self):
        """Initialize mappings for common token symbols to CoinGecko IDs."""
        self.token_id_cache = {
            # Major tokens
            "SOL": "solana",
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "USDC": "usd-coin",
            "USDT": "tether",
            "BONK": "bonk",
            "BOOP": "boop",  # Added BOOP token mapping
            "RAY": "raydium",
            "ORCA": "orca",
            "MSOL": "marinade-staked-sol",
            "mSOL": "marinade-staked-sol", 
            "STSOL": "lido-staked-sol",
            "stSOL": "lido-staked-sol",
            "WSOL": "wrapped-solana",
            "SRM": "serum",
            "MNGO": "mango-markets",
            "SAMO": "samoyedcoin",
            "SOOMER": "soomer",  # Added SOOMER token mapping
            "FARTCOIN": "fartcoin",  # Added FARTCOIN token mapping
            
            # Specific token variations
            "JUP": "jupiter-exchange", 
            "JUPY": "jupiter",
            "HPSQ": "hedgehog-protocol",
            "MANGO": "mango-markets",
            "ATLAS": "star-atlas",
            "POLIS": "star-atlas-dao",
            "ATLA": "atlas-navi",
            "BSO1": "bastion-protocol",
            "JSOL": "jsol",
            "7I5K": "7i5kld8tev",
            "7KBN": "7kbn",
            "9VMJ": "9vmj-token",
            "UXD": "uxd-stablecoin",
            "SLND": "solend",
            "RENDER": "render-token",
            "REN": "republic-protocol",
            "LDO": "lido-dao",
            "RNDR": "render-token",
            "SNY": "synthetify",
            "PORT": "port-finance",
            "COPE": "cope",
            "FIDA": "bonfida",
            "WBTC": "wrapped-bitcoin",
            "FWZ2": "fluxwave-ecosystem",
            "J1TO": "j1tax",
            "MANG": "mangoman",
            "MPLU": "mplug",
            "LAYER": "unilayer"  # Added LAYER token mapping
        }
        
        # Initialize address-to-ID mapping for Solana tokens
        self.address_to_id = {
            # Make sure addresses are lowercase for consistent lookup
            "so11111111111111111111111111111111111111112": "solana",
            "epjfwdd5aufqssqem2qn1xzybapC8G4wegGkzwyTDt1v".lower(): "usd-coin",
            "es9vmfrzacermjfrf4h2fyd4kconky11mcce8bennybe".lower(): "tether",
            "dezxaz8z7pnrnrjjz3wxborgixca6xjnb7yab1ppb263".lower(): "bonk",
            "boopkpwqe68msxlqbgogs8zbugn4gxalhfwnp7mpp1i".lower(): "boop",  # Added BOOP token address mapping
            "orcaektdk7lkz57vaayr9qensvepfiu6qemu1kektze".lower(): "orca", 
            "4k3dyjzvzp8emzwuxbbcjevwskkk59s5icnly3qrkx6r".lower(): "raydium",
            "msolzycxhdygdzu16g5qsh3i5k3z3kzk7ytfqcjm7so".lower(): "marinade-staked-sol",
            "7dhbwxmci3dt8ufywyzwebLxgycu7y3il6trkn1y7arj".lower(): "lido-staked-sol",
            "7vfcxtuxsx5wjv5jadk17duj4ksgau7utnkj4b963voxs".lower(): "ethereum",
            "9n4nbm75f5ui33zbpyxn59ewsge8cgshtateth5yFej9e".lower(): "bitcoin",
            "atlasxmbpqxbuybxpsv97usa3fpqyeqzqbuhgifcusxx".lower(): "star-atlas",
            "poliswxnnrwc6obu1vhiukQzfjGl4xdsu4g9qjz9qvk".lower(): "star-atlas-dao",
            "7xkxtg2cw87d97txjsdhpbd5jbkheTqa83tzrujosGasu".lower(): "samoyedcoin",
            "jupyiwryjfskupihA7hker8vutaefosybkedznsdvcn".lower(): "jupiter",
            "hpsqmvlym98yd6xekygxwp8qydvvnkpqjttuqzk2hzof9".lower(): "hedgehog-protocol", 
            "mangoczj36ajzykwvj3vny4gtonjfvenjmvvwaxlac".lower(): "mango-markets",
            "jsol21f4hvbzfgxvjw4rtrnuhvqyjyd5axkpbgm".lower(): "jsol",
            # SOOMER token with specific address provided by user
            "CTh5k7EHD2HBX64xZkeBDwmHskWvNq5WB8f4PWuW1hmz".lower(): "soomer",
            # LAYER token address mapping
            "LayerDUjJuZPxiKTfyYYpUM5m3y3gL3nAZL5BQ7uqQb6".lower(): "unilayer",
            # FARTCOIN token address mapping
            "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump".lower(): "fartcoin",
            # Additional specific token addresses
            "msolzycxhdygdzu16g5qsh3i5k3z3kzk7ytfqcjm7so".lower(): "marinade-staked-sol",
            "msolzycxhdygdzu16g5qsh3i5k3z3kzk7ytfqcjm7so".upper(): "marinade-staked-sol",
            "7dhbwxmci3dt8ufywyzweblxgycu7y3il6trkn1y7arj".lower(): "lido-staked-sol",
            "7dhbwxmci3dt8ufywyzweblxgycu7y3il6trkn1y7arj".upper(): "lido-staked-sol"
        }
    
    def _rate_limit_request(self):
        """Apply rate limiting to API requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.request_delay:
            sleep_time = self.request_delay - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, headers: Optional[Dict[str, str]] = None) -> Any:
        """
        Make a rate-limited request to the CoinGecko API.
        
        Args:
            endpoint: API endpoint to call
            headers: Optional request headers (including API key)
            
        Returns:
            JSON response from API or None on error
        """
        # Check if we're in a rate-limited cooldown period
        if hasattr(self, 'rate_limited_until') and time.time() < self.rate_limited_until:
            logger.warning(f"Skipping CoinGecko request - rate limit cooldown ({int(self.rate_limited_until - time.time())}s remaining)")
            return None
            
        # Apply rate limiting
        self._rate_limit_request()
        
        # Create default headers if none provided
        if headers is None:
            headers = {}
            
        # Prepare parameters object for query parameters
        params = {}
        
        # If endpoint already has query parameters, extract them
        if "?" in endpoint:
            endpoint_parts = endpoint.split("?", 1)
            endpoint = endpoint_parts[0]
            query_str = endpoint_parts[1]
            
            # Parse existing query parameters
            for param in query_str.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key] = value
                    
        # Add authentication based on API key type
        if self.api_key_type == "demo" and self.api_key:
            # Demo API key - add as header and query parameter
            headers["x-cg-demo-api-key"] = self.api_key
            params["x_cg_demo_api_key"] = self.api_key
        elif self.api_key_type == "pro" and self.api_key:
            # Pro API key - add as header only
            headers["x-cg-pro-api-key"] = self.api_key
            
        # Build the URL with the base URL
        url = f"{self.base_url}/{endpoint}"
        
        # Log the request details
        logger.info(f"Making CoinGecko API request to: {url}")
            
        try:
            logger.debug(f"Request headers: {headers}")
            logger.debug(f"Request params: {params}")
            
            # Convert params dict to query string
            query_string = "&".join([f"{k}={v}" for k, v in params.items()]) if params else ""
            
            # Add query string to URL if it exists
            if query_string:
                url = f"{url}?{query_string}"
                
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Received successful response from CoinGecko API: {url}")
                # Reset request delay on successful request
                if self.request_delay > 1.5:
                    self.request_delay = 1.5
                return response.json()
            else:
                logger.error(f"Error accessing CoinGecko API: {response.status_code} - {response.text}")
                if response.status_code == 429:
                    logger.warning("Rate limit exceeded. Implementing temporary pause.")
                    # Set a 60-second pause on all CoinGecko requests
                    self.rate_limited_until = time.time() + 60
                    logger.warning(f"CoinGecko requests paused for 60 seconds")
                    # Increase delay for future requests
                    self.request_delay = min(self.request_delay * 2.0, 8.0)
                return None
        except Exception as e:
            logger.error(f"Exception during CoinGecko API request: {str(e)}")
            return None
    
    def search_token(self, query: str, headers: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Search for tokens on CoinGecko by name or symbol.
        
        Args:
            query: Token name or symbol to search for
            headers: Optional API headers
            
        Returns:
            List of matching tokens with ID, symbol, and name
        """
        endpoint = f"search?query={query}"
        result = self._make_request(endpoint, headers=headers)
        
        if result and "coins" in result:
            return result["coins"]
        
        return []
    
    def get_token_id(self, symbol: str) -> Optional[str]:
        """
        Get the CoinGecko ID for a token symbol.
        
        Args:
            symbol: Token symbol (e.g., SOL, BTC)
            
        Returns:
            CoinGecko ID or None if not found
        """
        # Normalize symbol
        symbol = symbol.upper()
        
        # Check cache first
        if symbol in self.token_id_cache:
            return self.token_id_cache[symbol]
        
        # Search for the token
        search_results = self.search_token(symbol)
        
        for token in search_results:
            if token.get("symbol", "").upper() == symbol:
                # Found a match, cache it
                token_id = token.get("id")
                if token_id:  # Ensure token_id is not None
                    self.token_id_cache[symbol] = token_id
                    logger.info(f"Found CoinGecko ID for {symbol}: {token_id}")
                    return token_id
        
        logger.warning(f"Could not find CoinGecko ID for token: {symbol}")
        return None
    
    def get_price(self, token_ids: List[str], vs_currency: str = "usd", headers: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Get prices for multiple token IDs from CoinGecko.
        
        Args:
            token_ids: List of token IDs
            vs_currency: Currency to get price in (default: USD)
            headers: Optional request headers for authentication
            
        Returns:
            Dictionary of token IDs -> price data
        """
        # Filter out empty tokens
        token_ids = [tid for tid in token_ids if tid]
        
        if not token_ids:
            return {}
        
        # Check cache for fresh prices
        cached_results = {}
        tokens_to_fetch = []
        
        for token_id in token_ids:
            if token_id in self.price_cache:
                cache_entry = self.price_cache[token_id]
                cache_time = cache_entry.get("timestamp", 0)
                current_time = time.time()
                
                # If cache is fresh (within TTL), use it
                if current_time - cache_time < self.price_cache_ttl:
                    cached_results[token_id] = cache_entry
                else:
                    tokens_to_fetch.append(token_id)
            else:
                tokens_to_fetch.append(token_id)
        
        # If all prices were in cache, return them
        if not tokens_to_fetch:
            return {token_id: data["price_data"] for token_id, data in cached_results.items()}
        
        # Otherwise, fetch fresh prices from API
        ids_str = ",".join(tokens_to_fetch)
        
        # For Demo API keys, include as query parameter
        api_key = os.getenv("COINGECKO_API_KEY")
        if api_key and api_key.startswith("CG-") and self.api_key_type == "demo":
            endpoint = f"simple/price?ids={ids_str}&vs_currencies={vs_currency}&x_cg_demo_api_key={api_key}"
        else:
            endpoint = f"simple/price?ids={ids_str}&vs_currencies={vs_currency}"
        
        # Make the API request with authentication headers
        result = self._make_request(endpoint, headers=headers)
        current_time = time.time()
        
        if result:
            # Update cache with new results
            for token_id in tokens_to_fetch:
                if token_id in result:
                    price_data = result[token_id]
                    self.price_cache[token_id] = {
                        "price_data": price_data,
                        "timestamp": current_time
                    }
            
            # Combine with cached results
            combined_results = {**result}
            for token_id, data in cached_results.items():
                combined_results[token_id] = data["price_data"]
            
            return combined_results
        else:
            # If API failed, return whatever we had in cache
            return {token_id: data["price_data"] for token_id, data in cached_results.items()}
    
    def add_token_mapping(self, symbol: str, token_id: str) -> bool:
        """
        Add a new token mapping from symbol to CoinGecko ID.
        This is used to enhance price fetching capabilities.
        
        Args:
            symbol: Token symbol (e.g., "SOL", "BTC")
            token_id: CoinGecko ID (e.g., "solana", "bitcoin")
            
        Returns:
            True if the mapping was added, False otherwise
        """
        if not symbol or not token_id:
            logger.warning("Empty symbol or token_id passed to add_token_mapping")
            return False
            
        # Normalize symbol to uppercase
        symbol = symbol.upper()
        
        # Add to token_id_cache
        self.token_id_cache[symbol] = token_id
        logger.info(f"Added token mapping: {symbol} -> {token_id}")
        
        return True
        
    def add_address_mapping(self, address: str, token_id: str) -> bool:
        """
        Add a new token mapping from address to CoinGecko ID.
        This is used to enhance price fetching capabilities.
        
        Args:
            address: Token address (e.g., Solana token address)
            token_id: CoinGecko ID (e.g., "solana", "bitcoin")
            
        Returns:
            True if the mapping was added, False otherwise
        """
        if not address or not token_id:
            logger.warning("Empty address or token_id passed to add_address_mapping")
            return False
            
        # Normalize address to lowercase for consistent lookup
        address_lower = address.lower()
        
        # Add to address-to-id mapping
        if hasattr(self, 'address_to_id'):
            self.address_to_id[address_lower] = token_id
            logger.info(f"Added address mapping: {address} -> {token_id}")
            return True
        else:
            logger.warning("address_to_id mapping not available")
            return False

    def get_token_price_by_symbol(self, symbol: str, vs_currency: str = "usd") -> Optional[float]:
        """
        Get price for a token symbol.
        
        Args:
            symbol: Token symbol (e.g., SOL, BTC)
            vs_currency: Currency to get price in (default: USD)
            
        Returns:
            Token price or None if not found
        """
        # Normalize symbol
        symbol = symbol.upper()
        
        # Get token ID
        token_id = self.get_token_id(symbol)
        
        if not token_id:
            logger.warning(f"No CoinGecko ID found for token symbol: {symbol}")
            return None
        
        # Get price from API
        result = self.get_price([token_id], vs_currency)
        
        if result and token_id in result:
            price_data = result[token_id]
            price = price_data.get(vs_currency, 0)
            logger.info(f"Retrieved price for {symbol} ({token_id}): {price} {vs_currency.upper()}")
            return price
        
        logger.warning(f"Failed to get price for {symbol}")
        return None
    
    def get_token_prices_by_symbols(self, symbols: List[str], vs_currency: str = "usd") -> Dict[str, float]:
        """
        Get prices for multiple token symbols.
        
        Args:
            symbols: List of token symbols
            vs_currency: Currency to get price in (default: USD)
            
        Returns:
            Dictionary of symbols -> prices
        """
        # Normalize symbols
        symbols = [s.upper() for s in symbols if s]
        
        if not symbols:
            return {}
        
        # Get token IDs for all symbols
        token_ids = []
        symbol_to_id_map = {}
        
        for symbol in symbols:
            token_id = self.get_token_id(symbol)
            if token_id:
                token_ids.append(token_id)
                symbol_to_id_map[symbol] = token_id
        
        # Get prices for all token IDs
        price_data = self.get_price(token_ids, vs_currency)
        
        # Map prices back to symbols
        result = {}
        for symbol, token_id in symbol_to_id_map.items():
            if token_id in price_data:
                result[symbol] = price_data[token_id].get(vs_currency, 0)
            else:
                result[symbol] = 0
        
        return result
    
    def get_token_price_by_address(self, address: str, platform: str = "solana", vs_currency: str = "usd", headers: Optional[Dict[str, str]] = None) -> Optional[float]:
        """
        Get price for a token by address.
        
        Args:
            address: Token address (e.g., Solana token address)
            platform: Blockchain platform (default: solana)
            vs_currency: Currency to get price in (default: USD)
            headers: Optional headers for authentication
            
        Returns:
            Token price or None if not found
        """
        if not address:
            logger.warning("Empty address passed to get_token_price_by_address")
            return None
            
        # Create headers with API key if not provided
        if headers is None:
            headers = {}
            # Add API key from environment if available
            api_key = os.getenv("COINGECKO_API_KEY")
            use_query_param = False
            
            if api_key:
                # Check type of API key
                if api_key.startswith("CG-"):
                    # For Demo API key
                    headers["x-cg-demo-api-key"] = api_key
                    use_query_param = True
                    logger.info("Using CoinGecko Demo API key for address lookup")
                else:
                    # For Pro API keys (maintain backward compatibility)
                    headers["x-cg-pro-api-key"] = api_key
                    headers["x-cg-api-key"] = api_key
                    logger.info("Using CoinGecko Pro API key for address lookup")
        
        # First, check if we have the address in our address-to-id mapping
        address_lower = address.lower()
        if hasattr(self, 'address_to_id') and address_lower in self.address_to_id:
            token_id = self.address_to_id[address_lower]
            logger.info(f"Using cached token ID mapping for address {address}: {token_id}")
            
            # Get price using the token ID
            result = self.get_price([token_id], vs_currency, headers=headers)
            if result and token_id in result:
                price_data = result[token_id]
                price = price_data.get(vs_currency, 0)
                logger.info(f"Retrieved price for token address {address} via ID {token_id}: {price} {vs_currency.upper()}")
                return price
        
        # If we don't have a mapping or it failed, try the direct API endpoint
        api_key = os.getenv("COINGECKO_API_KEY")
        if api_key and api_key.startswith("CG-") and self.api_key_type == "demo":
            # For Demo API keys, include as query parameter
            endpoint = f"simple/token_price/{platform}?contract_addresses={address}&vs_currencies={vs_currency}&x_cg_demo_api_key={api_key}"
        else:
            # For Pro API keys or no key, use standard query format
            endpoint = f"simple/token_price/{platform}?contract_addresses={address}&vs_currencies={vs_currency}"
        
        # Make the API request with authentication headers
        result = self._make_request(endpoint, headers=headers)
        
        if result and address in result:
            price_data = result[address]
            price = price_data.get(vs_currency, 0)
            logger.info(f"Retrieved price for token address {address}: {price} {vs_currency.upper()}")
            return price
        
        logger.warning(f"Failed to get price for token address: {address}")
        return None

# Singleton instance for use throughout the app
coingecko_api = CoinGeckoAPI()