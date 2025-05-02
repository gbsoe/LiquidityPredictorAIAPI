"""
Token Data Service for SolPool Insight.

This module provides a service for retrieving, caching, and managing token data.
It supports:
- Token metadata retrieval from DeFi Aggregation API
- Token price tracking
- Token smart caching with TTL
- Default token data for common tokens
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Import the DeFi Aggregation API client
from defi_aggregation_api import DefiAggregationAPI

# Import the CoinGecko API client
try:
    from coingecko_api import CoinGeckoAPI, coingecko_api
except ImportError:
    logger.error("Unable to import CoinGeckoAPI - continuing without CoinGecko integration")
    coingecko_api = None

# Singleton token data service instance
_instance = None
_lock = threading.Lock()

class TokenDataService:
    """
    Service for retrieving, caching, and managing token data.
    
    Features:
    - Token metadata retrieval
    - Token price tracking
    - Smart caching with TTL
    - Default token data for common tokens
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the token data service.
        
        Args:
            api_key: API key for the DeFi Aggregation API
            base_url: Base URL for the DeFi Aggregation API
        """
        # Create API client
        self.api_client = DefiAggregationAPI(api_key=api_key, base_url=base_url)
        
        # Initialize token cache
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        self.token_cache_ttl = 3600  # 1 hour TTL for token data
        
        # Stats tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_errors": 0,
            "last_update": None,
        }
        
        # Initialize default token data
        self._init_default_tokens()
        
        # Preload all token data from API
        self.preload_all_tokens()
        
        logger.info("Initialized token data service")
        
    def preload_all_tokens(self):
        """
        Preload all token data from the API into the cache.
        This significantly improves token data availability and reduces Unknown token displays.
        """
        logger.info("Preloading all token data from API...")
        try:
            # Make API request to get all tokens using the internal _make_request method
            all_tokens = self.api_client._make_request("tokens")
            
            if not all_tokens or not isinstance(all_tokens, list):
                logger.warning("Empty or invalid response when preloading tokens")
                return
                
            logger.info(f"Retrieved {len(all_tokens)} tokens from API for preloading")
            
            # Track statistics for token data completeness
            tokens_loaded = 0
            tokens_with_address = 0
            tokens_with_name = 0
            tokens_with_price = 0
            address_cached = 0
            
            # First pass: Process all tokens and add them all to the cache
            for token in all_tokens:
                # Skip invalid tokens
                if not isinstance(token, dict):
                    continue
                    
                symbol = token.get("symbol", "").upper()
                address = token.get("address", "")
                
                # Skip tokens without either symbol or address
                if not symbol and not address:
                    continue
                
                # Process for consistent format
                processed_token = self._process_token_data(token)
                
                # Update cache by symbol if available
                if symbol:
                    # Only update cache if we have more data than before
                    current = self.token_cache.get(symbol, {})
                    
                    # Decide if we should update the cache
                    should_update = False
                    
                    # If we don't have this token yet, definitely add it
                    if symbol not in self.token_cache:
                        should_update = True
                    # If we have better data (address or name present), update it
                    elif (not current.get("address") and processed_token.get("address")) or \
                         (not current.get("name") and processed_token.get("name")):
                        should_update = True
                    # If we have a price and the current one doesn't
                    elif current.get("price", 0) == 0 and processed_token.get("price", 0) > 0:
                        should_update = True
                        
                    if should_update:
                        self.token_cache[symbol] = processed_token
                        tokens_loaded += 1
                        
                # Always cache by address if available, regardless of symbol
                address = processed_token.get("address", "")
                if address and address.strip():
                    # Add to address-keyed cache
                    self.token_cache[address] = processed_token
                    address_cached += 1
                    
                    # Update stats
                    if processed_token.get("name"):
                        tokens_with_name += 1
                    if processed_token.get("price", 0) > 0:
                        tokens_with_price += 1
                    tokens_with_address += 1
            
            # Report results including address-cached tokens
            logger.info(f"Successfully preloaded {tokens_loaded} tokens by symbol from API")
            logger.info(f"Additionally cached {address_cached} tokens by address")
            logger.info(f"Token data completeness: {tokens_with_address}/{tokens_loaded+address_cached} have addresses, " + 
                       f"{tokens_with_name}/{tokens_loaded+address_cached} have names, {tokens_with_price}/{tokens_loaded+address_cached} have prices")
            
            # Start a background thread to fetch prices for the most common tokens
            # This prevents blocking the UI during initial load
            self._start_background_price_fetching()
            
        except Exception as e:
            logger.error(f"Error preloading tokens from API: {str(e)}")
            # Fall back to default tokens which were already loaded
            
    def _start_background_price_fetching(self):
        """Start a background thread to fetch token prices without blocking the UI"""
        try:
            # Try to fetch any missing common tokens directly first
            self._fetch_missing_common_tokens()
            
            # Create a comprehensive list of priority tokens to fetch prices for
            priority_tokens = [
                "SOL", "USDC", "USDT", "ETH", "BTC", "BONK", "RAY", "ORCA", 
                "MSOL", "STSOL", "JUP", "JUPY", "ATLAS", "UXD", "HPsQ", 
                "SRM", "SAMO", "SLND", "mSOL", "stSOL", "MNGO", "COPE",
                "LDO", "RNDR", "FIDA", "ATLA", "POLI"
            ]
            
            # Start a background thread to fetch prices
            logger.info("Starting background thread for token price fetching")
            
            def fetch_prices_background():
                logger.info(f"Background price fetching started for {len(priority_tokens)} priority tokens")
                for symbol in priority_tokens:
                    if symbol in self.token_cache:
                        try:
                            # Check if we already have a price
                            if self.token_cache[symbol].get("price", 0) > 0:
                                logger.info(f"Token {symbol} already has price: {self.token_cache[symbol].get('price')}")
                                continue
                                
                            # Fetch price in background
                            price = None
                            
                            # Try to get price from CoinGecko if available
                            if coingecko_api is not None:
                                # First try by coingecko_id if available
                                coingecko_id = self.token_cache[symbol].get("coingecko_id")
                                if coingecko_id:
                                    try:
                                        logger.info(f"Fetching price for {symbol} using CoinGecko ID: {coingecko_id}")
                                        result = coingecko_api.get_price([coingecko_id], "usd")
                                        if result and coingecko_id in result:
                                            price = result[coingecko_id].get("usd", 0)
                                            logger.info(f"Retrieved price for {symbol} using ID {coingecko_id}: {price}")
                                        else:
                                            logger.warning(f"No price data returned for {symbol} with ID {coingecko_id}")
                                    except Exception as e:
                                        logger.warning(f"Error fetching price by ID for {symbol}: {e}")
                                
                                # If that didn't work, try by symbol
                                if not price or price == 0:
                                    logger.info(f"Trying to fetch price for {symbol} by symbol")
                                    try:
                                        price = coingecko_api.get_token_price_by_symbol(symbol)
                                    except Exception as e:
                                        logger.warning(f"Error fetching price by symbol for {symbol}: {e}")
                                
                                # If still no price, try by address
                                if (not price or price == 0) and self.token_cache[symbol].get("address"):
                                    address = self.token_cache[symbol].get("address")
                                    if address:  # Check if address is not None
                                        logger.info(f"Trying to fetch price for {symbol} by address: {address}")
                                        try:
                                            price = coingecko_api.get_token_price_by_address(address)
                                        except Exception as e:
                                            logger.warning(f"Error fetching price by address for {symbol}: {e}")
                            
                            # Update token cache with price if found
                            if price and price > 0:
                                logger.info(f"Updated price for {symbol} in background: {price}")
                                self.token_cache[symbol]["price"] = price
                                self.token_cache[symbol]["price_source"] = "coingecko"
                                self.token_cache[symbol]["last_updated"] = datetime.now().isoformat()
                                
                                # If token is also cached by address, update that cache entry too
                                address = self.token_cache[symbol].get("address")
                                if address and address in self.token_cache:
                                    self.token_cache[address]["price"] = price
                                    self.token_cache[address]["price_source"] = "coingecko"
                                    self.token_cache[address]["last_updated"] = datetime.now().isoformat()
                            
                            # Add a larger delay to avoid rate limits
                            time.sleep(3)
                            
                        except Exception as e:
                            logger.warning(f"Error fetching price for {symbol}: {str(e)}")
                
                logger.info("Background price fetching completed")
            
            # Start the background thread
            price_thread = threading.Thread(target=fetch_prices_background, daemon=True)
            price_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting background price fetching: {str(e)}")
            
    def _fetch_missing_common_tokens(self):
        """Fetch any missing data for common tokens directly"""
        # Expanded list of common tokens to ensure better coverage
        common_tokens = [
            "SOL", "USDC", "USDT", "ETH", "BTC", "MSOL", "BONK", "RAY", "ORCA", 
            "STSOL", "ATLA", "POLI", "JSOL", "JUPY", "HPSQ", "MNGO", "SAMO", 
            "7I5K", "7KBN", "9VMJ", "BSO1", "FWZ2", "J1TO", "MANG", "MPLU",
            "mSOL", "stSOL", "JUP", "SRM", "ATLAS", "UXD", "SLND", "COPE",
            "LDO", "RNDR", "FIDA" 
        ]
        
        logger.info(f"Fetching missing data for {len(common_tokens)} common tokens")
        
        # First try to get them all at once from the API
        try:
            all_tokens = self.api_client._make_request("tokens")
            if all_tokens and isinstance(all_tokens, list):
                for token in all_tokens:
                    symbol = token.get("symbol", "").upper()
                    if symbol and symbol in common_tokens:
                        processed = self._process_token_data(token)
                        self.token_cache[symbol] = processed
                        
                        # Also cache by address if available for direct lookups
                        address = processed.get("address", "")
                        if address and address.strip():
                            self.token_cache[address] = processed
                            
                        logger.info(f"Fetched common token from bulk API: {symbol}")
        except Exception as e:
            logger.warning(f"Error fetching all tokens: {e}")
        
        # Now try to fetch any remaining tokens individually
        for symbol in common_tokens:
            if symbol not in self.token_cache or not self.token_cache[symbol].get("address"):
                try:
                    # Try to fetch directly by symbol
                    token_data = self.api_client._make_request(f"tokens/{symbol}")
                    if token_data:
                        processed = self._process_token_data(token_data)
                        self.token_cache[symbol] = processed
                        
                        # Also cache by address if available for direct lookups
                        address = processed.get("address", "")
                        if address and address.strip():
                            self.token_cache[address] = processed
                            
                        logger.info(f"Directly fetched missing token: {symbol}")
                except Exception as e:
                    logger.warning(f"Could not fetch missing token {symbol}: {e}")
    
    def _init_default_tokens(self):
        """Initialize default token data for common tokens."""
        logger.info("Initializing default token data")
        
        # Default token data for common tokens
        default_tokens = {
            "SOL": {
                "symbol": "SOL",
                "name": "Solana",
                "address": "So11111111111111111111111111111111111111112",
                "decimals": 9,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/So11111111111111111111111111111111111111112/logo.png",
                "coingecko_id": "solana",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            },
            "USDC": {
                "symbol": "USDC",
                "name": "USD Coin",
                "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "decimals": 6,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v/logo.png",
                "coingecko_id": "usd-coin",
                "price": 1.0,
                "last_updated": datetime.now().isoformat(),
            },
            "BONK": {
                "symbol": "BONK",
                "name": "Bonk",
                "address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
                "decimals": 5,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263/logo.png",
                "coingecko_id": "bonk",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            },
            "ORCA": {
                "symbol": "ORCA",
                "name": "Orca",
                "address": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
                "decimals": 6,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE/logo.png",
                "coingecko_id": "orca",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            },
            "RAY": {
                "symbol": "RAY",
                "name": "Raydium",
                "address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
                "decimals": 6,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R/logo.png",
                "coingecko_id": "raydium",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            },
            "MSOL": {
                "symbol": "MSOL",
                "name": "Marinade Staked SOL",
                "address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
                "decimals": 9,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So/logo.png",
                "coingecko_id": "marinade-staked-sol",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            },
            "STSOL": {
                "symbol": "STSOL",
                "name": "Lido Staked SOL",
                "address": "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",
                "decimals": 9,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj/logo.png",
                "coingecko_id": "lido-staked-sol",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            },
            "ETH": {
                "symbol": "ETH",
                "name": "Ethereum",
                "address": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
                "decimals": 8,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs/logo.png",
                "coingecko_id": "ethereum",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            },
            "ATLAS": {
                "symbol": "ATLAS",
                "name": "Star Atlas",
                "address": "ATLASXmbPQxBUYbxPsV97usA3fPQYEqzQBUHgiFCUsXx",
                "decimals": 8,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/ATLASXmbPQxBUYbxPsV97usA3fPQYEqzQBUHgiFCUsXx/logo.png",
                "coingecko_id": "star-atlas",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            },
            "UXD": {
                "symbol": "UXD",
                "name": "UXD Stablecoin",
                "address": "7kbnvuGBxxj8AG9qp8Scn56muWGaRaFqxg1FsRp3PaFT",
                "decimals": 6,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/7kbnvuGBxxj8AG9qp8Scn56muWGaRaFqxg1FsRp3PaFT/logo.png",
                "coingecko_id": "uxd-stablecoin",
                "price": 1.0,
                "last_updated": datetime.now().isoformat(),
            },
            "ETH": {
                "symbol": "ETH",
                "name": "Wrapped Ethereum (Sollet)",
                "address": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
                "decimals": 8,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs/logo.png",
                "coingecko_id": "ethereum",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            },
            "BTC": {
                "symbol": "BTC",
                "name": "Wrapped Bitcoin (Sollet)",
                "address": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
                "decimals": 6,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E/logo.png",
                "coingecko_id": "bitcoin",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            },
            "USDT": {
                "symbol": "USDT",
                "name": "USDT",
                "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                "decimals": 6,
                "logo": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB/logo.png",
                "coingecko_id": "tether",
                "price": 1.0,
                "last_updated": datetime.now().isoformat(),
            },
        }
        
        # Add default tokens to cache
        for symbol, token_data in default_tokens.items():
            self.token_cache[symbol] = token_data
            
            # Also cache by address for direct lookups
            address = token_data.get("address", "")
            if address and address.strip():
                self.token_cache[address] = token_data
        
        logger.info(f"Initialized default token cache with {len(default_tokens)} tokens")
    
    def get_token_data(self, token_symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get token data by symbol.
        
        Args:
            token_symbol: Token symbol (e.g., "SOL", "USDC")
            force_refresh: Force a refresh from the API
            
        Returns:
            Token data or fallback data if not found
        """
        token_symbol = token_symbol.upper()
        
        # Skip processing for UNKNOWN tokens to reduce API calls for non-existent tokens
        if token_symbol == "UNKNOWN":
            return {
                "symbol": "UNKNOWN",
                "name": "Unknown Token",
                "address": "",
                "decimals": 0,
                "logo": "",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            }
        
        # Update stats
        self.stats["total_requests"] += 1
        
        # Check cache first if not forcing refresh
        if not force_refresh and token_symbol in self.token_cache:
            token_data = self.token_cache[token_symbol]
            cache_time = datetime.fromisoformat(token_data.get("last_updated", "2020-01-01T00:00:00"))
            
            # Check if cache is still valid
            if (datetime.now() - cache_time).total_seconds() < self.token_cache_ttl:
                self.stats["cache_hits"] += 1
                logger.debug(f"Token {token_symbol} found in cache")
                return token_data
        
        # Cache miss or forced refresh
        self.stats["cache_misses"] += 1
        
        try:
            # First try to get the specific token
            logger.info(f"Making API request for token: {token_symbol}")
            token_data = self.api_client._make_request(f"tokens/{token_symbol}")
            
            if token_data:
                # Process the token data
                processed_data = self._process_token_data(token_data)
                
                # Update the cache
                processed_data["last_updated"] = datetime.now().isoformat()
                self.token_cache[token_symbol] = processed_data
                
                # Also cache by address if available for faster lookups
                token_address = processed_data.get("address", "")
                if token_address:
                    self.token_cache[token_address] = processed_data
                
                self.stats["last_update"] = datetime.now().isoformat()
                
                return processed_data
        except Exception as e:
            logger.info(f"Token {token_symbol} not found via direct lookup, trying token list")
            
            # If direct token lookup fails, try fetching from the list
            try:
                # Get all tokens and find the one we want
                all_tokens = self.api_client._make_request("tokens")
                
                if isinstance(all_tokens, list):
                    # Find the token in the list
                    for token in all_tokens:
                        if token.get("symbol", "").upper() == token_symbol:
                            # Found the token in the list
                            processed_data = self._process_token_data(token)
                            
                            # Update the cache
                            processed_data["last_updated"] = datetime.now().isoformat()
                            self.token_cache[token_symbol] = processed_data
                            
                            # Also cache by address if available for faster lookups
                            token_address = processed_data.get("address", "")
                            if token_address:
                                self.token_cache[token_address] = processed_data
                            
                            self.stats["last_update"] = datetime.now().isoformat()
                            logger.info(f"Found token {token_symbol} in tokens list")
                            
                            return processed_data
            except Exception as list_error:
                logger.warning(f"Failed to get token {token_symbol} from tokens list: {str(list_error)}")
            
            # Both attempts failed
            logger.warning(f"Token {token_symbol} not found in API")
            self.stats["api_errors"] += 1
        
        # If we get here, API request failed or returned invalid data
        if token_symbol in self.token_cache:
            # Use cached data even if expired
            logger.info(f"Using cached token data for {token_symbol}")
            return self.token_cache[token_symbol]
        
        # Return basic info if token not found
        return {
            "symbol": token_symbol,
            "name": f"{token_symbol} Token",
            "address": "",
            "decimals": 6,  # Default to 6 decimals for Solana SPL tokens
            "logo": "",
            "price": 0,
            "chain": "solana",
            "last_updated": datetime.now().isoformat(),
        }
    
    def get_token_by_address(self, address: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get token data by address.
        
        Args:
            address: Token address
            force_refresh: Force a refresh from the API
            
        Returns:
            Token data or a fallback structure if not found
        """
        if not address or not isinstance(address, str):
            return {
                "symbol": "UNKNOWN",
                "name": "Unknown Token",
                "address": "",
                "decimals": 6,  # Default to 6 decimals for Solana SPL tokens
                "logo": "",
                "price": 0,
                "chain": "solana",
                "last_updated": datetime.now().isoformat(),
            }
            
        # Update stats
        self.stats["total_requests"] += 1
            
        # First check if address is directly in cache (from our enhanced preloading)
        if not force_refresh and address in self.token_cache:
            self.stats["cache_hits"] += 1
            
            # Track direct address cache hits separately for analytics
            if "direct_address_hits" not in self.stats:
                self.stats["direct_address_hits"] = 0
            self.stats["direct_address_hits"] += 1
            
            return self.token_cache[address]
            
        # Then check by comparing addresses in cache
        if not force_refresh:
            for symbol, token_data in self.token_cache.items():
                if token_data.get("address", "").lower() == address.lower():
                    self.stats["cache_hits"] += 1
                    # Also add to cache with address as key for future lookups
                    self.token_cache[address] = token_data
                    return token_data
        
        # Cache miss, increment stats
        self.stats["cache_misses"] += 1
        
        # If not in cache or forcing refresh, try API directly
        try:
            logger.info(f"Making API request for token by address: {address}")
            token_data = self.api_client._make_request(f"tokens/address/{address}")
            
            if token_data:
                # Process the token data
                processed_data = self._process_token_data(token_data)
                
                # Update the cache
                processed_data["last_updated"] = datetime.now().isoformat()
                symbol = processed_data.get("symbol", "UNKNOWN")
                self.token_cache[symbol] = processed_data
                
                # Also cache by address for direct lookups
                self.token_cache[address] = processed_data
                
                self.stats["last_update"] = datetime.now().isoformat()
                logger.info(f"Found token with address {address} via direct lookup")
                
                return processed_data
        except Exception as e:
            logger.info(f"Token with address {address} not found via direct lookup, trying token list")
            
            # If direct lookup fails, try searching in all tokens
            try:
                # Get all tokens and search by address
                all_tokens = self.api_client._make_request("tokens")
                
                if isinstance(all_tokens, list):
                    for token in all_tokens:
                        if token.get("address", "").lower() == address.lower():
                            # Found the token in the list
                            processed_data = self._process_token_data(token)
                            
                            # Update the cache
                            processed_data["last_updated"] = datetime.now().isoformat()
                            symbol = processed_data.get("symbol", "UNKNOWN")
                            self.token_cache[symbol] = processed_data
                            
                            # Also cache by address for faster lookups next time
                            self.token_cache[address] = processed_data
                            
                            self.stats["last_update"] = datetime.now().isoformat()
                            logger.info(f"Found token with address {address} in tokens list")
                            
                            return processed_data
            except Exception as list_error:
                logger.warning(f"Failed to get token with address {address} from tokens list: {str(list_error)}")
            
            # Both attempts failed
            logger.warning(f"Token with address {address} not found in API")
            self.stats["api_errors"] += 1
        
        # If we get here, API request failed or returned invalid data
        # Return a minimal token data structure with the address
        return {
            "symbol": "UNKNOWN",
            "name": "Unknown Token",
            "address": address,
            "decimals": 6,  # Default to 6 decimals for Solana SPL tokens
            "logo": "",
            "price": 0,
            "chain": "solana",
            "last_updated": datetime.now().isoformat(),
        }
    
    def get_token_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get token metadata by symbol.
        
        Args:
            symbol: Token symbol (e.g., "SOL", "USDC")
            
        Returns:
            Token metadata dictionary with symbol, name, address, etc.
        """
        # This method uses get_token_data for consistency
        return self.get_token_data(token_symbol=symbol)
    
    def get_tokens_for_pool(self, pool_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get detailed token data for a pool.
        
        Args:
            pool_data: Pool data with token information
            
        Returns:
            List of enriched token data
        """
        tokens = []
        
        # Check if the pool already has token data in the tokens array
        if "tokens" in pool_data and isinstance(pool_data["tokens"], list):
            pool_tokens = pool_data["tokens"]
            
            for token_data in pool_tokens:
                token_symbol = token_data.get("symbol", "")
                if token_symbol:
                    # Get detailed token data
                    detailed_token = self.get_token_data(token_symbol)
                    
                    # Merge with pool token data
                    merged_token = {**token_data, **detailed_token}
                    
                    # Ensure price is included
                    if "price" not in merged_token or not merged_token["price"]:
                        merged_token["price"] = token_data.get("price", 0)
                    
                    tokens.append(merged_token)
                else:
                    # If no symbol, see if we can get by address
                    token_address = token_data.get("address", "")
                    if token_address:
                        detailed_token = self.get_token_by_address(token_address)
                        tokens.append({**token_data, **detailed_token})
                    else:
                        # Just use the original token data
                        tokens.append(token_data)
        else:
            # Fallback to legacy token fields
            token1_symbol = pool_data.get("token1_symbol", "")
            token2_symbol = pool_data.get("token2_symbol", "")
            
            if token1_symbol:
                token1_data = self.get_token_data(token1_symbol)
                token1_data["price"] = pool_data.get("token1_price", 0)
                tokens.append(token1_data)
            
            if token2_symbol:
                token2_data = self.get_token_data(token2_symbol)
                token2_data["price"] = pool_data.get("token2_price", 0)
                tokens.append(token2_data)
        
        return tokens
    
    def update_pool_token_data(self, pool_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a pool with enhanced token data.
        
        Args:
            pool_data: Pool data to enhance
            
        Returns:
            Enhanced pool data with detailed token information
        """
        enhanced_pool = pool_data.copy()
        
        # Get enhanced token data
        tokens = self.get_tokens_for_pool(pool_data)
        
        # Update the pool with enhanced token data
        enhanced_pool["tokens"] = tokens
        
        # Update legacy token fields for backward compatibility
        if len(tokens) > 0:
            enhanced_pool["token1_symbol"] = tokens[0].get("symbol", enhanced_pool.get("token1_symbol", ""))
            enhanced_pool["token1_address"] = tokens[0].get("address", enhanced_pool.get("token1_address", ""))
            enhanced_pool["token1_price"] = tokens[0].get("price", enhanced_pool.get("token1_price", 0))
            enhanced_pool["token1_decimals"] = tokens[0].get("decimals", enhanced_pool.get("token1_decimals", 0))
            enhanced_pool["token1_logo"] = tokens[0].get("logo", enhanced_pool.get("token1_logo", ""))
        
        if len(tokens) > 1:
            enhanced_pool["token2_symbol"] = tokens[1].get("symbol", enhanced_pool.get("token2_symbol", ""))
            enhanced_pool["token2_address"] = tokens[1].get("address", enhanced_pool.get("token2_address", ""))
            enhanced_pool["token2_price"] = tokens[1].get("price", enhanced_pool.get("token2_price", 0))
            enhanced_pool["token2_decimals"] = tokens[1].get("decimals", enhanced_pool.get("token2_decimals", 0))
            enhanced_pool["token2_logo"] = tokens[1].get("logo", enhanced_pool.get("token2_logo", ""))
        
        return enhanced_pool
    
    def _process_token_data(self, token_data: Any) -> Dict[str, Any]:
        """
        Process token data from the API.
        
        Args:
            token_data: Raw token data from the API
            
        Returns:
            Processed token data
        """
        if isinstance(token_data, dict):
            # Extract the symbol with proper validation
            raw_symbol = token_data.get("symbol", "")
            # Convert to uppercase and remove any whitespace or unwanted characters
            symbol = raw_symbol.upper().strip() if raw_symbol else "UNKNOWN"
            
            # Get token address from various possible sources
            address = ""
            for address_key in ["address", "mint", "tokenAddress", "mintAddress"]:
                if address_key in token_data and token_data[address_key]:
                    address = token_data[address_key]
                    break
            
            # Get token name with proper validation and formatting
            raw_name = token_data.get("name", "")
            if raw_name:
                name = raw_name.strip()
            elif symbol and symbol != "UNKNOWN":
                # If no name but symbol exists, create a readable name from symbol
                name = f"{symbol} Token"
            else:
                # Last resort fallback
                name = "Unknown Token"
            
            # Get decimal places with validation
            try:
                decimals = int(token_data.get("decimals", 0))
            except (ValueError, TypeError):
                # Default to 6 decimals for Solana SPL tokens if value is invalid
                decimals = 6
                
            # Get price with validation
            try:
                price = float(token_data.get("price", 0))
            except (ValueError, TypeError):
                price = 0
                
            # Extract token data from the API response
            processed = {
                "symbol": symbol,
                "name": name,
                "address": address,
                "decimals": decimals,
                "logo": token_data.get("logoURI", token_data.get("logo", "")),
                "price": price,
                "price_source": token_data.get("price_source", "defi_api"),
                "coingecko_id": token_data.get("coingeckoId", ""),
                "last_updated": datetime.now().isoformat(),
                "id": token_data.get("id", 0),  # Add token ID from the API
                "active": bool(token_data.get("active", True)),
            }
            
            # Add any additional fields from the API
            for key, value in token_data.items():
                if key not in processed:
                    processed[key] = value
            
            # Get token price from CoinGecko if price is 0 or missing
            symbol = processed.get("symbol", "").upper()
            if symbol and symbol != "UNKNOWN" and processed.get("price", 0) == 0:
                try:
                    # Handle special token cases
                    # For MSOL and STSOL, add special handling since these are common problematic tokens
                    if symbol in ["MSOL", "mSOL"]:
                        # Set explicit coingecko_id for Marinade Staked SOL
                        processed["coingecko_id"] = "marinade-staked-sol"
                        logger.info(f"Using explicit CoinGecko ID mapping for {symbol}: marinade-staked-sol")
                    elif symbol in ["STSOL", "stSOL"]:
                        # Set explicit coingecko_id for Lido Staked SOL
                        processed["coingecko_id"] = "lido-staked-sol"
                        logger.info(f"Using explicit CoinGecko ID mapping for {symbol}: lido-staked-sol")
                    
                    # Skip CoinGecko if API client is not available
                    if coingecko_api is None:
                        logger.warning(f"CoinGecko API not available, skipping price lookup for {symbol}")
                        price = None
                    else:
                        # Try with CoinGecko ID first if available
                        coingecko_id = processed.get("coingecko_id")
                        price = None
                        
                        if coingecko_id:
                            # Use the ID directly if available
                            logger.info(f"Fetching price for {symbol} using CoinGecko ID: {coingecko_id}")
                            result = coingecko_api.get_price([coingecko_id], "usd")
                            if result and coingecko_id in result:
                                price = result[coingecko_id].get("usd", 0)
                                logger.info(f"Retrieved price for {symbol} using coingecko_id {coingecko_id}: {price}")
                            else:
                                logger.warning(f"No price data returned for {symbol} with ID {coingecko_id}")
                        
                        # If no price or no coingecko_id, try by symbol
                        if not price or price == 0:
                            logger.info(f"Fetching price for {symbol} from CoinGecko by symbol")
                            price = coingecko_api.get_token_price_by_symbol(symbol)
                            if price and price > 0:
                                logger.info(f"Retrieved price for {symbol} by symbol lookup: {price}")
                        
                        # If still no price, try by address
                        if (not price or price == 0) and processed.get("address"):
                            address = processed.get("address")
                            if address and isinstance(address, str):
                                logger.info(f"Fetching price for {symbol} from CoinGecko by address: {address}")
                                price = coingecko_api.get_token_price_by_address(address)
                                if price and price > 0:
                                    logger.info(f"Retrieved price for {symbol} by address lookup: {price}")
                    
                    # Special case for staked SOL tokens: if price still not available, 
                    # use SOL price and slightly adjust it
                    if (not price or price == 0) and symbol in ["MSOL", "mSOL", "STSOL", "stSOL"]:
                        logger.info(f"Trying to estimate {symbol} price based on SOL price")
                        sol_price = 0
                        
                        # Try to get SOL price from cache or API
                        if "SOL" in self.token_cache and self.token_cache["SOL"].get("price", 0) > 0:
                            sol_price = self.token_cache["SOL"].get("price", 0)
                            if sol_price and sol_price > 0:
                                logger.info(f"Using cached SOL price for {symbol} estimation: {sol_price}")
                        elif coingecko_api is not None:
                            try:
                                # Only fetch SOL price if we're not in a cooldown period
                                if not hasattr(coingecko_api, 'rate_limited_until') or time.time() >= coingecko_api.rate_limited_until:
                                    sol_price_value = coingecko_api.get_token_price_by_symbol("SOL")
                                    if sol_price_value and sol_price_value > 0:
                                        sol_price = sol_price_value
                                        logger.info(f"Using fresh SOL price for {symbol} estimation: {sol_price}")
                            except Exception as e:
                                logger.warning(f"Error fetching SOL price for estimation: {e}")
                            
                        # Apply appropriate multiplier based on token
                        if sol_price and isinstance(sol_price, (int, float)) and sol_price > 0:
                            if symbol in ["MSOL", "mSOL"]:
                                price = sol_price * 1.08  # MSOL typically has ~8% premium
                                logger.info(f"Estimated MSOL price from SOL: {price}")
                            elif symbol in ["STSOL", "stSOL"]:
                                price = sol_price * 1.06  # STSOL typically has ~6% premium
                                logger.info(f"Estimated STSOL price from SOL: {price}")
                    
                    # Update price if found
                    if price is not None and price > 0:
                        processed["price"] = price
                        processed["price_source"] = "coingecko"
                        logger.info(f"Updated price for {symbol} from CoinGecko: {price}")
                
                except Exception as e:
                    logger.warning(f"Error getting price from CoinGecko for {symbol}: {str(e)}")
            
            return processed
        elif isinstance(token_data, list) and len(token_data) > 0:
            # If API returned a list, use the first item
            return self._process_token_data(token_data[0])
        else:
            # Invalid token data
            return {
                "symbol": "UNKNOWN",
                "name": "Unknown Token",
                "address": "",
                "decimals": 6,  # Default to 6 decimals for Solana SPL tokens
                "logo": "",
                "price": 0,
                "chain": "solana",
                "last_updated": datetime.now().isoformat(),
            }
    
    def get_all_tokens(self) -> List[Dict[str, Any]]:
        """
        Fetch all tokens from the API.
        
        Returns:
            A list of token data dictionaries.
        """
        try:
            # Track request stats
            self.stats["total_requests"] += 1
            self.stats["last_update"] = datetime.now().isoformat()
            
            # Make API request to get all tokens
            all_tokens = self.api_client._make_request("tokens")
            
            if not all_tokens or not isinstance(all_tokens, list):
                logger.warning("Empty or invalid response when fetching all tokens")
                return list(self.token_cache.values())  # Return cached tokens as fallback
            
            # Process each token and add to the cache with proper formatting
            tokens_loaded = 0
            for token in all_tokens:
                # Check if token has required fields
                if "symbol" in token and "address" in token:
                    symbol = token.get("symbol", "").upper()
                    
                    # Process the token using our standard method for consistency
                    processed_token = self._process_token_data(token)
                    
                    # Add to cache
                    self.token_cache[symbol] = processed_token
                    tokens_loaded += 1
            
            logger.info(f"Loaded and processed {tokens_loaded} tokens from API")
            
            # Return processed tokens from cache to ensure consistent formatting
            return list(self.token_cache.values())
            
        except Exception as e:
            logger.error(f"Error fetching all tokens: {str(e)}")
            self.stats["api_errors"] += 1
            
            # Return cached tokens as fallback
            if self.token_cache:
                return list(self.token_cache.values())
            return []
    
    def get_token_categories(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get token categorization by DEX or platform.
        
        Returns:
            Dictionary with DEX names as keys and lists of tokens as values
        """
        try:
            # Get all tokens
            tokens = self.get_all_tokens()
            
            # Get supported DEXes from the DeFi Aggregation API
            dexes = self.api_client._make_request("dexes")
            
            # Define default categories if API doesn't return them
            if not dexes or not isinstance(dexes, list):
                dexes = [
                    {"name": "Raydium", "id": "raydium"},
                    {"name": "Orca", "id": "orca"},
                    {"name": "Meteora", "id": "meteora"}
                ]
            
            # Initialize categories
            categories = {dex.get("id", "unknown"): [] for dex in dexes}
            categories["other"] = []  # For tokens not in specific DEX
            
            # Get pools to map tokens to DEXes
            try:
                all_pools = []
                # Try to fetch pools from each DEX
                for dex in dexes:
                    dex_id = dex.get("id", "")
                    if dex_id:
                        pools = self.api_client._make_request(
                            f"pools?source={dex_id}&limit=100"
                        )
                        if pools and isinstance(pools, list):
                            all_pools.extend(pools)
            except Exception as e:
                logger.error(f"Error fetching pools for token categorization: {str(e)}")
                all_pools = []
            
            # Create a mapping of tokens to DEXes
            token_to_dex_map = {}
            for pool in all_pools:
                dex = pool.get("source", "other")
                # Extract token info (handle different API formats)
                tokens_in_pool = []
                if "token1_symbol" in pool:
                    tokens_in_pool.append(pool.get("token1_symbol", "").upper())
                    tokens_in_pool.append(pool.get("token2_symbol", "").upper())
                elif "tokens" in pool and isinstance(pool["tokens"], list):
                    tokens_in_pool = [t.get("symbol", "").upper() for t in pool["tokens"] 
                                    if "symbol" in t]
                
                # Update token-to-dex mapping
                for token_symbol in tokens_in_pool:
                    if token_symbol:
                        if token_symbol not in token_to_dex_map:
                            token_to_dex_map[token_symbol] = set()
                        token_to_dex_map[token_symbol].add(dex)
            
            # Categorize tokens by DEX
            for token in tokens:
                symbol = token.get("symbol", "").upper()
                if symbol in token_to_dex_map:
                    # Add token to each DEX category it belongs to
                    for dex in token_to_dex_map[symbol]:
                        if dex in categories:
                            categories[dex].append(token)
                else:
                    # Add to other category if not found in any DEX
                    categories["other"].append(token)
            
            return categories
            
        except Exception as e:
            logger.error(f"Error categorizing tokens: {str(e)}")
            # Return minimal default categories
            return {
                "raydium": self._get_default_tokens_for_dex("raydium"),
                "orca": self._get_default_tokens_for_dex("orca"),
                "meteora": self._get_default_tokens_for_dex("meteora"),
                "other": []
            }
    
    def _get_default_tokens_for_dex(self, dex_name: str) -> List[Dict[str, Any]]:
        """Helper to get default tokens for a DEX when API fails"""
        common_tokens = list(self.token_cache.values())
        
        # For demo purposes, distribute tokens across DEXes
        if dex_name == "raydium":
            return [t for t in common_tokens if t.get("symbol", "").upper() in 
                   ["SOL", "USDC", "RAY", "ATLAS"]]
        elif dex_name == "orca":
            return [t for t in common_tokens if t.get("symbol", "").upper() in 
                   ["ORCA", "USDT", "BTC", "ETH"]]
        elif dex_name == "meteora":
            return [t for t in common_tokens if t.get("symbol", "").upper() in 
                   ["BONK", "MSOL", "UXD"]]
        
        return []
        
    def get_tokens_by_dex(self, dex_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get tokens used by a specific DEX.
        
        Args:
            dex_name: Name of the DEX
            
        Returns:
            Dictionary mapping token symbols to token data
        """
        logger.info(f"Getting tokens for DEX: {dex_name}")
        
        try:
            # Get categories from token_categories
            categories = self.get_token_categories()
            
            # Get tokens for the requested DEX
            dex_tokens = categories.get(dex_name, [])
            
            # Convert to dictionary keyed by symbol
            result = {}
            for token in dex_tokens:
                symbol = token.get("symbol", "").upper()
                if symbol:
                    result[symbol] = token
                    
            if not result:
                logger.warning(f"No tokens found for DEX: {dex_name}, using defaults")
                # If no tokens found, use defaults
                default_tokens = self._get_default_tokens_for_dex(dex_name)
                for token in default_tokens:
                    symbol = token.get("symbol", "").upper()
                    if symbol:
                        result[symbol] = token
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting tokens by DEX {dex_name}: {str(e)}")
            # Return default tokens in case of error
            default_tokens = self._get_default_tokens_for_dex(dex_name)
            result = {}
            for token in default_tokens:
                symbol = token.get("symbol", "").upper()
                if symbol:
                    result[symbol] = token
                    
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get token service statistics.
        
        Returns:
            Dictionary with token service statistics
        """
        # Calculate hit ratio
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_ratio = self.stats["cache_hits"] / total if total > 0 else 0
        
        # Count symbols and addresses in cache
        symbols = 0
        addresses = 0
        
        for key in self.token_cache.keys():
            # Heuristic to identify likely addresses vs symbols
            if len(key) > 30:
                addresses += 1
            else:
                symbols += 1
        
        # Add direct address hits to stats if not present
        if "direct_address_hits" not in self.stats:
            self.stats["direct_address_hits"] = 0
        
        return {
            "total_requests": self.stats["total_requests"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "api_errors": self.stats["api_errors"],
            "hit_ratio": hit_ratio,
            "cache_size": len(self.token_cache),
            "tokens_by_symbol": symbols,
            "tokens_by_address": addresses,
            "direct_address_hits": self.stats["direct_address_hits"],
            "last_update": self.stats["last_update"],
        }
    
    def clear_cache(self):
        """Clear the token cache."""
        logger.info("Clearing token cache")
        
        # Preserve default tokens
        default_tokens = {}
        for symbol, token_data in self.token_cache.items():
            if symbol in ["SOL", "USDC", "USDT", "ETH", "BTC", "BONK", "ORCA", "RAY", "ATLAS", "UXD"]:
                default_tokens[symbol] = token_data
        
        # Reset cache
        self.token_cache = default_tokens
        
        # Reset stats
        self.stats["cache_hits"] = 0
        self.stats["cache_misses"] = 0
        self.stats["api_errors"] = 0
        self.stats["last_update"] = datetime.now().isoformat()
        
        logger.info(f"Token cache cleared. Retained {len(default_tokens)} default tokens.")


def get_token_service(api_key: Optional[str] = None, base_url: Optional[str] = None) -> TokenDataService:
    """
    Get the singleton token data service instance.
    
    Args:
        api_key: Optional API key
        base_url: Optional base URL
        
    Returns:
        TokenDataService instance
    """
    global _instance, _lock
    
    with _lock:
        if _instance is None:
            _instance = TokenDataService(api_key=api_key, base_url=base_url)
    
    return _instance