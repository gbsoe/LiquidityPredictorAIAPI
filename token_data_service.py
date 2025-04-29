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

# Import the DeFi Aggregation API client
from defi_aggregation_api import DefiAggregationAPI

# Configure logging
logger = logging.getLogger(__name__)

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
        
        logger.info("Initialized token data service")
    
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
            "name": token_symbol,
            "address": "",
            "decimals": 0,
            "logo": "",
            "price": 0,
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
        if not address:
            return {
                "symbol": "UNKNOWN",
                "name": "Unknown Token",
                "address": "",
                "decimals": 0,
                "logo": "",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            }
            
        # Look in cache first
        for token_data in self.token_cache.values():
            if token_data.get("address", "").lower() == address.lower():
                if not force_refresh:
                    return token_data
                break
        
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
            "decimals": 0,
            "logo": "",
            "price": 0,
            "last_updated": datetime.now().isoformat(),
        }
    
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
            # Extract token data from the API response
            processed = {
                "symbol": token_data.get("symbol", "UNKNOWN"),
                "name": token_data.get("name", "Unknown Token"),
                "address": token_data.get("address", token_data.get("mint", "")),
                "decimals": token_data.get("decimals", 0),
                "logo": token_data.get("logoURI", token_data.get("logo", "")),
                "price": token_data.get("price", 0),
                "coingecko_id": token_data.get("coingeckoId", ""),
                "last_updated": datetime.now().isoformat(),
            }
            
            # Add any additional fields from the API
            for key, value in token_data.items():
                if key not in processed:
                    processed[key] = value
            
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
                "decimals": 0,
                "logo": "",
                "price": 0,
                "last_updated": datetime.now().isoformat(),
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get token service statistics.
        
        Returns:
            Dictionary with token service statistics
        """
        # Calculate hit ratio
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_ratio = self.stats["cache_hits"] / total if total > 0 else 0
        
        return {
            "total_requests": self.stats["total_requests"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "api_errors": self.stats["api_errors"],
            "hit_ratio": hit_ratio,
            "cache_size": len(self.token_cache),
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