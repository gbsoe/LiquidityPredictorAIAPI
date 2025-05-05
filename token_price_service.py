"""Token Price Service for SolPool Insight
Fetches token prices from CoinGecko API and other sources
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenPriceService:
    """Service for fetching and caching token prices"""
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the token price service
        
        Args:
            use_cache: Whether to use the cache (default: True)
        """
        self.token_mapping = {}
        self.cached_prices = {}
        self.cache_price_sources = {}  # Track which source provided each price
        self.cache_timestamp = 0
        self.use_cache = use_cache
        self.cache_file = "token_price_cache.json"
        self.cache_validity_period = 60 * 60  # 1 hour in seconds
        
        # Load static mappings for common tokens
        self._load_default_mappings()
        
        # Load cached prices if available
        if use_cache:
            self._load_cache()
    
    def _load_default_mappings(self) -> None:
        """Load default token symbol to CoinGecko ID mappings"""
        # Common token mappings
        self.token_mapping = {
            "SOL": "solana",
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "USDC": "usd-coin",
            "USDT": "tether",
            "BONK": "bonk",
            "WIF": "dogwifhat",
            "JTO": "jito-governance",
            "PYTH": "pyth-network",
            "RNDR": "render-token",
            "RAY": "raydium",
            "JUP": "jupiter-exchange",
            "SAMO": "samoyedcoin",
            "DFL": "defi-land",
            "DUST": "dust-protocol",
            "USDC": "usd-coin",
            "USDT": "tether",
            "SOOMER": "soomer",
        }
    
    def _load_cache(self) -> None:
        """Load cached prices from file if available"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    cache_data = json.load(f)
                    self.cached_prices = cache_data.get("prices", {})
                    self.cache_timestamp = cache_data.get("timestamp", 0)
                    self.cache_price_sources = cache_data.get("sources", {})
                    logger.info(f"Loaded {len(self.cached_prices)} token prices from cache")
        except Exception as e:
            logger.error(f"Error loading price cache: {e}")
            # Reset cache if there was an error
            self.cached_prices = {}
            self.cache_timestamp = 0
            self.cache_price_sources = {}
    
    def _save_cache(self) -> None:
        """Save current prices to cache file"""
        try:
            cache_data = {
                "prices": self.cached_prices,
                "timestamp": int(time.time()),
                "sources": self.cache_price_sources
            }
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f)
            logger.info(f"Saved {len(self.cached_prices)} token prices to cache")
        except Exception as e:
            logger.error(f"Error saving price cache: {e}")
    
    def _is_cache_valid(self) -> bool:
        """Check if the cache is still valid"""
        if not self.use_cache:
            return False
            
        current_time = int(time.time())
        age = current_time - self.cache_timestamp
        return age < self.cache_validity_period
    
    def get_token_price_from_defi_api(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a token directly from the DeFi API
        
        Args:
            symbol: Token symbol (e.g., "SOL", "BTC")
            
        Returns:
            Current price in USD, or None if not available
        """
        try:
            # Normalize symbol
            symbol = symbol.upper()
            
            # Special handling for stablecoins
            if symbol in ["USDC", "USDT", "DAI", "BUSD", "USDH", "TUSD"]:
                return 1.0
                
            # DeFi API endpoint for token prices
            api_url = "https://api.defi-insight.com/api/v1/token-prices"
            params = {"symbol": symbol}
            
            response = requests.get(api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and "price" in data:
                    price = float(data["price"])
                    logger.info(f"Retrieved {symbol} price from DeFi API: ${price}")
                    return price
            
            logger.warning(f"Failed to get {symbol} price from DeFi API: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error getting {symbol} price from DeFi API: {e}")
            return None
    
    def get_token_price(self, symbol: str, return_source: bool = False) -> Union[Optional[float], Tuple[Optional[float], str]]:
        """
        Get the current price for a token, trying multiple sources with CoinGecko as priority
        
        Args:
            symbol: Token symbol (e.g., "SOL", "BTC")
            return_source: Whether to return the price source along with the price
            
        Returns:
            If return_source is False: Current price in USD, or None if not available
            If return_source is True: Tuple of (price, source) where source is "defi_api", "coingecko", or "none"
        """
        # Check if symbol is None or empty
        if not symbol:
            if return_source:
                return (None, "none")
            return None
            
        # Normalize symbol
        symbol = symbol.upper() if isinstance(symbol, str) else symbol
        
        # Check if it's a known stablecoin
        if symbol in ["USDC", "USDT", "DAI", "BUSD", "USDH", "TUSD"]:
            if return_source:
                return (1.0, "fixed")
            return 1.0
        
        # Check cache first
        if self.use_cache and self._is_cache_valid() and symbol in self.cached_prices:
            price = self.cached_prices[symbol]
            source = self.cache_price_sources.get(symbol, "unknown")
            logger.debug(f"Using cached price for {symbol}: ${price} from {source}")
            if return_source:
                return (price, source)
            return price
        
        # Try CoinGecko first via our extended client
        price = None
        source = "none"
        
        # If we have a CoinGecko ID mapping, try that first
        coingecko_id = self.token_mapping.get(symbol)
        if coingecko_id:
            try:
                # Use our improved CoinGeckoAPI client
                from coingecko_api import CoinGeckoAPI
                coingecko_client = CoinGeckoAPI()
                price_data = coingecko_client.get_price([coingecko_id], "usd")
                
                if coingecko_id in price_data and "usd" in price_data[coingecko_id]:
                    price = price_data[coingecko_id]["usd"]
                    source = "coingecko"
                    
                    # Validate SOL price (known issue with some providers)
                    if symbol == "SOL" and price > 1000:
                        logger.warning(f"Skipping unrealistic SOL price from CoinGecko: ${price}")
                        price = None
                        source = "none"  
                    else:
                        # Cache the price
                        self.cached_prices[symbol] = price
                        self.cache_price_sources[symbol] = source
                        logger.info(f"Retrieved {symbol} price from CoinGecko: ${price}")
            except Exception as e:
                logger.warning(f"Error getting {symbol} price from CoinGecko: {e}")
        
        # If CoinGecko failed, try the DeFi API
        if price is None:
            try:
                defi_price = self.get_token_price_from_defi_api(symbol)
                if defi_price is not None:
                    price = defi_price
                    source = "defi_api"
                    
                    # Validate SOL price (known issue with some providers)
                    if symbol == "SOL" and price > 1000:
                        logger.warning(f"Skipping unrealistic SOL price from DeFi API: ${price}")
                        price = None
                        source = "none"
                    else:
                        # Cache the price
                        self.cached_prices[symbol] = price
                        self.cache_price_sources[symbol] = source
                        
            except Exception as e:
                logger.warning(f"Error getting {symbol} price from DeFi API: {e}")
        
        # Save updated cache
        if self.use_cache and price is not None:
            self._save_cache()
            
        if return_source:
            return (price, source)
        return price
    
    def get_price_by_token_address(self, token_address: str) -> Optional[float]:
        """
        Get the price of a token using its address on the Solana blockchain
        
        Args:
            token_address: Solana token address
            
        Returns:
            Current price in USD, or None if not available
        """
        try:
            # Try using CoinGecko API for token address lookup
            from coingecko_api import CoinGeckoAPI
            coingecko_client = CoinGeckoAPI()
            price = coingecko_client.get_price_by_address(token_address, "solana")
            
            if price is not None:
                logger.info(f"Retrieved price for token {token_address} from CoinGecko: ${price}")
                return price
                
            # Fallback to other sources if needed...
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for token address {token_address}: {e}")
            return None
    
    def get_multiple_prices_from_defi_api(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get prices for multiple tokens at once from the DeFi API
        
        Args:
            symbols: List of token symbols
            
        Returns:
            Dictionary mapping symbols to prices
        """
        result = {}
        
        try:
            # Normalize symbols
            normalized_symbols = [s.upper() for s in symbols if s != "UNKNOWN"]
            if not normalized_symbols:
                return result
                
            # Add default prices for stablecoins
            for symbol in normalized_symbols:
                if symbol in ["USDC", "USDT", "DAI", "BUSD", "USDH", "TUSD"]:
                    result[symbol] = 1.0
            
            # Filter out stablecoins
            symbols_to_fetch = [s for s in normalized_symbols if s not in result]
            if not symbols_to_fetch:
                return result
                
            # Batch the symbols to avoid too long URLs
            max_symbols_per_request = 10
            batches = [symbols_to_fetch[i:i+max_symbols_per_request] for i in range(0, len(symbols_to_fetch), max_symbols_per_request)]
            
            for batch in batches:
                # DeFi API endpoint for multiple token prices
                api_url = "https://api.defi-insight.com/api/v1/token-prices/batch"
                params = {"symbols": ",".join(batch)}
                
                response = requests.get(api_url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, dict):
                        for symbol, price_data in data.items():
                            if "price" in price_data:
                                price = float(price_data["price"])
                                result[symbol.upper()] = price
                        
                logger.info(f"Retrieved {len(result)} token prices from DeFi API")
            
            return result
                
        except Exception as e:
            logger.error(f"Error fetching multiple prices from DeFi API: {e}")
            return result
            
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get prices for multiple tokens at once, prioritizing CoinGecko as the source
        
        Args:
            symbols: List of token symbols
            
        Returns:
            Dictionary mapping symbols to prices
        """
        # Set default stablecoin prices to reduce API calls
        result = {
            "USDC": 1.0,
            "USDT": 1.0,
            "UST": 1.0,
            "BUSD": 1.0,
            "DAI": 1.0,
            "USDH": 1.0,
            "TUSD": 1.0,
        }
        
        # Filter out stablecoins and unknown tokens
        symbols_to_fetch = [s for s in symbols if s not in result and s != "UNKNOWN"]
        if not symbols_to_fetch:
            return result
            
        # Add delay between API calls
        time.sleep(1.2)
        
        # Check which symbols we need to fetch
        symbols_to_fetch = []
        
        normalized_symbols = [s.upper() for s in symbols if s != "UNKNOWN"]
        
        # Try cache first for CoinGecko prices only
        if self._is_cache_valid():
            for symbol in normalized_symbols:
                cache_source = self.cache_price_sources.get(symbol, "none") if hasattr(self, 'cache_price_sources') else "none"
                # Only trust cache if source is CoinGecko or for stable coins
                if (symbol in self.cached_prices and 
                    (cache_source == "coingecko" or
                     symbol in ["USDC", "USDT", "DAI", "BUSD", "TUSD", "USDH"])):
                    result[symbol] = self.cached_prices[symbol]
                    logger.debug(f"Using cached price for {symbol} from {cache_source}: {self.cached_prices[symbol]}")
                else:
                    symbols_to_fetch.append(symbol)
        else:
            symbols_to_fetch = normalized_symbols
        
        if not symbols_to_fetch:
            return result
        
        # Map symbols to CoinGecko IDs
        symbol_to_id_map = {}
        coingecko_ids = []
        
        for symbol in symbols_to_fetch:
            coingecko_id = self.token_mapping.get(symbol)
            if coingecko_id:
                coingecko_ids.append(coingecko_id)
                symbol_to_id_map[symbol] = coingecko_id
                
        # If we have ID mappings, fetch from CoinGecko in batch
        if coingecko_ids:
            try:
                # Use our improved CoinGeckoAPI client
                from coingecko_api import CoinGeckoAPI
                coingecko_client = CoinGeckoAPI()
                logger.info(f"Fetching prices for {len(coingecko_ids)} tokens from CoinGecko by ID")
                price_data = coingecko_client.get_price(coingecko_ids, "usd")
                
                for symbol, coingecko_id in symbol_to_id_map.items():
                    if coingecko_id in price_data and "usd" in price_data[coingecko_id]:
                        price = price_data[coingecko_id]["usd"]
                        # Validate SOL price
                        if symbol == "SOL" and price > 1000:
                            logger.warning(f"Skipping unrealistic SOL price from CoinGecko: ${price}")
                        else:
                            result[symbol] = price
                            self.cached_prices[symbol] = price
                            if not hasattr(self, 'cache_price_sources'):
                                self.cache_price_sources = {}
                            self.cache_price_sources[symbol] = "coingecko"
                            logger.info(f"Retrieved {symbol} price from CoinGecko client: ${price}")
                            # Remove from symbols that need fetching
                            if symbol in symbols_to_fetch:
                                symbols_to_fetch.remove(symbol)
            except Exception as e:
                logger.warning(f"Error using CoinGecko client for batch price lookup: {e}")
        
        # Try individual symbol lookup for remaining symbols
        remaining_symbols = [s for s in symbols_to_fetch if s not in result]
        for symbol in remaining_symbols[:]:  # Use slice to create a copy for iteration
            try:
                # Try individual lookup for symbol
                from coingecko_api import CoinGeckoAPI
                coingecko_client = CoinGeckoAPI()
                price = coingecko_client.get_price_by_symbol(symbol)
                if price is not None and price > 0:
                    logger.info(f"Retrieved {symbol} price from CoinGecko by symbol lookup: ${price}")
                    result[symbol] = price
                    self.cached_prices[symbol] = price
                    if not hasattr(self, 'cache_price_sources'):
                        self.cache_price_sources = {}
                    self.cache_price_sources[symbol] = "coingecko"
                    if symbol in remaining_symbols:
                        remaining_symbols.remove(symbol)
            except Exception as e:
                logger.warning(f"Error using CoinGecko client for individual token {symbol}: {e}")
                
        # Special case: Handle SOOMER token by token address lookup
        if "SOOMER" in remaining_symbols:
            logger.info("Special handling for SOOMER token in batch request")
            soomer_price = self.get_price_by_token_address("CTh5k7EHD2HBX64xZkeBDwmHskWvNq5WB8f4PWuW1hmz")
            if soomer_price is not None:
                result["SOOMER"] = soomer_price
                self.cached_prices["SOOMER"] = soomer_price
                self.cache_price_sources["SOOMER"] = "coingecko_address"
                logger.info(f"Retrieved SOOMER price by direct address lookup: ${soomer_price}")
                # Remove SOOMER from remaining symbols since we've handled it
                remaining_symbols.remove("SOOMER")
        
        # Only fallback to DeFi API for tokens we couldn't get from CoinGecko
        if remaining_symbols:
            logger.info(f"Falling back to DeFi API for {len(remaining_symbols)} tokens that CoinGecko couldn't provide")
            defi_api_prices = self.get_multiple_prices_from_defi_api(remaining_symbols)
            
            for symbol in remaining_symbols:
                if symbol in defi_api_prices:
                    # Skip SOL from DeFi API (known issue)
                    if symbol == "SOL" and defi_api_prices[symbol] > 1000:
                        logger.warning(f"Skipping unrealistic SOL price from DeFi API: ${defi_api_prices[symbol]}")
                    else:
                        result[symbol] = defi_api_prices[symbol]
                        self.cached_prices[symbol] = defi_api_prices[symbol]
                        self.cache_price_sources[symbol] = "defi_api"
                        logger.info(f"Retrieved {symbol} price from DeFi API fallback: ${defi_api_prices[symbol]}")
        
        # Save updated cache
        if self.use_cache:
            self._save_cache()
            
        logger.info(f"Retrieved prices for {len(result)}/{len(normalized_symbols)} tokens (prioritizing CoinGecko)")
        return result
    
    def add_token_mapping(self, symbol: str, coingecko_id: str) -> None:
        """
        Add a custom token mapping
        
        Args:
            symbol: Token symbol (e.g., "BONK")
            coingecko_id: CoinGecko ID (e.g., "bonk")
        """
        self.token_mapping[symbol.upper()] = coingecko_id.lower()
    
    def refresh_cache(self) -> None:
        """Force a refresh of the price cache"""
        # Get all known symbols
        symbols = list(self.token_mapping.keys())
        if not symbols:
            return
        
        # Fetch in batches to avoid rate limits
        batch_size = 25
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            self.get_multiple_prices(batch)
            
            # Sleep briefly to avoid rate limits
            if i + batch_size < len(symbols):
                time.sleep(1.2)  # CoinGecko rate limit is 50 calls per minute

# Singleton instance for use throughout the app
price_service = TokenPriceService()

def get_token_price(symbol: str, return_source: bool = False) -> Union[float, Tuple[float, str]]:
    """Convenience function to get a token price"""
    result = price_service.get_token_price(symbol, return_source)
    
    # Handle return_source=True case (tuple return)
    if return_source:
        # Result should be a tuple (price, source)
        if isinstance(result, tuple) and len(result) == 2:
            price, source = result
            # Ensure price is a float
            try:
                price_float = 0.0 if price is None else float(price)
                return (price_float, source)
            except (ValueError, TypeError):
                return (0.0, source)
        else:
            # Handle unexpected return format
            return (0.0, "none")
    
    # Handle return_source=False case (float return)
    else:
        # Result should be just the price
        try:
            return 0.0 if result is None else float(result)
        except (ValueError, TypeError):
            return 0.0

def get_multiple_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Convenience function to get multiple token prices with API key integration
    
    This ensures that all token price requests properly use the authenticated API
    with the CoinGecko API key from the environment.
    """
    # Log that we're using the API key
    coingecko_api_key = os.getenv("COINGECKO_API_KEY")
    if coingecko_api_key:
        logger.info(f"Using CoinGecko API key for multiple token price lookup")
        
    return price_service.get_multiple_prices(symbols)

def update_pool_with_token_prices(pool: dict) -> dict:
    """
    Update a pool dictionary with current token prices from CoinGecko
    
    The system prioritizes getting prices from CoinGecko for accuracy,
    and falls back to DeFi API only if needed.
    
    Args:
        pool: Dictionary containing pool data with token1_symbol and token2_symbol
        
    Returns:
        Updated pool dictionary with token prices and price sources
    """
    if not pool:
        return pool
        
    # Make a copy to avoid modifying the original
    updated_pool = pool.copy()
    
    # First check if we have token symbols from the tokens array
    tokens_array = pool.get("tokens", [])
    token1 = None
    token2 = None
    
    # Try to get token symbols from tokens array first
    if len(tokens_array) >= 2:
        # Extract symbols from tokens array
        token1 = tokens_array[0].get("symbol")
        token2 = tokens_array[1].get("symbol")
        
        # Log what we found
        logger.info(f"Found tokens in array: {token1} / {token2}")
    
    # Fallback to token1_symbol and token2_symbol fields if needed
    if not token1 and "token1_symbol" in pool:
        token1 = pool["token1_symbol"]
    
    if not token2 and "token2_symbol" in pool:
        token2 = pool["token2_symbol"]
    
    # Skip if we don't have token symbols
    if not token1 or not token2:
        logger.warning(f"Missing token symbols for pool {pool.get('id', 'unknown')}")
        return updated_pool
    
    # Normalize symbols for lookup (uppercase)
    token1 = token1.upper() if isinstance(token1, str) else token1
    token2 = token2.upper() if isinstance(token2, str) else token2
    
    # Get current prices with sources
    logger.info(f"Fetching prices for tokens: {token1} / {token2}")
    
    # Get token1 price with source
    token1_result = price_service.get_token_price(token1, return_source=True)
    if isinstance(token1_result, tuple) and len(token1_result) == 2:
        token1_price = float(token1_result[0]) if token1_result[0] is not None else 0.0
        token1_source = token1_result[1]
    else:
        # Handle case where result is not a tuple
        token1_price = float(token1_result) if token1_result is not None else 0.0
        token1_source = "none"
    
    # Get token2 price with source
    token2_result = price_service.get_token_price(token2, return_source=True)
    if isinstance(token2_result, tuple) and len(token2_result) == 2:
        token2_price = float(token2_result[0]) if token2_result[0] is not None else 0.0
        token2_source = token2_result[1]
    else:
        # Handle case where result is not a tuple
        token2_price = float(token2_result) if token2_result is not None else 0.0
        token2_source = "none"
    
    # Special case: If token is SOL and price looks unrealistic, try to fetch from a specific source
    if token1.upper() == "SOL" and token1_price > 1000:
        try:
            # Try to get a corrected SOL price from CoinGecko directly
            from coingecko_api import CoinGeckoAPI
            coingecko_client = CoinGeckoAPI()
            corrected_price = coingecko_client.get_price(["solana"], "usd")
            if "solana" in corrected_price and "usd" in corrected_price["solana"]:
                token1_price = corrected_price["solana"]["usd"]
                token1_source = "coingecko_corrected"
                logger.info(f"Corrected SOL price: ${token1_price}")
        except Exception as e:
            logger.error(f"Error fetching corrected SOL price: {e}")
            
    if token2.upper() == "SOL" and token2_price > 1000:
        try:
            # Try to get a corrected SOL price from CoinGecko directly
            from coingecko_api import CoinGeckoAPI
            coingecko_client = CoinGeckoAPI()
            corrected_price = coingecko_client.get_price(["solana"], "usd")
            if "solana" in corrected_price and "usd" in corrected_price["solana"]:
                token2_price = corrected_price["solana"]["usd"]
                token2_source = "coingecko_corrected"
                logger.info(f"Corrected SOL price: ${token2_price}")
        except Exception as e:
            logger.error(f"Error fetching corrected SOL price: {e}")
    
    # Update the pool with prices
    updated_pool["token1_price"] = token1_price
    updated_pool["token2_price"] = token2_price
    updated_pool["token1_price_source"] = token1_source
    updated_pool["token2_price_source"] = token2_source
    
    # Calculate USD values if available
    try:
        # Try to get token amounts from various possible field names
        token1_amount = None
        token2_amount = None
        
        # First check the tokens array structure
        if len(tokens_array) >= 2:
            token1_amount = tokens_array[0].get("amount")
            token2_amount = tokens_array[1].get("amount")
        
        # Fallback to direct fields
        if token1_amount is None and "token1_amount" in pool:
            token1_amount = pool["token1_amount"]
            
        if token2_amount is None and "token2_amount" in pool:
            token2_amount = pool["token2_amount"]
            
        # Calculate USD values if we have amounts
        if token1_amount is not None and token1_price is not None:
            token1_usd = float(token1_amount) * token1_price
            updated_pool["token1_usd_value"] = token1_usd
            
        if token2_amount is not None and token2_price is not None:
            token2_usd = float(token2_amount) * token2_price
            updated_pool["token2_usd_value"] = token2_usd
            
        # Calculate total liquidity in USD
        token1_usd = updated_pool.get("token1_usd_value", 0)
        token2_usd = updated_pool.get("token2_usd_value", 0)
        
        if token1_usd > 0 or token2_usd > 0:
            updated_pool["liquidity_usd"] = token1_usd + token2_usd
            updated_pool["liquidity_updated_at"] = int(time.time())
            
    except Exception as e:
        logger.error(f"Error calculating USD values for pool {pool.get('id', 'unknown')}: {e}")
    
    return updated_pool