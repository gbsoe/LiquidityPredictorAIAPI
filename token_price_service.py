"""
Token Price Service for SolPool Insight
Fetches token prices from CoinGecko API
"""

import os
import time
import logging
import requests
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("token_price_service")

# Constants
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
CACHE_FILE = "token_price_cache.json"
CACHE_EXPIRY_SECONDS = 3600  # 1 hour

# Default Solana token mapping - CoinGecko uses different IDs
DEFAULT_TOKEN_MAPPING = {
    # Standard tokens
    "SOL": "solana",
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "USDC": "usd-coin",
    "USDT": "tether",
    "JUP": "jupiter",
    "RAY": "raydium", 
    "ORCA": "orca",
    "BONK": "bonk",
    "SAMO": "samoyedcoin",
    "WIF": "dogwifhat",
    "PYTH": "pyth-network",
    "MNGO": "mango-markets",
    "SRM": "serum",
    "SABER": "saber",
    "ATLAS": "star-atlas",
    "POLIS": "star-atlas-dao",
    "GARI": "gari-network",
    "COPE": "cope",
    "SLND": "solend",
    "AVAX": "avalanche-2",
    "BNB": "binancecoin",
    # Special staked SOL tokens
    "MSOL": "marinade-staked-sol",
    "mSOL": "marinade-staked-sol",
    "STSOL": "lido-staked-sol",
    "stSOL": "lido-staked-sol",
    "ADA": "cardano",
    "DAI": "dai",
    "BUSD": "binance-usd",
    "USDH": "usdh",
    
    # Tokens from the screenshot that need specific mappings
    "SOOMER": "soomer", # Updated to match CoinGecko ID 
    "SOGENT": "sogent",
    "SPEC": "spectrecoin",
    
    # API-specific symbols (important: these match actual token symbols in the API!)
    "So11": "solana",        # Solana
    "Es9v": "usd-coin",      # USDC variant
    "9n4n": "bitcoin",       # BTC variant
    "4k3D": "raydium",       # RAY
    "EPjF": "usd-coin",      # USDC
    "DezX": "ethereum",      # ETH
    "7vfC": "ethereum",      # ETH variant
    
    # Common abbreviated token addresses
    "So11111111111111111111111111111111111111112": "solana",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "usd-coin",
    "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R": "raydium",
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": "marinade-staked-sol",
    "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E": "bitcoin",
    "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs": "ethereum",
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "ethereum",
    "CTh5k7EHD2HBX64xZkeBDwmHskWvNq5WB8f4PWuW1hmz": "soomer", # SOOMER token address
}

class TokenPriceService:
    """Service for fetching and caching token prices"""
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the token price service
        
        Args:
            use_cache: Whether to use the cache (default: True)
        """
        self.use_cache = use_cache
        self.token_mapping = DEFAULT_TOKEN_MAPPING.copy()
        self.cached_prices = {}
        self.cache_price_sources = {}  # Track the source of each price (defi_api, coingecko)
        self.last_cache_update = datetime.min
        
        # Load cache if it exists
        if self.use_cache:
            self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cached prices from file if available"""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    self.cached_prices = cache_data.get("prices", {})
                    self.cache_price_sources = cache_data.get("price_sources", {})
                    last_update_str = cache_data.get("last_update")
                    if last_update_str:
                        self.last_cache_update = datetime.fromisoformat(last_update_str)
                logger.info(f"Loaded {len(self.cached_prices)} token prices from cache")
        except Exception as e:
            logger.error(f"Error loading price cache: {e}")
            self.cached_prices = {}
            self.cache_price_sources = {}
            self.last_cache_update = datetime.min
    
    def _save_cache(self) -> None:
        """Save current prices to cache file"""
        try:
            cache_data = {
                "prices": self.cached_prices,
                "price_sources": self.cache_price_sources,
                "last_update": datetime.now().isoformat()
            }
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved {len(self.cached_prices)} token prices to cache")
        except Exception as e:
            logger.error(f"Error saving price cache: {e}")
    
    def _is_cache_valid(self) -> bool:
        """Check if the cache is still valid"""
        if not self.use_cache:
            return False
        
        cache_age = datetime.now() - self.last_cache_update
        return cache_age.total_seconds() < CACHE_EXPIRY_SECONDS
    
    def get_token_price_from_defi_api(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a token directly from the DeFi API
        
        Args:
            symbol: Token symbol (e.g., "SOL", "BTC")
            
        Returns:
            Current price in USD, or None if not available
        """
        # Skip processing for UNKNOWN tokens to avoid spam warnings
        if symbol == "UNKNOWN":
            return None
            
        try:
            # Get the API key from environment
            api_key = os.getenv("DEFI_API_KEY")
            if not api_key:
                logger.warning("No API key found for DeFi API")
                return None
                
            # Fetch price from DeFi API using the GET Token endpoint
            url = f"https://filotdefiapi.replit.app/api/v1/tokens/{symbol}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 429:
                logger.warning(f"Rate limit exceeded for DeFi API when fetching {symbol}")
                time.sleep(1)  # Back off for a second
                return None
                
            response.raise_for_status()
            data = response.json()
            
            # The API returns an array of tokens where one might match our symbol
            if isinstance(data, list) and len(data) > 0:
                # Find the token that matches our symbol (case-insensitive)
                for token in data:
                    if token.get("symbol", "").upper() == symbol.upper():
                        price = token.get("price", 0)
                        
                        # Update cache
                        self.cached_prices[symbol.upper()] = price
                        # Set the price source
                        self.cache_price_sources[symbol.upper()] = "defi_api"
                        
                        if self.use_cache:
                            self._save_cache()
                        
                        logger.info(f"Retrieved {symbol} price from DeFi API: ${price}")
                        return price
                        
                logger.warning(f"Token {symbol} not found in DeFi API response")
                return None
            else:
                logger.warning(f"Invalid or empty response for token {symbol} from DeFi API")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching price for {symbol} from DeFi API: {e}")
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
        # Skip processing for UNKNOWN tokens to avoid spam warnings
        if symbol == "UNKNOWN":
            return (None, "none") if return_source else None
            
        # Try cache first if it's valid and the source is CoinGecko
        if self._is_cache_valid() and symbol in self.cached_prices:
            cache_source = self.cache_price_sources.get(symbol.upper(), "none") if hasattr(self, 'cache_price_sources') else "none"
            # Only use cached values from CoinGecko or for stablecoins
            if (cache_source == "coingecko" or 
                (symbol.upper() in ["USDC", "USDT", "DAI"] and 
                 self.cached_prices[symbol.upper()] == 1.0)):
                logger.debug(f"Using cached price for {symbol} from {cache_source}: {self.cached_prices[symbol.upper()]}")
                return (self.cached_prices[symbol.upper()], cache_source) if return_source else self.cached_prices[symbol.upper()]
        
        # For all tokens, always try CoinGecko first 
        logger.info(f"Using CoinGecko as primary source for {symbol} price")
        price = None
        
        # Convert symbol to CoinGecko ID
        coingecko_id = self.token_mapping.get(symbol.upper())
        if coingecko_id:
            # Try CoinGecko API by ID
            try:
                # Use our improved CoinGeckoAPI client
                from coingecko_api import CoinGeckoAPI
                coingecko_client = CoinGeckoAPI()
                price_data = coingecko_client.get_price([coingecko_id], "usd")
                if coingecko_id in price_data and "usd" in price_data[coingecko_id]:
                    price = price_data[coingecko_id]["usd"]
                    logger.info(f"Retrieved {symbol} price from CoinGecko by ID: ${price}")
                else:
                    logger.warning(f"No price data found for {symbol} in CoinGecko response")
            except Exception as e:
                logger.warning(f"Error using CoinGecko client by ID: {e}")
        
        # If no price by ID, try by symbol
        if price is None:
            try:
                # Use our CoinGeckoAPI client
                from coingecko_api import CoinGeckoAPI
                coingecko_client = CoinGeckoAPI()
                price = coingecko_client.get_token_price_by_symbol(symbol)
                if price is not None and price > 0:
                    logger.info(f"Retrieved {symbol} price from CoinGecko by symbol: ${price}")
            except Exception as e:
                logger.warning(f"Error using CoinGecko client by symbol: {e}")
        
        # If we got a price from CoinGecko, cache and return it
        if price is not None and price > 0:
            # Validate price (especially for SOL to catch errors)
            if symbol.upper() == "SOL" and price > 1000:
                logger.warning(f"Detected potentially incorrect SOL price from CoinGecko: ${price}")
                price = None
            else:
                # Cache the result
                self.cached_prices[symbol.upper()] = price
                if not hasattr(self, 'cache_price_sources'):
                    self.cache_price_sources = {}
                self.cache_price_sources[symbol.upper()] = "coingecko"
                if self.use_cache:
                    self._save_cache()
                return (price, "coingecko") if return_source else price
        
        # Only fallback to DeFi API if CoinGecko fails
        price = self.get_token_price_from_defi_api(symbol)
        if price is not None:
            # Check for obviously wrong SOL price (over $1000)
            if symbol.upper() == "SOL" and price > 1000:
                logger.warning(f"Detected potentially incorrect SOL price from DeFi API: ${price}")
                return (None, "none") if return_source else None
            else:
                # Cache the result
                self.cached_prices[symbol.upper()] = price
                if not hasattr(self, 'cache_price_sources'):
                    self.cache_price_sources = {}
                self.cache_price_sources[symbol.upper()] = "defi_api"
                if self.use_cache:
                    self._save_cache()
                return (price, "defi_api") if return_source else price
                
        # If neither source worked, attempt CoinGecko by address as last resort
        # Special case for tokens with known addresses
        coingecko_id = self.token_mapping.get(symbol.upper())
        if not coingecko_id:
            # Only log warnings for non-UNKNOWN tokens to reduce noise
            logger.warning(f"No mapping for token: {symbol}")
            
            # Special case for SOOMER - try to get price by address
            if symbol.upper() == "SOOMER":
                # Try direct address lookup for SOOMER which has address CTh5k7EHD2HBX64xZkeBDwmHskWvNq5WB8f4PWuW1hmz
                logger.info(f"Attempting direct address lookup for SOOMER token")
                soomer_price = self.get_price_by_token_address("CTh5k7EHD2HBX64xZkeBDwmHskWvNq5WB8f4PWuW1hmz")
                if soomer_price is not None:
                    # Update cache
                    self.cached_prices[symbol.upper()] = soomer_price
                    if not hasattr(self, 'cache_price_sources'):
                        self.cache_price_sources = {}
                    self.cache_price_sources[symbol.upper()] = "coingecko"
                    
                    if self.use_cache:
                        self._save_cache()
                    
                    logger.info(f"Retrieved SOOMER price by address: ${soomer_price}")
                    return (soomer_price, "coingecko") if return_source else soomer_price
            
            return (None, "none") if return_source else None
        
        try:
            # Import the CoinGeckoAPI client here to avoid circular imports
            try:
                from coingecko_api import CoinGeckoAPI
                # Create a client instance if needed
                coingecko_api = CoinGeckoAPI()
                
                # Use the client to get price data
                logger.info(f"Using CoinGecko API client to fetch {coingecko_id} price")
                price_data = coingecko_api.get_price([coingecko_id], "usd")
                
                if coingecko_id in price_data and "usd" in price_data[coingecko_id]:
                    price = price_data[coingecko_id]["usd"]
                    logger.info(f"Retrieved {symbol} price from CoinGecko client: ${price}")
                    
                    # Update cache
                    self.cached_prices[symbol.upper()] = price
                    # Store the price source
                    if not hasattr(self, 'cache_price_sources'):
                        self.cache_price_sources = {}
                    self.cache_price_sources[symbol.upper()] = "coingecko"
                    
                    if self.use_cache:
                        self._save_cache()
                    
                    return price
                else:
                    logger.warning(f"Price data not found for {symbol} (CoinGecko ID: {coingecko_id}) in client response")
                    # Fall back to direct API call if client fails
                    
            except (ImportError, Exception) as e:
                logger.warning(f"Could not use CoinGecko API client: {e}. Falling back to direct API call.")
            
            # Fallback: Direct API call
            url = f"{COINGECKO_API_URL}/simple/price"
            params = {
                "ids": coingecko_id,
                "vs_currencies": "usd"
            }
            
            # Check if we have a CoinGecko API key
            coingecko_api_key = os.getenv("COINGECKO_API_KEY")
            headers = {}
            if coingecko_api_key:
                # Check if it's a Demo API key (starts with CG-) or Pro API key
                if coingecko_api_key.startswith("CG-"):
                    # For Demo API keys
                    headers["x-cg-demo-api-key"] = coingecko_api_key
                    # Also add as query parameter for some endpoints
                    params["x_cg_demo_api_key"] = coingecko_api_key
                else:
                    # For Pro API keys (maintain backward compatibility)
                    headers["x-cg-pro-api-key"] = coingecko_api_key
                    headers["x-cg-api-key"] = coingecko_api_key
                logger.info(f"Using CoinGecko API key for {symbol}")
            
            logger.info(f"Making direct CoinGecko API call for {coingecko_id}")
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if coingecko_id in data and "usd" in data[coingecko_id]:
                price = data[coingecko_id]["usd"]
                
                # Update cache
                self.cached_prices[symbol.upper()] = price
                # Store the price source in another cache dictionary
                if not hasattr(self, 'cache_price_sources'):
                    self.cache_price_sources = {}
                self.cache_price_sources[symbol.upper()] = "coingecko"
                
                if self.use_cache:
                    self._save_cache()
                
                logger.info(f"Retrieved {symbol} price from CoinGecko: ${price}")
                return (price, "coingecko") if return_source else price
            else:
                logger.warning(f"Price data not found for {symbol} (CoinGecko ID: {coingecko_id})")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching price for {symbol} from CoinGecko: {e}")
            return None
            
    def get_price_by_token_address(self, token_address: str) -> Optional[float]:
        """
        Get the price of a token using its address on the Solana blockchain
        
        Args:
            token_address: Solana token address
            
        Returns:
            Current price in USD, or None if not available
        """
        try:
            # Check if we have a CoinGecko API key
            coingecko_api_key = os.getenv("COINGECKO_API_KEY")
            if not coingecko_api_key:
                logger.warning("No CoinGecko API key found for token address lookup")
                return None
                
            # Use the CoinGecko API to get the price by address
            url = f"{COINGECKO_API_URL}/coins/solana/contract/{token_address}"
            headers = {}
            params = {}
            
            # Check if it's a Demo API key (starts with CG-) or Pro API key
            if coingecko_api_key.startswith("CG-"):
                # For Demo API keys
                headers["x-cg-demo-api-key"] = coingecko_api_key
                # Also add as query parameter
                params["x_cg_demo_api_key"] = coingecko_api_key
            else:
                # For Pro API keys (maintain backward compatibility)
                headers["x-cg-pro-api-key"] = coingecko_api_key
                headers["x-cg-api-key"] = coingecko_api_key
            
            logger.info(f"Looking up token price by address: {token_address}")
            response = requests.get(url, headers=headers, timeout=10)
            
            # Handle 404 errors (token not found)
            if response.status_code == 404:
                logger.warning(f"Token with address {token_address} not found on CoinGecko")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            # Extract the price
            if "market_data" in data and "current_price" in data["market_data"] and "usd" in data["market_data"]["current_price"]:
                price = data["market_data"]["current_price"]["usd"]
                logger.info(f"Retrieved price for token address {token_address}: ${price}")
                
                # If we have the token symbol in the response, update our mapping
                if "symbol" in data:
                    symbol = data["symbol"].upper()
                    if "id" in data:
                        self.token_mapping[symbol] = data["id"]
                        logger.info(f"Added token mapping: {symbol} -> {data['id']}")
                        
                    # Also add address mapping
                    self.token_mapping[token_address] = data["id"]
                    logger.info(f"Added address mapping: {token_address} -> {data['id']}")
                    
                return price
            else:
                logger.warning(f"Price data not found for token address {token_address}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching price for token address {token_address}: {e}")
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
        
        # Initial default values for stablecoin prices (1.0) to display while loading
        default_prices = {
            "USDC": 1.00,
            "USDT": 1.00,
            "UXD": 1.00,
            "USDH": 1.00,
            "BUSD": 1.00,
            "DAI": 1.00,
        }
        
        # Uppercase all input symbols for comparison
        normalized_symbols = [s.upper() for s in symbols if s != "UNKNOWN"]
        
        # Set default prices for stablecoins only
        for symbol in normalized_symbols:
            if symbol in default_prices:
                result[symbol] = default_prices[symbol]
                self.cached_prices[symbol] = default_prices[symbol]
                self.cache_price_sources[symbol] = "defi_api"  # These are treated as coming from the API
                logger.info(f"Set default stablecoin price for {symbol}: ${default_prices[symbol]}")
        
        # If we have all prices, return early
        if len(result) == len(normalized_symbols):
            return result
            
        # Otherwise, try the API too
        # Get the API key
        api_key = os.getenv("DEFI_API_KEY")
        if not api_key:
            logger.warning("No API key found for DeFi API")
            return result
            
        try:
            # First try a general token lookup
            url = "https://filotdefiapi.replit.app/api/v1/tokens"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 429:
                logger.warning("Rate limit exceeded for DeFi API when fetching multiple tokens")
                time.sleep(1)  # Back off for a second
                return result
                
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                # Process all tokens in the list
                for token in data:
                    token_symbol = token.get("symbol", "").upper()
                    if token_symbol in normalized_symbols and token_symbol not in result:
                        # If API returns 0 price but we have a default price for stablecoins, use it
                        price = token.get("price", 0)
                        if price == 0 and token_symbol in default_prices:
                            price = default_prices[token_symbol]
                        
                        # Update result and cache
                        result[token_symbol] = price
                        self.cached_prices[token_symbol] = price
                        self.cache_price_sources[token_symbol] = "defi_api"
                
                # Save updated cache
                if self.use_cache:
                    self._save_cache()
                    
                logger.info(f"Retrieved {len(result)} token prices from DeFi API")
            else:
                logger.warning("Invalid or empty response from DeFi API")
                
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
        # Check which symbols we need to fetch
        symbols_to_fetch = []
        result = {}
        
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
                price = coingecko_client.get_token_price_by_symbol(symbol)
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
    
    # Check for obviously incorrect SOL price (over $1000 is clearly wrong)
    if token1.upper() == "SOL" and token1_price > 1000:
        logger.warning(f"Detected incorrect SOL price for token1: ${token1_price}. Attempting to fix.")
        # Try to get a correct price from CoinGecko
        try:
            from coingecko_api import CoinGeckoAPI
            coingecko_client = CoinGeckoAPI()
            price_data = coingecko_client.get_price(["solana"], "usd")
            if "solana" in price_data and "usd" in price_data["solana"]:
                token1_price = price_data["solana"]["usd"]
                token1_source = "coingecko"
                logger.info(f"Retrieved corrected SOL price from CoinGecko: ${token1_price}")
        except Exception as e:
            logger.error(f"Error fetching corrected SOL price: {e}")
            
    if token2.upper() == "SOL" and token2_price > 1000:
        logger.warning(f"Detected incorrect SOL price for token2: ${token2_price}. Attempting to fix.")
        # Try to get a correct price from CoinGecko
        try:
            from coingecko_api import CoinGeckoAPI
            coingecko_client = CoinGeckoAPI()
            price_data = coingecko_client.get_price(["solana"], "usd")
            if "solana" in price_data and "usd" in price_data["solana"]:
                token2_price = price_data["solana"]["usd"]
                token2_source = "coingecko"
                logger.info(f"Retrieved corrected SOL price from CoinGecko: ${token2_price}")
        except Exception as e:
            logger.error(f"Error fetching corrected SOL price: {e}")
    
    # Add prices and sources to pool data
    updated_pool["token1_price"] = token1_price
    updated_pool["token1_price_source"] = token1_source
    logger.info(f"Set token1 price for {token1}: {token1_price} (source: {token1_source})")
    
    updated_pool["token2_price"] = token2_price
    updated_pool["token2_price_source"] = token2_source
    logger.info(f"Set token2 price for {token2}: {token2_price} (source: {token2_source})")
    
    # Add these token symbols back to ensure they're preserved
    updated_pool["token1_symbol"] = token1
    updated_pool["token2_symbol"] = token2
        
    return updated_pool

# Example usage
if __name__ == "__main__":
    # Test the price service
    service = TokenPriceService(use_cache=True)
    
    # Test individual price
    sol_price = service.get_token_price("SOL")
    print(f"SOL price: ${sol_price}")
    
    # Test multiple prices
    prices = service.get_multiple_prices(["SOL", "BTC", "ETH", "BONK", "JUP"])
    for symbol, price in prices.items():
        print(f"{symbol} price: ${price}")