"""
Token Price Service for SolPool Insight
Fetches token prices from CoinGecko API
"""

import os
import time
import logging
import requests
import json
from typing import Dict, Any, List, Optional
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
    "ADA": "cardano",
    "DAI": "dai",
    "BUSD": "binance-usd",
    "USDH": "usdh",
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
                    last_update_str = cache_data.get("last_update")
                    if last_update_str:
                        self.last_cache_update = datetime.fromisoformat(last_update_str)
                logger.info(f"Loaded {len(self.cached_prices)} token prices from cache")
        except Exception as e:
            logger.error(f"Error loading price cache: {e}")
            self.cached_prices = {}
            self.last_cache_update = datetime.min
    
    def _save_cache(self) -> None:
        """Save current prices to cache file"""
        try:
            cache_data = {
                "prices": self.cached_prices,
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
    
    def get_token_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a token
        
        Args:
            symbol: Token symbol (e.g., "SOL", "BTC")
            
        Returns:
            Current price in USD, or None if not available
        """
        # Try cache first
        if self._is_cache_valid() and symbol in self.cached_prices:
            return self.cached_prices[symbol]
        
        # Convert symbol to CoinGecko ID
        coingecko_id = self.token_mapping.get(symbol.upper())
        if not coingecko_id:
            logger.warning(f"No mapping for token: {symbol}")
            return None
        
        try:
            # Fetch price from CoinGecko
            url = f"{COINGECKO_API_URL}/simple/price"
            params = {
                "ids": coingecko_id,
                "vs_currencies": "usd"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if coingecko_id in data and "usd" in data[coingecko_id]:
                price = data[coingecko_id]["usd"]
                
                # Update cache
                self.cached_prices[symbol.upper()] = price
                if self.use_cache:
                    self._save_cache()
                
                logger.info(f"Retrieved {symbol} price: ${price}")
                return price
            else:
                logger.warning(f"Price data not found for {symbol} (CoinGecko ID: {coingecko_id})")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get prices for multiple tokens at once
        
        Args:
            symbols: List of token symbols
            
        Returns:
            Dictionary mapping symbols to prices
        """
        # Check which symbols we need to fetch
        symbols_to_fetch = []
        result = {}
        
        for symbol in symbols:
            symbol = symbol.upper()
            # Use cache if valid
            if self._is_cache_valid() and symbol in self.cached_prices:
                result[symbol] = self.cached_prices[symbol]
            elif symbol in self.token_mapping:
                symbols_to_fetch.append(symbol)
            else:
                logger.warning(f"No mapping for token: {symbol}")
        
        if not symbols_to_fetch:
            return result
        
        # Convert symbols to CoinGecko IDs
        coingecko_ids = [self.token_mapping[s] for s in symbols_to_fetch if s in self.token_mapping]
        
        if not coingecko_ids:
            return result
        
        try:
            # Fetch prices from CoinGecko
            url = f"{COINGECKO_API_URL}/simple/price"
            params = {
                "ids": ",".join(coingecko_ids),
                "vs_currencies": "usd"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process results and update cache
            for symbol in symbols_to_fetch:
                coingecko_id = self.token_mapping.get(symbol)
                if coingecko_id in data and "usd" in data[coingecko_id]:
                    price = data[coingecko_id]["usd"]
                    result[symbol] = price
                    self.cached_prices[symbol] = price
            
            # Save updated cache
            if self.use_cache:
                self._save_cache()
                
            logger.info(f"Retrieved prices for {len(result)} tokens")
            return result
                
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
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

def get_token_price(symbol: str) -> float:
    """Convenience function to get a token price"""
    price = price_service.get_token_price(symbol)
    return price if price is not None else 0.0

def get_multiple_prices(symbols: List[str]) -> Dict[str, float]:
    """Convenience function to get multiple token prices"""
    return price_service.get_multiple_prices(symbols)

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