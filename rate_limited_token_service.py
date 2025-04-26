"""
Rate-Limited Token Price Service

This module extends the existing token price service with enhanced rate limiting
and fallback mechanisms to deal with API restrictions from services like Helix.
"""

import os
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Import the original service
from token_price_service import TokenPriceService, price_service, get_token_price, get_multiple_prices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rate_limited_token_service")

# Constants
RATE_LIMIT_REQUESTS = 30  # Max requests per minute (conservative)
RATE_LIMIT_WINDOW = 60    # Window in seconds (1 minute)
DEFAULT_BACKOFF = 2       # Default backoff time in seconds
MAX_RETRIES = 3           # Maximum number of retries for a request

class RateLimitedTokenService:
    """Token price service with enhanced rate limiting and fallback mechanisms"""
    
    def __init__(self):
        """Initialize the rate-limited service"""
        self.base_service = price_service
        self.request_timestamps = []
        self.backoff_time = DEFAULT_BACKOFF
        self.retry_counts = {}
    
    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits
        
        Returns:
            True if we can make a request, False if we should wait
        """
        now = time.time()
        
        # Remove timestamps older than our window
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if now - ts < RATE_LIMIT_WINDOW]
        
        # Check if we've made too many requests
        if len(self.request_timestamps) >= RATE_LIMIT_REQUESTS:
            return False
        
        return True
    
    def _wait_for_rate_limit(self) -> None:
        """Wait until we're within rate limits"""
        while not self._check_rate_limit():
            logger.info(f"Rate limit reached, waiting {self.backoff_time}s...")
            time.sleep(self.backoff_time)
            
            # Increase backoff for next time (up to 10 seconds)
            self.backoff_time = min(10, self.backoff_time * 1.5)
    
    def _record_request(self) -> None:
        """Record that we've made a request"""
        self.request_timestamps.append(time.time())
        
        # Reset backoff if we're not being rate limited much
        if len(self.request_timestamps) < RATE_LIMIT_REQUESTS / 2:
            self.backoff_time = DEFAULT_BACKOFF
    
    def get_token_price(self, symbol: str) -> float:
        """
        Get token price with rate limiting
        
        Args:
            symbol: Token symbol (e.g., "SOL")
            
        Returns:
            Current price or 0.0 if not available
        """
        # Check if we already have this in cache
        if (self.base_service._is_cache_valid() and 
            symbol.upper() in self.base_service.cached_prices):
            return self.base_service.cached_prices[symbol.upper()]
        
        # Initialize retry counter for this symbol
        request_key = f"price_{symbol}"
        if request_key not in self.retry_counts:
            self.retry_counts[request_key] = 0
            
        # Check if we've exceeded retry limit
        if self.retry_counts[request_key] >= MAX_RETRIES:
            logger.warning(f"Max retries exceeded for {symbol}, using fallback")
            return 0.0
            
        # Wait for rate limit if needed
        self._wait_for_rate_limit()
        
        try:
            # Make the request
            self._record_request()
            price = self.base_service.get_token_price(symbol)
            
            # Reset retry counter on success
            self.retry_counts[request_key] = 0
            
            return price if price is not None else 0.0
            
        except Exception as e:
            # Increment retry counter
            self.retry_counts[request_key] += 1
            
            logger.error(f"Error getting price for {symbol} (attempt {self.retry_counts[request_key]}): {e}")
            
            # Back off for a bit
            time.sleep(self.backoff_time)
            
            # Retry if we haven't exceeded limit
            if self.retry_counts[request_key] < MAX_RETRIES:
                return self.get_token_price(symbol)
            else:
                return 0.0
    
    def get_multiple_prices(self, symbols: List[str], batch_size: int = 5) -> Dict[str, float]:
        """
        Get multiple token prices with rate limiting and batching
        
        Args:
            symbols: List of token symbols
            batch_size: Size of batches to request (default: 5)
            
        Returns:
            Dictionary of token prices
        """
        # Start with what we have in cache
        result = {}
        symbols_to_fetch = []
        
        # Check which symbols we need to fetch vs which are in cache
        for symbol in symbols:
            upper_symbol = symbol.upper()
            if (self.base_service._is_cache_valid() and 
                upper_symbol in self.base_service.cached_prices):
                result[upper_symbol] = self.base_service.cached_prices[upper_symbol]
            else:
                symbols_to_fetch.append(upper_symbol)
        
        # Process in batches to avoid rate limits
        for i in range(0, len(symbols_to_fetch), batch_size):
            batch = symbols_to_fetch[i:i + batch_size]
            
            # Wait for rate limit
            self._wait_for_rate_limit()
            
            try:
                # Make the request
                self._record_request()
                batch_results = self.base_service.get_multiple_prices(batch)
                
                # Add results to our dictionary
                result.update(batch_results)
                
                # Wait a bit between batches
                if i + batch_size < len(symbols_to_fetch):
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error getting prices for batch {i//batch_size + 1}: {e}")
                # Still continue with next batch
        
        return result
    
    def update_pool_with_token_prices(self, pool: dict) -> dict:
        """
        Update a pool with token prices using rate limiting
        
        Args:
            pool: Pool data dictionary
            
        Returns:
            Updated pool dictionary with token prices
        """
        if not pool or "token1_symbol" not in pool or "token2_symbol" not in pool:
            return pool
            
        # Make a copy to avoid modifying the original
        updated_pool = pool.copy()
        
        # Get token symbols
        token1 = pool["token1_symbol"]
        token2 = pool["token2_symbol"]
        
        # Get current prices with rate limiting
        prices = self.get_multiple_prices([token1, token2])
        
        # Add prices to pool data
        updated_pool["token1_price"] = prices.get(token1.upper(), 0.0)
        updated_pool["token2_price"] = prices.get(token2.upper(), 0.0)
            
        return updated_pool

# Create a singleton instance
rate_limited_service = RateLimitedTokenService()

# Convenience functions
def get_price_with_rate_limit(symbol: str) -> float:
    """Get a token price with rate limiting"""
    return rate_limited_service.get_token_price(symbol)

def get_prices_with_rate_limit(symbols: List[str]) -> Dict[str, float]:
    """Get multiple token prices with rate limiting"""
    return rate_limited_service.get_multiple_prices(symbols)

def update_pool_prices_with_rate_limit(pool: dict) -> dict:
    """Update a pool with token prices using rate limiting"""
    return rate_limited_service.update_pool_with_token_prices(pool)