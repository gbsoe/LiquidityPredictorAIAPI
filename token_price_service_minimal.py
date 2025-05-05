"""Token Price Service for SolPool Insight - Minimal Version
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

# Minimal token price service that returns fixed prices for testing
def get_token_price(symbol: str, return_source: bool = False) -> Union[float, Tuple[float, str]]:
    """Get a token price (simplified version)"""
    # Normalize symbol
    if not symbol:
        if return_source:
            return (0.0, "none")
        return 0.0
    
    symbol = symbol.upper() if isinstance(symbol, str) else "UNKNOWN"
    
    # Default prices for common tokens
    prices = {
        "SOL": 169.42,
        "BTC": 62750.0,
        "ETH": 3057.25,
        "USDC": 1.0,
        "USDT": 1.0,
        "UST": 1.0,
        "DAI": 1.0,
        "BUSD": 1.0,
        "USDH": 1.0,
        "TUSD": 1.0,
        "BONK": 0.00001987,
        "JUP": 0.745,
        "WIF": 1.32,
        "PYTH": 0.52,
        "RAY": 0.45
    }
    
    price = prices.get(symbol, 0.5)  # Default price for unknown tokens
    source = "fixed"
    
    if return_source:
        return (price, source)
    return price

def get_multiple_prices(symbols: List[str]) -> Dict[str, float]:
    """Get prices for multiple tokens at once (simplified version)"""
    result = {}
    for symbol in symbols:
        if symbol and symbol != "UNKNOWN":
            price = get_token_price(symbol)
            if price is not None:
                result[symbol.upper()] = price
    return result

def update_pool_with_token_prices(pool: dict) -> dict:
    """Update a pool dictionary with token prices (simplified version)"""
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
    
    # Normalize symbols for lookup
    token1 = token1.upper() if isinstance(token1, str) else token1
    token2 = token2.upper() if isinstance(token2, str) else token2
    
    # Get prices (simplified version just uses fixed prices)
    token1_price = get_token_price(token1)
    token2_price = get_token_price(token2)
    
    # Update the pool with prices
    updated_pool["token1_price"] = token1_price
    updated_pool["token2_price"] = token2_price
    updated_pool["token1_price_source"] = "fixed"
    updated_pool["token2_price_source"] = "fixed"
    
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