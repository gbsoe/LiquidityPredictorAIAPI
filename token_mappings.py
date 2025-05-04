"""
Token Mappings for CoinGecko Integration

This module provides explicit mappings between token symbols and their
corresponding CoinGecko IDs to ensure accurate price retrieval.
"""

import os
import json
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default mappings for common tokens
DEFAULT_TOKEN_MAPPINGS = {
    # Major tokens
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "USDC": "usd-coin",
    "USDT": "tether",
    
    # Staked tokens
    "MSOL": "marinade-staked-sol",
    "STSOL": "lido-staked-sol",
    
    # Solana ecosystem tokens
    "BONK": "bonk",
    "RAY": "raydium",
    "ORCA": "orca",
    "ATLAS": "star-atlas",
    "POLIS": "star-atlas-dao",
    "SAMO": "samoyedcoin",
    "JUP": "jupiter",
    "BOOP": "boop-2",
    
    # Add mappings for tokens shown in the screenshot
    "SOGENT": "sogent", 
    "SOOMER": "soomer",
    "SPEC": "spec-finance",
    
    # Other common tokens
    "FIDA": "bonfida",
    "MNGO": "mango-markets",
    "SRM": "serum",
    "SLND": "solend",
    "LDO": "lido-dao",
    "RNDR": "render-token",
    "UXD": "uxd-stablecoin",
    "COPE": "cope",
    "FARTCOIN": "fartcoin",
    "POOP": "poochain-powering-poo-fun"
}

def load_token_mappings() -> Dict[str, str]:
    """
    Load token mappings from configuration or use defaults.
    
    Returns:
        Dictionary mapping token symbols to CoinGecko IDs
    """
    mappings = DEFAULT_TOKEN_MAPPINGS.copy()
    
    # Try to load custom mappings from a JSON file if it exists
    mappings_file = os.path.join("config", "token_mappings.json")
    if os.path.exists(mappings_file):
        try:
            with open(mappings_file, "r") as f:
                custom_mappings = json.load(f)
                if isinstance(custom_mappings, dict):
                    mappings.update(custom_mappings)
                    logger.info(f"Loaded {len(custom_mappings)} custom token mappings")
        except Exception as e:
            logger.warning(f"Error loading custom token mappings: {e}")
    
    return mappings

def initialize_coingecko_mappings(coingecko_api):
    """
    Initialize CoinGecko API with token mappings.
    
    Args:
        coingecko_api: CoinGecko API instance to configure
    """
    if not coingecko_api:
        logger.warning("CoinGecko API not available, skipping mapping initialization")
        return
        
    # Load mappings
    mappings = load_token_mappings()
    
    # Initialize CoinGecko with mappings
    for symbol, token_id in mappings.items():
        coingecko_api.add_token_mapping(symbol, token_id)
        logger.info(f"Registered token mapping: {symbol} -> {token_id}")
    
    # Log summary
    logger.info(f"Initialized CoinGecko with {len(mappings)} token mappings")

# Special function to get address for a specific token
def get_token_address(symbol: str) -> Optional[str]:
    """
    Get known token address for a symbol.
    This is used to complement the CoinGecko mappings when addresses are needed.
    
    Args:
        symbol: Token symbol
        
    Returns:
        Token address or None if not found
    """
    # Common token addresses on Solana
    addresses = {
        "SOL": "So11111111111111111111111111111111111111112",
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
        "ETH": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
        "MSOL": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
        "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "ATLAS": "ATLASXmbPQxBUYbxPsV97usA3fPQYEqzQBUHgiFCUsXx",
        "BOOP": "boopkpWqe68MSxLqBGogs8ZbUDN4GXaLhFwNP7mpP1i",
        "SAMO": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
        "SOGENT": "2sXsiQDXiEG2c8VitBbt4CnGSDGjLt4P3dn81Hto7csF"
    }
    
    return addresses.get(symbol.upper())