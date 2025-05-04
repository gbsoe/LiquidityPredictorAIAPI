#!/usr/bin/env python3
"""
Initialize Token Mappings for CoinGecko API

This script ensures that all tokens have proper mappings in CoinGecko, particularly
focusing on the tokens shown in the user's screenshot that weren't showing prices.

This is a one-time setup script to populate the token mappings into CoinGecko.
"""

import logging
from coingecko_api import coingecko_api
from token_mappings import DEFAULT_TOKEN_MAPPINGS, initialize_coingecko_mappings, get_token_address

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("token_mapper")

def set_explicit_token_mappings():
    """Set explicit mappings for tokens that don't auto-map correctly."""
    # These are the mappings for tokens from the screenshot
    mappings = {
        "SOGENT": "sogent",
        "SOOMER": "soomer",
        "SPEC": "spec-finance",
        "POOP": "poochain-powering-poo-fun",
        "FARTCOIN": "fartcoin",
        "LAYER": "layer2dao",
        "LFG": "linkforgood", 
        "BOOP": "boop-2",
        "ATLAS": "star-atlas",
        "MSOL": "marinade-staked-sol",
        "STSOL": "lido-staked-sol"
    }
    
    # Addresses for these tokens
    addresses = {
        "SOGENT": "2sXsiQDXiEG2c8VitBbt4CnGSDGjLt4P3dn81Hto7csF",
        "SOOMER": "CTh5k7EHD2HBX64xZkeBDwmHskWvNq5WB8f4PWuW1hmz",
        "SPEC": "2DPKk2yp777kefkfHZJNZMLWX1EGWQJ2abAGbTibAPpu", 
        "POOP": "Ero5K81LWiJnvGaX3NZaG9e84EbA3Lib6uZyoNgfJ2s3",
        "FARTCOIN": "4eAqqHZMZfNfTiicDdJMc5DTkKdpYpnvbqss7vZXJeKa",
        "LAYER": "LayeracuK5G5kVWMuNMGUdwQKVXk7DzYQ4Ywj9K6M1X",
        "LFG": "LFG1a1VPLhuyxnzGEWQAQnUvn8x27p9LZxUHGYfKXjD",
        "BOOP": "boopkpWqe68MSxLqBGogs8ZbUDN4GXaLhFwNP7mpP1i", 
        "MSOL": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
        "STSOL": "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj"
    }
    
    # Add all mappings to CoinGecko
    count = 0
    for symbol, coingecko_id in mappings.items():
        # Add symbol mapping
        coingecko_api.add_token_mapping(symbol, coingecko_id)
        logger.info(f"Added token mapping: {symbol} -> {coingecko_id}")
        count += 1
        
        # If we have an address, add address mapping
        if symbol in addresses:
            address = addresses[symbol]
            coingecko_api.add_address_mapping(address, coingecko_id)
            logger.info(f"Added address mapping: {address} -> {coingecko_id}")
    
    logger.info(f"Added {count} explicit token mappings to CoinGecko")

def main():
    """Main function to initialize all token mappings."""
    logger.info("Initializing token mappings...")
    
    # Initialize with the standard mappings
    initialize_coingecko_mappings(coingecko_api)
    
    # Add explicit mappings for problematic tokens
    set_explicit_token_mappings()
    
    logger.info("Token mapping initialization complete")

if __name__ == "__main__":
    main()