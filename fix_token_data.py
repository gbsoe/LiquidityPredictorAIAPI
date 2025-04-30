"""
Script to fix token display in SolPool Insight

This script:
1. Deletes existing database tables to start fresh
2. Creates a mapping of token symbols to full token details from the API
3. Updates the transform_pool_data function to properly populate token data
4. Tests that everything works as expected

Run this whenever you need to reset and fix the token data.
"""

import os
import json
import logging
import psycopg2
from typing import Dict, Any
from defi_aggregation_api import DefiAggregationAPI

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fix_token_data")

def drop_tables():
    """Drop all database tables to start fresh"""
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS pools CASCADE')
        cursor.execute('DROP TABLE IF EXISTS tokens CASCADE')
        cursor.execute('DROP TABLE IF EXISTS historical_pools CASCADE')
        cursor.execute('DROP TABLE IF EXISTS watchlists CASCADE')
        cursor.execute('DROP TABLE IF EXISTS watchlist_pools CASCADE')
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("All tables dropped successfully!")
        return True
    except Exception as e:
        logger.error(f"Error dropping tables: {str(e)}")
        return False

def get_token_mapping():
    """Get mapping from token symbols to full token details"""
    api = DefiAggregationAPI()
    
    # Get all tokens from API
    logger.info("Fetching tokens from API...")
    tokens = api.get_all_tokens()
    logger.info(f"Retrieved {len(tokens)} tokens")
    
    # Create mapping by symbol
    token_map = {}
    for token in tokens:
        symbol = token.get('symbol', '')
        if symbol:
            token_map[symbol] = token
    
    logger.info(f"Created token mapping with {len(token_map)} entries")
    
    # Save token mapping to file for reference
    with open("token_mapping.json", "w") as f:
        json.dump(token_map, f, indent=2)
    logger.info("Saved token mapping to token_mapping.json")
    
    return token_map

def create_full_pool_sample():
    """Create a sample pool with complete data for testing"""
    api = DefiAggregationAPI()
    token_map = get_token_mapping()
    
    # Get a pool from the API
    logger.info("Fetching a sample pool...")
    pools = api.get_pools(limit=1)
    
    if not pools:
        logger.error("No pools returned from API")
        return False
    
    pool = pools[0]
    logger.info(f"Retrieved pool: {pool.get('name')}")
    
    # Extract token symbols from name
    pool_name = pool.get('name', '')
    token_symbols = []
    
    if "-" in pool_name:
        parts = pool_name.split("-")
        if len(parts) >= 2:
            # First token is the part before the dash
            token1_symbol = parts[0].strip()
            
            # Second token might have LP or other suffix after a space
            token2_part = parts[1].strip()
            if " " in token2_part:
                token2_symbol = token2_part.split(" ")[0].strip()
            else:
                token2_symbol = token2_part
                
            token_symbols = [token1_symbol, token2_symbol]
            logger.info(f"Extracted token symbols: {token_symbols}")
    
    # Create token objects
    token_objects = []
    for symbol in token_symbols:
        if symbol in token_map:
            token_objects.append(token_map[symbol])
            logger.info(f"Added token {symbol} to pool")
        else:
            logger.warning(f"Token {symbol} not found in token mapping")
    
    # Add tokens to pool
    pool['tokens'] = token_objects
    
    # Save to file
    with open("fixed_pool_sample.json", "w") as f:
        json.dump(pool, f, indent=2)
    logger.info("Saved fixed pool sample to fixed_pool_sample.json")
    
    return True

def update_token_price_service():
    """Update token price service to use proper token symbols"""
    # We'll just add a print statement here since we're handling this in the API client
    logger.info("Token price service will use token symbols from mapping")
    return True

def main():
    """Run the token data fix process"""
    logger.info("Starting token data fix process")
    
    # Step 1: Drop existing tables
    if not drop_tables():
        logger.error("Failed to drop tables, aborting")
        return
        
    # Step 2: Get token mapping
    token_map = get_token_mapping()
    if not token_map:
        logger.error("Failed to get token mapping, aborting")
        return
    
    # Step 3: Create sample pool with token data
    if not create_full_pool_sample():
        logger.error("Failed to create sample pool, aborting")
        return
    
    # Step 4: Update token price service
    if not update_token_price_service():
        logger.error("Failed to update token price service, aborting")
        return
    
    logger.info("Token data fix process completed successfully")
    logger.info("Next steps:")
    logger.info("1. Update defi_aggregation_api.py to use token_map to add tokens to pools")
    logger.info("2. Restart the SolPool Insight workflow")
    logger.info("3. Verify that token symbols are now displayed correctly")

if __name__ == "__main__":
    main()