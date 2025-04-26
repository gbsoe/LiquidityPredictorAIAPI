"""
Database Population Script for SolPool Insight

This script generates realistic liquidity pool data and stores it in the database.
It's designed to work with rate-limited APIs by implementing robust caching.
"""

import os
import json
import random
import argparse
import time
from datetime import datetime, timedelta
import db_handler

# Import the rate-limited token service for better API handling
from rate_limited_token_service import update_pool_prices_with_rate_limit

# Import functions from solpool_insight.py for generating sample data
from solpool_insight import generate_sample_data

def populate_database(num_pools=50, replace_existing=False):
    """
    Populate the database with pool data
    
    Args:
        num_pools: Number of pools to generate (default: 50)
        replace_existing: Whether to replace existing data (default: False)
    
    Returns:
        Number of pools stored
    """
    print(f"Generating {num_pools} pools...")
    
    # Generate sample data with realistic values
    pools = generate_sample_data(num_pools)
    
    # Ensure token prices are populated with adequate rate limiting
    pools_with_prices = []
    
    # Process in small batches to avoid rate limiting (5 pools at a time)
    batch_size = 5
    for i in range(0, len(pools), batch_size):
        batch = pools[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(pools) + batch_size - 1)//batch_size}...")
        for pool in batch:
            try:
                # Use our rate limited token service to avoid API rate limits
                pool_with_prices = update_pool_prices_with_rate_limit(pool)
                pools_with_prices.append(pool_with_prices)
            except Exception as e:
                print(f"Error getting token prices: {e}")
                # Still add the pool even without prices
                pools_with_prices.append(pool)
                
        # Our rate limited service handles pauses internally, so we don't need an extra pause here
    
    # Store in database
    print(f"Storing {len(pools_with_prices)} pools in database...")
    count = db_handler.store_pools(pools_with_prices, replace=replace_existing)
    
    # Also save to JSON as backup
    with open("extracted_pools.json", "w") as f:
        json.dump(pools_with_prices, f, indent=2)
    
    print(f"Successfully stored {count} pools in database.")
    print(f"Also saved pools to extracted_pools.json as backup.")
    
    return count

def main():
    """Main function to parse command line arguments and populate database"""
    parser = argparse.ArgumentParser(description="Populate SolPool Insight database with realistic pool data")
    parser.add_argument("--count", type=int, default=50, help="Number of pools to generate (default: 50)")
    parser.add_argument("--replace", action="store_true", help="Replace existing data in database")
    args = parser.parse_args()
    
    # Initialize database
    db_handler.init_db()
    
    # Populate database
    count = populate_database(args.count, args.replace)
    print(f"Database population complete. Added {count} pools.")

if __name__ == "__main__":
    main()