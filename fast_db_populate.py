"""
Fast Database Population Script for SolPool Insight

This script quickly populates the database with 50 pools by cloning and modifying the
existing 10 pools to create 40 additional pools for testing.
"""

import os
import json
import random
import time
from datetime import datetime, timedelta
import db_handler
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

def fast_populate_db():
    """Quickly populate the database with additional pools"""
    print("Fast database population starting...")
    
    # Initialize database connection
    if not hasattr(db_handler, 'engine') or db_handler.engine is None:
        print("Database connection not available")
        return False
    
    # Create a session
    Session = sessionmaker(bind=db_handler.engine)
    session = Session()
    
    try:
        # Get existing pool count
        count_result = session.execute(sa.text("SELECT COUNT(*) FROM liquidity_pools")).scalar()
        print(f"Current pool count: {count_result}")
        
        # If we already have 50+ pools, no need to add more
        if count_result >= 50:
            print("Database already has 50+ pools, no need to add more.")
            return True
        
        # Get existing pools
        existing_pools = db_handler.get_pools()
        print(f"Retrieved {len(existing_pools)} existing pools")
        
        if not existing_pools:
            print("No existing pools found to clone")
            return False
        
        # How many more we need to reach 50
        pools_to_add = 50 - count_result
        print(f"Adding {pools_to_add} more pools...")
        
        # Generate new pools by cloning and modifying existing ones
        new_pools = []
        
        # DEXes for variety
        dexes = ["Raydium", "Orca", "Jupiter", "Meteora", "Saber"]
        
        # Categories for variety
        categories = ["Major", "Meme", "DeFi", "Gaming", "Stablecoin", "Other"]
        
        for i in range(pools_to_add):
            # Pick a random existing pool to clone
            source_pool = random.choice(existing_pools)
            
            # Create a copy and modify it
            new_pool = source_pool.copy()
            
            # Generate a unique ID
            new_pool["id"] = ''.join(random.choices('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz', k=44))
            
            # Modify some attributes to make it different
            new_pool["dex"] = random.choice(dexes)
            new_pool["category"] = random.choice(categories)
            
            # Adjust metrics slightly
            new_pool["liquidity"] = source_pool["liquidity"] * random.uniform(0.8, 1.2)
            new_pool["volume_24h"] = source_pool["volume_24h"] * random.uniform(0.7, 1.3)
            new_pool["apr"] = source_pool["apr"] * random.uniform(0.9, 1.1)
            
            # Adjust changes
            new_pool["apr_change_24h"] = random.uniform(-2, 2)
            new_pool["apr_change_7d"] = random.uniform(-5, 5)
            new_pool["tvl_change_24h"] = random.uniform(-3, 3)
            new_pool["tvl_change_7d"] = random.uniform(-7, 7)
            
            # Add to our new pools list
            new_pools.append(new_pool)
        
        # Store the new pools in the database
        count = db_handler.store_pools(new_pools, replace=False)
        print(f"Successfully added {count} new pools to database")
        
        # Verify new count
        new_count = session.execute(sa.text("SELECT COUNT(*) FROM liquidity_pools")).scalar()
        print(f"Updated pool count: {new_count}")
        
        return True
    except Exception as e:
        print(f"Error populating database: {e}")
        return False
    finally:
        session.close()

if __name__ == "__main__":
    # Initialize database schema if needed
    db_handler.init_db()
    
    # Populate database with additional pools
    success = fast_populate_db()
    
    if success:
        print("Database population completed successfully")
    else:
        print("Database population failed")