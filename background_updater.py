"""
Background Data Updater for Liquidity Pool Information

This module provides a background service that:
1. Continuously fetches pool data in tranches
2. Updates the extracted_pools.json file with fresh data
3. Manages the update frequency to avoid rate limits
"""

import json
import logging
import os
import random
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("background_updater.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("background_updater")

# Global flag to control the background thread
keep_running = True

def get_current_pools() -> List[Dict[str, Any]]:
    """Load the current pool data from the cache file"""
    cache_file = "extracted_pools.json"
    
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                pools = json.load(f)
                if pools and isinstance(pools, list):
                    return pools
    except Exception as e:
        logger.error(f"Error loading current pools: {e}")
    
    return []

def save_pools(pools: List[Dict[str, Any]]) -> bool:
    """Save the updated pools to the cache file"""
    cache_file = "extracted_pools.json"
    temp_file = "extracted_pools.tmp.json"
    
    try:
        # First write to a temporary file
        with open(temp_file, "w") as f:
            json.dump(pools, f, indent=2)
        
        # Then rename to the actual file (atomic operation)
        os.replace(temp_file, cache_file)
        logger.info(f"Successfully saved {len(pools)} pools to {cache_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving pools: {e}")
        return False

def update_pools_for_dex(current_pools: List[Dict[str, Any]], dex_name: str, max_new: int = 3) -> List[Dict[str, Any]]:
    """
    Update pools for a specific DEX by fetching new data
    
    Args:
        current_pools: Current pool data
        dex_name: DEX to update
        max_new: Maximum number of new pools to add (reduced to prevent rate limiting)
        
    Returns:
        Updated list of pools
    """
    try:
        # Import here to avoid circular imports
        from onchain_extractor import OnChainExtractor
        
        # Get the RPC endpoint from the environment
        rpc_endpoint = os.getenv("SOLANA_RPC_ENDPOINT")
        if not rpc_endpoint:
            logger.error("No RPC endpoint configured")
            return current_pools
        
        logger.info(f"Updating pools for {dex_name}")
        
        # Initialize the extractor with timeout protection
        try:
            extractor = OnChainExtractor(rpc_endpoint=rpc_endpoint)
        except Exception as ext_error:
            logger.error(f"Failed to initialize extractor: {ext_error}")
            return current_pools
        
        # Map of DEX names to their extraction methods
        dex_program_ids = {
            "Raydium": ["675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"],
            "Orca": ["whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"],
            "Jupiter": ["JUP2jxvXaqu7NQY1GmNF4m1vodw12LVXYxbFL2uJvfo"],
            "Saber": ["SSwpkEEcbUqx4vtoEByFjSkhKdCT862DNVb52nZg1UZ"],
            "Meteora": ["M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K"]
        }
        
        # Get the current program IDs for this DEX
        if dex_name not in dex_program_ids:
            logger.warning(f"Unknown DEX: {dex_name}")
            return current_pools
        
        program_ids = dex_program_ids[dex_name]
        if not program_ids:
            logger.warning(f"No program IDs for DEX: {dex_name}")
            return current_pools
        
        # Extract a batch of new pools for this DEX
        new_pools = []
        try:
            # Since specific extraction methods may not be implemented in OnChainExtractor,
            # use a safer approach with extract_and_enrich_pools
            try:
                all_pools = extractor.extract_and_enrich_pools(max_per_dex=max_new)
                if all_pools:
                    # Filter to just this DEX
                    for pool in all_pools:
                        if isinstance(pool, dict) and pool.get('dex') == dex_name:
                            new_pools.append(pool)
                    logger.info(f"Found {len(new_pools)} pools for {dex_name}")
            except AttributeError:
                # If extract_and_enrich_pools isn't available, try a more generic approach
                logger.warning(f"extract_and_enrich_pools not found, using generic approach")
                
                # Try to get program accounts
                for program_id in program_ids:
                    try:
                        accounts = extractor.get_program_accounts(program_id)
                        if accounts:
                            logger.info(f"Found {len(accounts)} accounts for {dex_name} program {program_id}")
                            # Create basic pool data
                            for i, acct in enumerate(accounts[:max_new]):
                                pool_id = acct.get('pubkey', f"unknown-{i}")
                                # Create a minimal pool entry
                                pool = {
                                    "id": pool_id,
                                    "dex": dex_name,
                                    "name": f"{dex_name} Pool {i+1}",
                                    "created_at": datetime.now().isoformat(),
                                    "updated_at": datetime.now().isoformat()
                                }
                                new_pools.append(pool)
                    except Exception as prog_error:
                        logger.error(f"Error fetching program accounts for {program_id}: {prog_error}")
                
                # If we managed to get any pools, get token data if possible
                if new_pools and hasattr(extractor, 'get_token_metadata'):
                    try:
                        for pool in new_pools:
                            # Try to get token data - this is very implementation dependent
                            # so just a basic attempt that might work
                            token_data = extractor.get_token_metadata(pool.get('id'))
                            if token_data:
                                pool.update(token_data)
                    except Exception as token_error:
                        logger.error(f"Error fetching token data: {token_error}")
            
            # Add a reasonable delay between program IDs to avoid rate limiting
            time.sleep(3.0)
        except Exception as e:
            logger.error(f"Error extracting pools for {dex_name}: {e}")
        
        if not new_pools:
            logger.warning(f"No new pools extracted for {dex_name}")
            return current_pools
        
        logger.info(f"Extracted {len(new_pools)} new pools for {dex_name}")
        
        # Create a map of current pools by ID
        current_pool_map = {pool["id"]: pool for pool in current_pools}
        
        # Add new pools and update existing ones
        for pool in new_pools:
            pool_id = pool["id"]
            if pool_id in current_pool_map:
                # Update existing pool
                current_pool_map[pool_id].update(pool)
            else:
                # Add new pool
                current_pool_map[pool_id] = pool
        
        # Convert back to list
        updated_pools = list(current_pool_map.values())
        logger.info(f"Updated pool list now contains {len(updated_pools)} pools")
        
        return updated_pools
    except Exception as e:
        logger.error(f"Error in update_pools_for_dex for {dex_name}: {e}")
        return current_pools

def background_update_thread():
    """Main function for the background update thread"""
    logger.info("Starting background update thread")
    
    # List of DEXes to update in rotation - start with the smaller ones first to avoid rate limiting
    dexes = ["Saber", "Meteora", "Jupiter", "Orca", "Raydium"]
    dex_index = 0
    
    # Error counter to back off more aggressively if errors continue
    error_count = 0
    
    while keep_running:
        try:
            # Get the next DEX to update
            dex_name = dexes[dex_index]
            dex_index = (dex_index + 1) % len(dexes)
            
            logger.info(f"Preparing to update pools for {dex_name}")
            
            # Get current pools
            try:
                current_pools = get_current_pools()
                initial_count = len(current_pools)
                logger.info(f"Retrieved {initial_count} current pools")
            except Exception as pool_error:
                logger.error(f"Error getting current pools: {pool_error}")
                current_pools = []
                initial_count = 0
            
            # Update pools for this DEX with error handling
            try:
                updated_pools = update_pools_for_dex(current_pools, dex_name)
            except Exception as update_error:
                logger.error(f"Error updating pools for {dex_name}: {update_error}")
                updated_pools = current_pools
            
            # Save if we have changes
            if len(updated_pools) > initial_count:
                try:
                    save_pools(updated_pools)
                    logger.info(f"Added {len(updated_pools) - initial_count} new pools from {dex_name}")
                    # Reset error count on success
                    error_count = 0
                except Exception as save_error:
                    logger.error(f"Error saving updated pools: {save_error}")
                    error_count += 1
            
            # Wait before the next update to avoid rate limits
            # Longer interval between 120-180 seconds (2-3 minutes) to reduce load
            wait_time = random.uniform(120, 180)
            
            # If we've had errors, increase wait time
            if error_count > 0:
                wait_time *= (1 + error_count * 0.5)  # 50% more time per error
                
            logger.info(f"Waiting {wait_time:.1f}s before next update")
            
            # Break wait into small chunks to check keep_running flag
            for _ in range(int(wait_time)):
                if not keep_running:
                    break
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Critical error in background update thread: {e}")
            error_count += 1
            # Wait before retry - longer if we've had multiple errors
            wait_time = 60 * (1 + error_count)  # 1 minute + 1 minute per previous error
            logger.info(f"Backing off for {wait_time} seconds after error")
            time.sleep(wait_time)

def start_background_updater():
    """Start the background updater thread"""
    global keep_running
    
    # Make sure we're not already running
    if keep_running:
        # Create and start the thread
        update_thread = threading.Thread(
            target=background_update_thread,
            daemon=True  # Daemon threads exit when the main thread exits
        )
        update_thread.start()
        logger.info("Background updater started")
        return True
    return False

def stop_background_updater():
    """Stop the background updater thread"""
    global keep_running
    keep_running = False
    logger.info("Background updater stopping...")
    return True

if __name__ == "__main__":
    # Test the background updater
    start_background_updater()
    
    # Run for 5 minutes
    try:
        print("Background updater running for 5 minutes... Press Ctrl+C to stop early")
        time.sleep(300)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stop_background_updater()
        print("Background updater stopped")