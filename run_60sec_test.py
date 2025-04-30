"""
60-Second Pool Data Retrieval Test

This script will retrieve liquidity pool data from the DeFi API
for exactly 60 seconds and save the results to both the database
and a JSON file for inspection.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("60sec_test")

# Check for API key
api_key = os.getenv("DEFI_API_KEY")
if not api_key:
    logger.error("No API key found. Please set the DEFI_API_KEY environment variable.")
    sys.exit(1)

# Import our modules
try:
    from defi_aggregation_api import DefiAggregationAPI
    import db_handler
    from token_price_service import TokenPriceService, get_token_price, update_pool_with_token_prices
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    sys.exit(1)

# Initialize API client
api = DefiAggregationAPI(api_key=api_key)

def run_test():
    """Run the 60-second test"""
    logger.info("Starting 60-second pool data retrieval test")
    logger.info(f"Using API base URL: {api.base_url}")
    
    # Initialize stats
    start_time = time.time()
    end_time = start_time + 60  # Run for 60 seconds
    
    stats = {
        "total_pools": 0,
        "unique_pools": set(),
        "total_tokens": 0,
        "unique_tokens": set(),
        "dex_counts": {},
        "api_calls": 0,
        "errors": 0,
        "db_writes": 0
    }
    
    all_pools = []
    
    # Get supported DEXes
    logger.info("Retrieving supported DEXes...")
    try:
        dexes = api.get_supported_dexes()
        logger.info(f"Found {len(dexes)} supported DEXes: {', '.join(dexes)}")
    except Exception as e:
        logger.error(f"Error retrieving DEXes: {e}")
        dexes = ["raydium", "orca", "meteora"]  # Fallback
    
    # Main test loop - keep fetching until time runs out
    while time.time() < end_time:
        remaining_time = end_time - time.time()
        logger.info(f"Continuing test for {remaining_time:.1f} more seconds")
        
        try:
            # For each DEX, fetch pools
            for dex in dexes:
                if time.time() >= end_time:
                    break
                    
                logger.info(f"Fetching pools for DEX: {dex}")
                
                try:
                    # Get transformable pools for this DEX (limit 20 to avoid overloading)
                    pools = api.get_pools(limit=20, source=dex.lower())
                    stats["api_calls"] += 1
                    
                    if not pools:
                        logger.warning(f"No pools returned for DEX: {dex}")
                        continue
                        
                    logger.info(f"Retrieved {len(pools)} pools for DEX: {dex}")
                    
                    # Transform and track
                    transformed_pools = []
                    for pool in pools:
                        transformed = api.transform_pool_data(pool)
                        if transformed:
                            # Add token prices
                            transformed = update_pool_with_token_prices(transformed)
                            transformed_pools.append(transformed)
                            
                            # Track stats
                            stats["total_pools"] += 1
                            stats["unique_pools"].add(transformed["id"])
                            
                            # Track tokens
                            if "token1_symbol" in transformed and transformed["token1_symbol"] != "Unknown":
                                stats["unique_tokens"].add(transformed["token1_symbol"])
                                stats["total_tokens"] += 1
                            if "token2_symbol" in transformed and transformed["token2_symbol"] != "Unknown":
                                stats["unique_tokens"].add(transformed["token2_symbol"])
                                stats["total_tokens"] += 1
                                
                            # Track DEX counts
                            dex_name = transformed.get("dex", "Unknown")
                            if dex_name in stats["dex_counts"]:
                                stats["dex_counts"][dex_name] += 1
                            else:
                                stats["dex_counts"][dex_name] = 1
                    
                    # Store in database
                    try:
                        stored = db_handler.store_pools(transformed_pools)
                        stats["db_writes"] += stored
                        logger.info(f"Stored {stored} pools in database")
                        
                        # Add to our collection
                        all_pools.extend(transformed_pools)
                    except Exception as db_err:
                        logger.error(f"Database error: {db_err}")
                        stats["errors"] += 1
                    
                except Exception as dex_err:
                    logger.error(f"Error retrieving pools for DEX {dex}: {dex_err}")
                    stats["errors"] += 1
                    
                # Sleep briefly to avoid rate limits
                time.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Test loop error: {e}")
            stats["errors"] += 1
            time.sleep(1)  # Backoff on error
    
    # Test complete - summarize results
    elapsed = time.time() - start_time
    logger.info(f"Test completed in {elapsed:.2f} seconds")
    
    # Convert sets to lists for JSON serialization
    stats["unique_pools"] = list(stats["unique_pools"])
    stats["unique_tokens"] = list(stats["unique_tokens"])
    
    # Add timing info
    stats["start_time"] = datetime.fromtimestamp(start_time).isoformat()
    stats["end_time"] = datetime.fromtimestamp(time.time()).isoformat()
    stats["elapsed_seconds"] = elapsed
    stats["pools_per_second"] = stats["total_pools"] / elapsed if elapsed > 0 else 0
    
    # Save results to JSON for inspection
    results = {
        "stats": stats,
        "test_duration": f"{elapsed:.2f} seconds",
        "timestamp": datetime.now().isoformat(),
        "sample_pools": all_pools[:5] if all_pools else []  # Just save a few samples
    }
    
    # Save all pools to a file
    try:
        with open("60sec_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved to 60sec_test_results.json")
        
        # Also save all the pools
        with open("60sec_test_all_pools.json", "w") as f:
            json.dump(all_pools, f, indent=2)
        logger.info(f"All {len(all_pools)} pools saved to 60sec_test_all_pools.json")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("60-SECOND TEST RESULTS")
    print("="*60)
    print(f"Total pools retrieved: {stats['total_pools']}")
    print(f"Unique pools found: {len(stats['unique_pools'])}")
    print(f"Unique tokens found: {len(stats['unique_tokens'])}")
    print(f"API calls made: {stats['api_calls']}")
    print(f"Database writes: {stats['db_writes']}")
    print(f"Errors encountered: {stats['errors']}")
    print(f"Test duration: {elapsed:.2f} seconds")
    print(f"Processing rate: {stats['pools_per_second']:.2f} pools/second")
    
    print("\nDEX DISTRIBUTION:")
    for dex_name, count in stats["dex_counts"].items():
        print(f"  {dex_name}: {count} pools")
    
    print("\nTOKENS FOUND:")
    token_list = ", ".join(sorted(stats["unique_tokens"]))
    if len(token_list) > 100:
        token_list = token_list[:100] + "..."
    print(f"  {token_list}")
    
    print("\n" + "="*60)
    
    return stats

if __name__ == "__main__":
    # Initialize database
    db_handler.init_db()
    
    # Run the test
    stats = run_test()
    
    # Print recommendation
    print("\nTest complete! To verify the results:")
    print("1. Check '60sec_test_results.json' for detailed metrics")
    print("2. Check '60sec_test_all_pools.json' for complete pool data")
    print("3. Restart the application to view the pools in the UI")
    print("\nNote: All data retrieved is authentic from the API source.")