"""
Continuous test of the DeFi API that runs for exactly 60 seconds,
repeatedly retrieving pool data to simulate sustained load.
"""

import os
import json
import time
import sys
from datetime import datetime
from collections import Counter
from defi_aggregation_api import DefiAggregationAPI

def extract_token_symbols_from_name(name):
    """Extract token symbols from pool name"""
    if not name or '-' not in name:
        return "Unknown", "Unknown"
        
    parts = name.split('-')
    if len(parts) < 2:
        return "Unknown", "Unknown"
        
    # First part is usually token1
    token1 = parts[0].strip()
    
    # Second part might have extra text (like "LP" or DEX name)
    # Try to extract just the token symbol
    token2_parts = parts[1].split(' ')
    token2 = token2_parts[0].strip()
    
    return token1, token2

def run_continuous_api_test(runtime_seconds=60):
    """Run a continuous API test for exactly the specified runtime."""
    start_time = time.time()
    end_time = start_time + runtime_seconds
    
    # Initialize stats
    api_calls = 0
    total_pool_responses = 0
    unique_pool_ids = set()
    all_pools = []
    unique_tokens = set()
    token_pairs = []
    dexes = []
    errors = []
    api_call_times = []
    
    # Initialize API client
    api = DefiAggregationAPI()
    
    print(f"Starting continuous DeFi API test for {runtime_seconds} seconds...")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Continue making API calls until we reach the time limit
    offset = 0
    batch_size = 20  # Smaller batch size to get more API calls
    
    print("Beginning API calls in sequence...")
    
    try:
        while time.time() < end_time:
            # Make API request with timing
            api_calls += 1
            
            call_start = time.time()
            
            print(f"\nAPI Call #{api_calls} (offset={offset}, limit={batch_size})")
            
            try:
                pools = api._make_request("pools", params={"limit": batch_size, "offset": offset})
                call_duration = time.time() - call_start
                api_call_times.append(call_duration)
                
                pool_count = len(pools) if pools and isinstance(pools, list) else 0
                print(f"✅ Received {pool_count} pools in {call_duration:.2f} seconds")
                
                if not pools or not isinstance(pools, list):
                    print(f"❌ Invalid response received")
                    errors.append(f"Invalid response for offset {offset}")
                    # Reset offset and continue
                    offset = 0
                    continue
                
                # Process pools
                total_pool_responses += pool_count
                
                # Process each pool (just count tokens, don't store entire pool data)
                for pool in pools:
                    pool_id = pool.get('poolId', '')
                    unique_pool_ids.add(pool_id)
                    
                    # Get pool name and DEX
                    name = pool.get('name', 'Unknown')
                    dex = pool.get('source', 'Unknown')
                    
                    # Extract token symbols
                    token1, token2 = extract_token_symbols_from_name(name)
                    
                    # Track tokens and pairs
                    if token1 != "Unknown":
                        unique_tokens.add(token1)
                    if token2 != "Unknown":
                        unique_tokens.add(token2)
                        
                    if token1 != "Unknown" and token2 != "Unknown":
                        token_pairs.append(f"{token1}-{token2}")
                    
                    # Track DEX
                    if dex != "Unknown":
                        dexes.append(dex)
                
                # Advance offset for next call, or reset if we've reached the end
                if pool_count < batch_size:
                    print("Reached end of data, resetting offset...")
                    offset = 0
                else:
                    offset += batch_size
                    
                # Print progress every 10 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    print(f"\n--- {int(elapsed)} seconds elapsed ---")
                    print(f"API calls so far: {api_calls}")
                    print(f"Pools processed: {total_pool_responses}")
                    print(f"Unique pools: {len(unique_pool_ids)}")
                    print(f"Unique tokens: {len(unique_tokens)}")
                    print("")
                
                # Brief pause to avoid hammering the API too hard
                time.sleep(0.1)
                
            except Exception as e:
                error_msg = f"Error in API call {api_calls}: {str(e)}"
                print(f"❌ {error_msg}")
                errors.append(error_msg)
                # Reset offset and continue
                offset = 0
                time.sleep(1)  # Wait a bit longer after an error
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    # Ensure we run for exactly the requested time
    remaining = end_time - time.time()
    if remaining > 0:
        print(f"Waiting {remaining:.2f} seconds to complete full {runtime_seconds}-second test...")
        time.sleep(remaining)
    
    # Calculate final stats
    total_runtime = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"DEFI API {runtime_seconds}-SECOND CONTINUOUS TEST RESULTS")
    print("="*60)
    
    print(f"\nActual test duration: {total_runtime:.2f} seconds")
    print(f"Total API calls: {api_calls}")
    print(f"Total pool responses: {total_pool_responses}")
    print(f"Unique pools found: {len(unique_pool_ids)}")
    print(f"Unique tokens found: {len(unique_tokens)}")
    
    if api_calls > 0:
        print(f"Avg. time per API call: {sum(api_call_times) / len(api_call_times):.2f} seconds")
        print(f"Fastest API call: {min(api_call_times):.2f} seconds")
        print(f"Slowest API call: {max(api_call_times):.2f} seconds")
        print(f"API calls per second: {api_calls / total_runtime:.2f}")
        print(f"Pools per second: {total_pool_responses / total_runtime:.2f}")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for i, error in enumerate(errors[:3], 1):
            print(f"  {i}. {error}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more")
    
    # Token analysis
    token_counts = Counter()
    for token in unique_tokens:
        token_counts[token] += 1
    
    token_pair_counts = Counter(token_pairs)
    dex_counts = Counter(dexes)
    
    print("\nTOP TOKENS:")
    for token, count in token_counts.most_common(10):
        print(f"  {token}: {count} occurrences")
    
    print("\nTOP TOKEN PAIRS:")
    for pair, count in token_pair_counts.most_common(10):
        print(f"  {pair}: {count} occurrences")
    
    print("\nDEXES:")
    for dex, count in dex_counts.most_common():
        print(f"  {dex}: {count} occurrences")
    
    print("\nALL UNIQUE TOKENS:")
    tokens_sorted = sorted(list(unique_tokens))
    tokens_display = ", ".join(tokens_sorted)
    if len(tokens_display) > 100:
        print(f"  {tokens_display[:100]}... (and {len(tokens_sorted) - len(tokens_display[:100].split(', '))} more)")
    else:
        print(f"  {tokens_display}")
    
    print("\n" + "="*60)
    print(f"Test completed at {datetime.now().strftime('%H:%M:%S')}")
    
    return {
        "runtime": total_runtime,
        "api_calls": api_calls,
        "total_pools": total_pool_responses,
        "unique_pools": len(unique_pool_ids),
        "unique_tokens": len(unique_tokens),
        "errors": len(errors),
        "avg_call_time": sum(api_call_times) / len(api_call_times) if api_call_times else 0,
        "calls_per_second": api_calls / total_runtime,
        "pools_per_second": total_pool_responses / total_runtime,
        "top_tokens": token_counts.most_common(10),
        "top_pairs": token_pair_counts.most_common(10)
    }

if __name__ == "__main__":
    # Run for 30 seconds to avoid environment timeout
    runtime = 30  # seconds
    run_continuous_api_test(runtime)