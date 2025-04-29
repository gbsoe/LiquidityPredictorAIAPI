"""
Quick version of the DeFi API test that will complete within timeout constraints
"""

import os
import json
import time
from datetime import datetime
from collections import Counter
from defi_aggregation_api import DefiAggregationAPI

def run_quick_test(test_duration=15):
    """Run a quick API test for a limited duration"""
    print(f"Starting quick DeFi API test for approximately {test_duration} seconds...")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    end_time = start_time + test_duration
    
    # Initialize stats
    api_calls = 0
    pools_received = 0
    unique_pool_ids = set()
    unique_tokens = set()
    token_pairs = []
    dexes = []
    
    # Initialize API client
    api = DefiAggregationAPI()
    
    # Run test for limited cycles
    max_cycles = 10
    cycle = 0
    
    while cycle < max_cycles and time.time() < end_time:
        cycle += 1
        api_calls += 1
        
        print(f"\nAPI Call #{api_calls} (cycle {cycle}/{max_cycles})")
        
        # Make API request
        call_start = time.time()
        pools = api._make_request("pools", params={"limit": 20, "offset": 0})
        call_duration = time.time() - call_start
        
        if pools and isinstance(pools, list):
            pool_count = len(pools)
            pools_received += pool_count
            print(f"✅ Received {pool_count} pools in {call_duration:.2f} seconds")
            
            # Process a few pools for token info
            for pool in pools[:5]:  # Process only the first 5 for speed
                pool_id = pool.get('poolId', '')
                unique_pool_ids.add(pool_id)
                
                # Get pool name and extract token symbols
                name = pool.get('name', 'Unknown')
                dex = pool.get('source', 'Unknown')
                
                if name and '-' in name:
                    parts = name.split('-')
                    token1 = parts[0].strip()
                    token2 = parts[1].split(' ')[0].strip()
                    
                    if token1 != "Unknown":
                        unique_tokens.add(token1)
                    if token2 != "Unknown":
                        unique_tokens.add(token2)
                        
                    token_pairs.append(f"{token1}-{token2}")
                
                # Track DEX
                if dex != "Unknown":
                    dexes.append(dex)
        else:
            print(f"❌ Failed to retrieve pools")
            
        elapsed = time.time() - start_time
        print(f"Elapsed: {elapsed:.2f} seconds / {test_duration} seconds")
        
        # Don't go over the time limit
        if time.time() + 2 > end_time:
            break
            
        # Brief pause
        time.sleep(0.1)
    
    # Calculate stats
    total_runtime = time.time() - start_time
    
    print("\n" + "="*60)
    print("QUICK DEFI API TEST RESULTS")
    print("="*60)
    
    print(f"\nTest duration: {total_runtime:.2f} seconds")
    print(f"API calls: {api_calls}")
    print(f"Pools received: {pools_received}")
    print(f"Unique pools processed: {len(unique_pool_ids)}")
    print(f"Unique tokens found: {len(unique_tokens)}")
    
    if api_calls > 0:
        print(f"Pools per second: {pools_received / total_runtime:.2f}")
        print(f"API calls per second: {api_calls / total_runtime:.2f}")
    
    # Token analysis
    token_counts = Counter()
    for token in unique_tokens:
        token_counts[token] += 1
    
    token_pair_counts = Counter(token_pairs)
    dex_counts = Counter(dexes)
    
    print("\nTOP TOKENS:")
    for token, count in token_counts.most_common(10):
        print(f"  {token}: {count} mentions")
    
    print("\nTOP TOKEN PAIRS:")
    for pair, count in token_pair_counts.most_common(10):
        print(f"  {pair}: {count} mentions")
    
    print("\nDEXES:")
    for dex, count in dex_counts.most_common():
        print(f"  {dex}: {count} pools")
    
    print("\nALL TOKENS FOUND:")
    print(", ".join(sorted(list(unique_tokens))))
    
    print("\n" + "="*60)
    print(f"Test completed at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    run_quick_test(15)  # Run for just 15 seconds