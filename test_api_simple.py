"""
Simple test of the DeFi API that runs for up to 60 seconds or until it retrieves all data.
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

def run_api_test(max_runtime_seconds=60):
    """Run an API test for up to the specified runtime."""
    start_time = time.time()
    end_time = start_time + max_runtime_seconds
    
    # Initialize stats
    api_calls = 0
    total_pools = 0
    unique_pool_ids = set()
    all_pools = []
    unique_tokens = set()
    token_pairs = []
    dexes = []
    errors = []
    
    # Initialize API client
    api = DefiAggregationAPI()
    
    print(f"Starting DeFi API test for up to {max_runtime_seconds} seconds...")
    print(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Test parameters
    batch_size = 50
    offset = 0
    
    while time.time() < end_time:
        try:
            api_calls += 1
            print(f"\nAPI Call #{api_calls} - Retrieving pools (offset={offset}, limit={batch_size})")
            
            # Make API request
            start_call = time.time()
            pools = api._make_request("pools", params={"limit": batch_size, "offset": offset})
            call_duration = time.time() - start_call
            
            if not pools or not isinstance(pools, list):
                print(f"❌ Invalid response received")
                errors.append(f"Invalid response for offset {offset}")
                break
                
            print(f"✅ Received {len(pools)} pools in {call_duration:.2f} seconds")
            
            # Process pool data
            new_pools = 0
            for pool in pools:
                pool_id = pool.get('poolId', '')
                
                # Skip if we've already processed this pool
                if pool_id in unique_pool_ids:
                    continue
                    
                unique_pool_ids.add(pool_id)
                new_pools += 1
                
                # Get pool data
                name = pool.get('name', 'Unknown')
                dex = pool.get('source', 'Unknown')
                
                # Parse token symbols from name
                token1, token2 = extract_token_symbols_from_name(name)
                
                # Track tokens
                if token1 != "Unknown":
                    unique_tokens.add(token1)
                if token2 != "Unknown":
                    unique_tokens.add(token2)
                    
                # Track token pair
                if token1 != "Unknown" and token2 != "Unknown":
                    token_pairs.append(f"{token1}-{token2}")
                
                # Track DEX
                if dex != "Unknown":
                    dexes.append(dex)
                
                # Store pool data
                pool_data = {
                    'id': pool.get('id', 'Unknown'),
                    'pool_id': pool_id,
                    'name': name,
                    'dex': dex,
                    'token1': token1,
                    'token2': token2,
                    # Get metrics
                    'liquidity': pool.get('metrics', {}).get('tvl', 0),
                    'volume_24h': pool.get('metrics', {}).get('volumeUsd', 0),
                    'apr': pool.get('metrics', {}).get('apy24h', 0)
                }
                
                all_pools.append(pool_data)
            
            total_pools += new_pools
            print(f"Total unique pools: {total_pools} (added {new_pools} new pools)")
            
            # If we received fewer pools than the batch size, we've reached the end
            if len(pools) < batch_size:
                print("Reached end of available data")
                break
                
            # Update offset for next batch
            offset += batch_size
            
            # Check elapsed time
            elapsed = time.time() - start_time
            print(f"Elapsed: {elapsed:.2f} seconds / {max_runtime_seconds} seconds")
            
            # Break if we're about to exceed our time limit
            if time.time() + 3 > end_time:  # Leave 3 seconds margin
                print("Approaching time limit, stopping")
                break
                
            # Brief pause to avoid hammering the API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error during API call: {str(e)}")
            errors.append(str(e))
            break
    
    # Calculate final stats
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print("\n" + "="*60)
    print("DEFI API TEST RESULTS")
    print("="*60)
    
    print(f"\nTest duration: {total_runtime:.2f} seconds")
    print(f"API calls: {api_calls}")
    print(f"Unique pools: {total_pools}")
    
    if api_calls > 0:
        print(f"Avg. time per API call: {total_runtime / api_calls:.2f} seconds")
        print(f"Avg. pools per API call: {total_pools / api_calls:.2f}")
    
    print(f"Unique tokens: {len(unique_tokens)}")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for i, error in enumerate(errors[:3], 1):
            print(f"  {i}. {error}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more")
    
    if not all_pools:
        print("No pools were retrieved!")
        return
    
    # Pool metrics
    total_liquidity = sum(p.get('liquidity', 0) for p in all_pools)
    total_volume = sum(p.get('volume_24h', 0) for p in all_pools)
    
    # Calculate average APR, handling division by zero
    if all_pools:
        avg_apr = sum(p.get('apr', 0) for p in all_pools) / len(all_pools)
    else:
        avg_apr = 0
    
    print(f"\nPOOL METRICS:")
    print(f"Total Liquidity: ${total_liquidity:,.0f}")
    print(f"Total 24h Volume: ${total_volume:,.0f}")
    print(f"Average APR: {avg_apr:.2f}%")
    
    # Token analysis
    token_counts = Counter()
    for pool in all_pools:
        token1 = pool.get('token1')
        token2 = pool.get('token2')
        if token1 and token1 != "Unknown":
            token_counts[token1] += 1
        if token2 and token2 != "Unknown":
            token_counts[token2] += 1
    
    token_pair_counts = Counter(token_pairs)
    dex_counts = Counter(dexes)
    
    print("\nTOP TOKENS:")
    for token, count in token_counts.most_common(10):
        print(f"  {token}: {count} occurrences")
    
    print("\nTOP TOKEN PAIRS:")
    for pair, count in token_pair_counts.most_common(10):
        print(f"  {pair}: {count} pools")
    
    print("\nDEXES:")
    for dex, count in dex_counts.most_common():
        print(f"  {dex}: {count} pools")
        
    # All unique tokens
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
        "pools": total_pools,
        "tokens": len(unique_tokens),
        "token_pairs": len(token_pair_counts),
        "top_tokens": token_counts.most_common(10),
        "top_pairs": token_pair_counts.most_common(10),
        "dexes": dex_counts,
        "total_liquidity": total_liquidity,
        "total_volume": total_volume,
        "avg_apr": avg_apr
    }

if __name__ == "__main__":
    max_runtime = 60  # seconds
    run_api_test(max_runtime)