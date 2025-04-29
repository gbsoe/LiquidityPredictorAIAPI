"""
Quick version of the DeFi API test that will complete within timeout constraints
"""

import time
import json
from defi_aggregation_api import DefiAggregationAPI
from datetime import datetime
from collections import Counter

def run_quick_test(test_duration=15):
    """Run a quick API test for a limited duration"""
    print(f"Starting {test_duration}-second quick API test...")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    end_time = start_time + test_duration
    
    # Initialize API
    api = DefiAggregationAPI()
    
    # Stats
    api_calls = 0
    total_pools = 0
    unique_pools = set()
    token_pairs = []
    dexes = []
    call_times = []
    
    # Make API calls until time runs out
    while time.time() < end_time:
        api_calls += 1
        
        # Start timing
        call_start = time.time()
        
        # Make API call
        print(f"\nAPI Call #{api_calls}...")
        
        try:
            # Get pools
            pools = api._make_request("pools", params={"limit": 20, "offset": 0})
            call_duration = time.time() - call_start
            call_times.append(call_duration)
            
            # Process results
            if pools and isinstance(pools, list):
                pool_count = len(pools)
                total_pools += pool_count
                
                print(f"✅ Received {pool_count} pools in {call_duration:.2f} seconds")
                
                # Extract details from first pool
                if pools:
                    pool = pools[0]
                    pool_id = pool.get('poolId', '')
                    unique_pools.add(pool_id)
                    
                    # Basic info
                    name = pool.get('name', 'Unknown')
                    dex = pool.get('source', 'Unknown')
                    
                    print(f"   Pool: {name}")
                    print(f"   DEX: {dex}")
                    print(f"   ID: {pool_id}")
                    
                    # Extract metrics
                    metrics = pool.get('metrics', {})
                    if metrics:
                        tvl = metrics.get('tvl', 0)
                        apy = metrics.get('apy', 0)
                        volume_24h = metrics.get('volume24h', 0)
                        
                        print(f"   TVL: ${tvl:,.2f}")
                        print(f"   APY: {apy:.2f}%")
                        print(f"   24h Volume: ${volume_24h:,.2f}")
                    
                    # Extract token info from name
                    if name and '-' in name:
                        parts = name.split('-')
                        if len(parts) >= 2:
                            token1 = parts[0].strip()
                            token2_parts = parts[1].split(' ')
                            token2 = token2_parts[0].strip()
                            
                            token_pair = f"{token1}-{token2}"
                            token_pairs.append(token_pair)
                            
                            print(f"   Tokens: {token1} and {token2}")
                    
                    # Track DEX
                    if dex != "Unknown":
                        dexes.append(dex)
            
            # Don't sleep after the last expected call
            if time.time() + 2 < end_time:
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Error in API call {api_calls}: {str(e)}")
            # Brief pause after error
            time.sleep(0.5)
    
    # Calculate stats
    test_duration = time.time() - start_time
    
    print("\n" + "="*60)
    print("QUICK API TEST RESULTS")
    print("="*60)
    
    print(f"\nTest duration: {test_duration:.2f} seconds")
    print(f"API calls completed: {api_calls}")
    print(f"Total pools retrieved: {total_pools}")
    print(f"Unique pools found: {len(unique_pools)}")
    
    if call_times:
        avg_call_time = sum(call_times) / len(call_times)
        print(f"Average call time: {avg_call_time:.2f} seconds")
        print(f"API calls per second: {api_calls / test_duration:.2f}")
    
    token_pair_counts = Counter(token_pairs)
    dex_counts = Counter(dexes)
    
    print("\nTOP TOKEN PAIRS:")
    for pair, count in token_pair_counts.most_common(3):
        print(f"  {pair}: {count} occurrences")
    
    print("\nDEXES:")
    for dex, count in dex_counts.most_common():
        print(f"  {dex}: {count} pools")
    
    print("\n" + "="*60)
    print(f"Test completed at {datetime.now().strftime('%H:%M:%S')}")
    
    # Save summary to file for reference
    summary = {
        "timestamp": datetime.now().isoformat(),
        "duration": test_duration,
        "api_calls": api_calls,
        "total_pools": total_pools,
        "unique_pools": len(unique_pools),
        "avg_call_time": avg_call_time if call_times else None,
        "token_pairs": dict(token_pair_counts),
        "dexes": dict(dex_counts)
    }
    
    try:
        with open("api_test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("Test summary saved to api_test_summary.json")
    except Exception as e:
        print(f"Could not save summary: {str(e)}")

if __name__ == "__main__":
    run_quick_test(15)