"""
Run a 60-second continuous test of the DeFi API
"""

import time
from defi_aggregation_api import DefiAggregationAPI
from datetime import datetime
from collections import Counter

def run_60_second_test():
    """Run a continuous test for 60 seconds"""
    print(f"Starting 60-second DeFi API continuous test...")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    end_time = start_time + 60
    
    # Initialize API client
    api = DefiAggregationAPI()
    
    # Stats
    api_calls = 0
    pools_retrieved = 0
    unique_pools = set()
    token_pairs = []
    dexes = []
    call_times = []
    errors = []
    
    print("Beginning continuous API calls...\n")
    
    try:
        while time.time() < end_time:
            # Make API call
            api_calls += 1
            call_start = time.time()
            
            try:
                print(f"API Call #{api_calls}...")
                pools = api._make_request("pools", params={"limit": 20, "offset": 0})
                call_duration = time.time() - call_start
                call_times.append(call_duration)
                
                if pools and isinstance(pools, list):
                    pools_count = len(pools)
                    pools_retrieved += pools_count
                    
                    print(f"✅ Received {pools_count} pools in {call_duration:.2f} seconds")
                    
                    # Process just the first pool to avoid too much output
                    if pools:
                        pool = pools[0]
                        pool_id = pool.get('poolId', '')
                        unique_pools.add(pool_id)
                        
                        # Get basic info
                        name = pool.get('name', 'Unknown')
                        dex = pool.get('source', 'Unknown')
                        
                        # Track DEX
                        if dex != "Unknown":
                            dexes.append(dex)
                        
                        # Extract token pairs from name
                        if name and '-' in name:
                            parts = name.split('-')
                            if len(parts) >= 2:
                                token1 = parts[0].strip()
                                token2_parts = parts[1].split(' ')
                                token2 = token2_parts[0].strip()
                                token_pairs.append(f"{token1}-{token2}")
                
                # Print progress every 10 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    seconds_left = int(end_time - time.time())
                    print(f"\n--- {int(elapsed)} seconds elapsed, {seconds_left} seconds remaining ---")
                    print(f"API calls: {api_calls}")
                    print(f"Total pools retrieved: {pools_retrieved}")
                    if call_times:
                        print(f"Average call time: {sum(call_times) / len(call_times):.2f} seconds")
                    print("")
                
                # Sleep briefly to avoid hammering the API too hard
                time.sleep(0.1)
                
            except Exception as e:
                error_msg = f"Error in API call {api_calls}: {str(e)}"
                print(f"❌ {error_msg}")
                errors.append(error_msg)
                time.sleep(1)  # Longer pause after error
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    # Calculate final stats
    total_runtime = time.time() - start_time
    
    print("\n" + "="*60)
    print("60-SECOND DeFi API TEST RESULTS")
    print("="*60)
    
    print(f"\nActual test duration: {total_runtime:.2f} seconds")
    print(f"Total API calls: {api_calls}")
    print(f"Total pool responses: {pools_retrieved}")
    print(f"Unique pools found: {len(unique_pools)}")
    
    if call_times:
        avg_call_time = sum(call_times) / len(call_times)
        print(f"Average call time: {avg_call_time:.2f} seconds")
        print(f"Fastest call: {min(call_times):.2f} seconds")
        print(f"Slowest call: {max(call_times):.2f} seconds")
        print(f"API calls per second: {api_calls / total_runtime:.2f}")
        print(f"Pools retrieved per second: {pools_retrieved / total_runtime:.2f}")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for i, error in enumerate(errors[:3], 1):
            print(f"  {i}. {error}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more")
    
    # Token pair analysis
    token_pair_counts = Counter(token_pairs)
    dex_counts = Counter(dexes)
    
    if token_pair_counts:
        print("\nMOST COMMON TOKEN PAIRS:")
        for pair, count in token_pair_counts.most_common(5):
            print(f"  {pair}: {count} occurrences")
    
    if dex_counts:
        print("\nDEXES DISTRIBUTION:")
        for dex, count in dex_counts.most_common():
            print(f"  {dex}: {count} occurrences")
    
    print("\n" + "="*60)
    print(f"Test completed at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    # Start the test
    run_60_second_test()