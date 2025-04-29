"""
Run a 30-second test of the DeFi API with summary output
"""

import time
import threading
from defi_aggregation_api import DefiAggregationAPI
from datetime import datetime
from collections import Counter

def run_api_calls(api, results, stop_event):
    """Run API calls in a thread until the stop event is set"""
    while not stop_event.is_set():
        try:
            # Make API call
            results["api_calls"] += 1
            call_start = time.time()
            
            # Get pools data
            pools = api._make_request("pools", params={"limit": 20, "offset": 0})
            call_time = time.time() - call_start
            results["call_times"].append(call_time)
            
            if pools and isinstance(pools, list):
                pool_count = len(pools)
                results["total_pools"] += pool_count
                print(f"✅ Call #{results['api_calls']}: Received {pool_count} pools in {call_time:.2f}s")
                
                # Process pool tokens (just from the first pool to keep it simple)
                if pools:
                    pool = pools[0]
                    pool_id = pool.get('poolId', '')
                    results["unique_pools"].add(pool_id)
                    
                    # Get pool name and DEX
                    name = pool.get('name', 'Unknown')
                    dex = pool.get('source', 'Unknown')
                    
                    # Extract token info from name
                    if name and '-' in name:
                        parts = name.split('-')
                        if len(parts) >= 2:
                            token1 = parts[0].strip()
                            token2_parts = parts[1].split(' ')
                            token2 = token2_parts[0].strip()
                            
                            # Track token pair
                            token_pair = f"{token1}-{token2}"
                            results["token_pairs"].append(token_pair)
                    
                    # Track DEX
                    if dex != "Unknown":
                        results["dexes"].append(dex)
            else:
                print(f"❌ Call #{results['api_calls']}: Invalid response")
                results["errors"].append(f"Invalid response in call {results['api_calls']}")
            
            # Brief pause to avoid hammering the API
            time.sleep(0.1)
            
        except Exception as e:
            error_msg = f"Error in API call {results['api_calls']}: {str(e)}"
            print(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            time.sleep(0.5)

def run_test(duration_seconds=30):
    """Run the test for the specified duration"""
    print(f"Starting {duration_seconds}-second DeFi API test...")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Initialize API client
    api = DefiAggregationAPI()
    
    # Initialize results
    results = {
        "start_time": time.time(),
        "api_calls": 0,
        "total_pools": 0,
        "unique_pools": set(),
        "token_pairs": [],
        "dexes": [],
        "call_times": [],
        "errors": []
    }
    
    # Run test in a separate thread so we can control timing precisely
    stop_event = threading.Event()
    api_thread = threading.Thread(
        target=run_api_calls, 
        args=(api, results, stop_event)
    )
    api_thread.daemon = True
    
    # Start the test
    print("Beginning API calls...\n")
    api_thread.start()
    
    # Wait for the duration
    time.sleep(duration_seconds)
    
    # Stop the test
    stop_event.set()
    
    # Give the thread time to finish
    api_thread.join(2)
    
    # Calculate stats
    end_time = time.time()
    results["duration"] = end_time - results["start_time"]
    
    print("\n" + "="*60)
    print(f"{duration_seconds}-SECOND DeFi API TEST RESULTS")
    print("="*60)
    
    print(f"\nTest duration: {results['duration']:.2f} seconds")
    print(f"Total API calls: {results['api_calls']}")
    print(f"Total pools retrieved: {results['total_pools']}")
    print(f"Unique pools found: {len(results['unique_pools'])}")
    
    if results["call_times"]:
        avg_call_time = sum(results["call_times"]) / len(results["call_times"])
        print(f"Average call time: {avg_call_time:.2f} seconds")
        print(f"Fastest call: {min(results['call_times']):.2f} seconds")
        print(f"Slowest call: {max(results['call_times']):.2f} seconds")
        print(f"API calls per second: {results['api_calls'] / results['duration']:.2f}")
        print(f"Pools per second: {results['total_pools'] / results['duration']:.2f}")
    
    # Error summary
    if results["errors"]:
        print(f"\nEncountered {len(results['errors'])} errors:")
        for i, error in enumerate(results["errors"][:3], 1):
            print(f"  {i}. {error}")
        if len(results["errors"]) > 3:
            print(f"  ... and {len(results['errors']) - 3} more")
    
    # Token pair analysis
    token_pair_counts = Counter(results["token_pairs"])
    dex_counts = Counter(results["dexes"])
    
    print("\nTOP TOKEN PAIRS:")
    for pair, count in token_pair_counts.most_common(3):
        print(f"  {pair}: {count} occurrences")
    
    print("\nDEXES:")
    for dex, count in dex_counts.most_common():
        print(f"  {dex}: {count} pools")
    
    print("\n" + "="*60)
    print(f"Test completed at {datetime.now().strftime('%H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    run_test(30)  # Run for 30 seconds