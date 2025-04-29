"""
Simple API test that makes a single call and prints results
"""

from defi_aggregation_api import DefiAggregationAPI
import time
import json
from collections import Counter

def run_simple_test():
    """Make a single API call and print detailed results"""
    print("Simple DeFi API Test - Single Call")
    print("=" * 50)
    
    # Init API
    api = DefiAggregationAPI()
    print(f"Using API base URL: {api.base_url}")
    
    # Single API call
    start = time.time()
    pools = api._make_request("pools", params={"limit": 20, "offset": 0})
    duration = time.time() - start
    
    if not pools or not isinstance(pools, list):
        print(f"Error: Invalid response: {pools}")
        return
    
    # Basic stats
    pool_count = len(pools)
    
    print(f"\nReceived {pool_count} pools in {duration:.2f} seconds")
    print(f"Average time per pool: {duration/pool_count:.2f} seconds")
    
    # Analyze token distribution
    token_pairs = []
    dexes = []
    
    print("\nPool Details:")
    print("-" * 50)
    
    for i, pool in enumerate(pools[:5], 1):  # Print details for first 5 pools
        pool_id = pool.get('poolId', 'Unknown')
        name = pool.get('name', 'Unknown')
        dex = pool.get('source', 'Unknown')
        
        print(f"\n{i}. {name}")
        print(f"   DEX: {dex}")
        print(f"   ID: {pool_id}")
        
        # Track DEX
        if dex != "Unknown":
            dexes.append(dex)
        
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
        tokens = []
        if name and '-' in name:
            parts = name.split('-')
            if len(parts) >= 2:
                token1 = parts[0].strip()
                token2_parts = parts[1].split(' ')
                token2 = token2_parts[0].strip()
                
                tokens = [token1, token2]
                token_pair = f"{token1}-{token2}"
                token_pairs.append(token_pair)
                
                print(f"   Tokens: {token1} and {token2}")
    
    # Count remaining pools
    if pool_count > 5:
        print(f"\n... and {pool_count - 5} more pools")
    
    # Token analysis
    all_tokens = []
    for pool in pools:
        name = pool.get('name', '')
        if name and '-' in name:
            parts = name.split('-')
            if len(parts) >= 2:
                token1 = parts[0].strip()
                token2_parts = parts[1].split(' ')
                token2 = token2_parts[0].strip()
                all_tokens.extend([token1, token2])
    
    token_counts = Counter(all_tokens)
    
    print("\nToken Distribution:")
    print("-" * 50)
    for token, count in token_counts.most_common(10):
        print(f"{token}: {count} occurrences")
    
    # DEX distribution
    dex_counts = Counter(dexes)
    print("\nDEX Distribution:")
    print("-" * 50)
    for dex, count in dex_counts.most_common():
        print(f"{dex}: {count} pools")
    
    # Save a copy of the raw data for reference
    try:
        with open("api_response_sample.json", "w") as f:
            json.dump(pools[:2], f, indent=2)  # Save just 2 pools to keep the file small
        print("\nSample response data saved to api_response_sample.json")
    except Exception as e:
        print(f"Could not save sample: {str(e)}")
    
    print("\nTest completed successfully.")

if __name__ == "__main__":
    run_simple_test()