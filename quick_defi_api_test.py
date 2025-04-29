"""
Quick test script to retrieve DeFi API data within 10 seconds
"""
import os
import json
import requests
import time
import concurrent.futures
from typing import Dict, Any, List, Optional

# Configure API settings
API_KEY = os.getenv("DEFI_API_KEY")
BASE_URL = "https://filotdefiapi.replit.app/api/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def fetch_pools(batch_size: int = 10, max_batches: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch pools from the DeFi API in parallel
    
    Args:
        batch_size: Number of pools per batch
        max_batches: Maximum number of batches to fetch
        
    Returns:
        List of pool data
    """
    print(f"Fetching pools in {max_batches} batches of {batch_size}...")
    start_time = time.time()
    
    all_pools = []
    
    # Create an offset list for parallel requests
    offsets = [batch_size * i for i in range(max_batches)]
    
    def fetch_batch(offset: int) -> List[Dict[str, Any]]:
        """Fetch a batch of pools starting at the given offset"""
        url = f"{BASE_URL}/pools"
        params = {
            "limit": batch_size,
            "offset": offset
        }
        
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=5)
            if response.status_code == 200:
                return response.json() if isinstance(response.json(), list) else []
            return []
        except Exception:
            return []
    
    # Use ThreadPoolExecutor for parallel requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_batches) as executor:
        # Submit all batch requests at once
        future_to_offset = {executor.submit(fetch_batch, offset): offset for offset in offsets}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_offset):
            offset = future_to_offset[future]
            try:
                batch_pools = future.result()
                if batch_pools:
                    print(f"  Retrieved {len(batch_pools)} pools from offset {offset}")
                    all_pools.extend(batch_pools)
            except Exception as e:
                print(f"  Error fetching batch at offset {offset}: {str(e)}")
    
    duration = time.time() - start_time
    print(f"Retrieved {len(all_pools)} pools in {duration:.2f} seconds")
    return all_pools

def fetch_tokens(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch tokens from the DeFi API
    
    Args:
        limit: Maximum number of tokens to retrieve
        
    Returns:
        List of token data
    """
    print(f"\nFetching up to {limit} tokens...")
    start_time = time.time()
    
    try:
        url = f"{BASE_URL}/tokens"
        params = {"limit": limit}
        
        response = requests.get(url, headers=HEADERS, params=params, timeout=3)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, list):
                    duration = time.time() - start_time
                    print(f"Retrieved {len(data)} tokens in {duration:.2f} seconds")
                    return data[:limit]
                else:
                    print(f"Unexpected response format: {type(data)}")
                    return []
            except Exception as e:
                print(f"Error processing token response: {str(e)}")
                return []
        else:
            print(f"Error response: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error fetching tokens: {str(e)}")
        return []

def summarize_pool_data(pools: List[Dict[str, Any]]) -> None:
    """Provide a quick summary of pool data"""
    if not pools:
        print("No pool data to summarize")
        return
    
    print(f"\n=== Pool Data Summary ({len(pools)} pools) ===")
    
    # Count sources (DEXes)
    sources = {}
    active_count = 0
    total_tvl = 0
    tvl_count = 0
    total_apy = 0
    apy_count = 0
    
    # Token tracking
    tokens_seen = set()
    token_pairs = []
    
    for pool in pools:
        # Count sources
        source = pool.get("source")
        if source:
            sources[source] = sources.get(source, 0) + 1
            
        # Count active pools
        if pool.get("active"):
            active_count += 1
            
        # Sum TVL and APY
        metrics = pool.get("metrics", {})
        if metrics:
            tvl = metrics.get("tvl")
            if tvl is not None:
                total_tvl += tvl
                tvl_count += 1
                
            apy = metrics.get("apy24h")
            if apy is not None:
                total_apy += apy
                apy_count += 1
        
        # Extract token information
        tokens = pool.get("tokens", [])
        pair = []
        for token in tokens:
            if isinstance(token, dict):
                symbol = token.get("symbol")
                if symbol:
                    tokens_seen.add(symbol)
                    pair.append(symbol)
        
        if len(pair) == 2:
            token_pairs.append(f"{pair[0]}-{pair[1]}")
    
    # Display summary
    print("\nPool Sources (DEXes):")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}")
    
    print(f"\nActive Pools: {active_count}/{len(pools)}")
    
    if tvl_count > 0:
        avg_tvl = total_tvl / tvl_count
        print(f"\nAverage TVL: ${avg_tvl:,.2f}")
    
    if apy_count > 0:
        avg_apy = total_apy / apy_count
        print(f"\nAverage APY (24h): {avg_apy:.2f}%")
    
    print(f"\nUnique Tokens: {len(tokens_seen)}")
    
    # Most common pairs (if we have any)
    if token_pairs:
        pair_counts = {}
        for pair in token_pairs:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
        print("\nMost Common Token Pairs:")
        for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {pair}: {count}")

def summarize_token_data(tokens: List[Dict[str, Any]]) -> None:
    """Provide a quick summary of token data"""
    if not tokens:
        print("No token data to summarize")
        return
    
    print(f"\n=== Token Data Summary ({len(tokens)} tokens) ===")
    
    # Print some example tokens
    print("\nSample Tokens:")
    for i, token in enumerate(tokens[:5]):
        symbol = token.get("symbol", "Unknown")
        name = token.get("name", "Unknown")
        address = token.get("address", "Unknown")
        price = token.get("price", 0)
        print(f"  {i+1}. {symbol} ({name}) - ${price} - {address[:8]}...")

def run_quick_test():
    """Run a quick test of the DeFi API connection"""
    if not API_KEY:
        print("Error: No DEFI_API_KEY found in environment variables")
        return False
    
    print(f"API key is set with length: {len(API_KEY)}")
    
    # Set the start time to enforce a 10-second limit
    start_time = time.time()
    
    # Fetch pools (large batch size, parallel requests)
    pools = fetch_pools(batch_size=15, max_batches=4)
    
    # Check if we still have time to fetch tokens
    if time.time() - start_time < 8:  # Leave 2 seconds for token processing
        tokens = fetch_tokens(20)
    else:
        print("\nSkipping token fetch to stay within time limit")
        tokens = []
    
    # Summarize the data
    summarize_pool_data(pools)
    summarize_token_data(tokens)
    
    # Check if we exceeded our time limit
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Save a sample of the data
    if pools:
        with open("quick_pool_sample.json", "w") as f:
            json.dump(pools[:5], f, indent=2)
        print("\nSaved sample pool data to quick_pool_sample.json")
    
    return True

if __name__ == "__main__":
    run_quick_test()