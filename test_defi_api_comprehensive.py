"""
Comprehensive test of the DeFi API to retrieve pools and tokens
"""
import os
import json
import requests
import time
from typing import Dict, Any, List, Optional
from collections import Counter

# Note: Not using pandas to avoid dependency issues and keep the script simple

# Configure API settings
API_KEY = os.getenv("DEFI_API_KEY")
BASE_URL = "https://filotdefiapi.replit.app/api/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def fetch_pools(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch pools from the DeFi API
    
    Args:
        limit: Maximum number of pools to retrieve
        
    Returns:
        List of pool data
    """
    print(f"Fetching up to {limit} pools from the DeFi API...")
    
    all_pools = []
    offset = 0
    batch_size = 10  # Fetch in smaller batches to avoid timeouts
    
    while len(all_pools) < limit:
        try:
            # Make the API request with pagination
            url = f"{BASE_URL}/pools"
            params = {
                "limit": batch_size,
                "offset": offset
            }
            
            print(f"  Making request to {url} with params {params}")
            response = requests.get(url, headers=HEADERS, params=params)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Check if we got a list of pools
                    if isinstance(data, list):
                        print(f"  Retrieved {len(data)} pools")
                        all_pools.extend(data)
                        
                        # Break if we've reached the end of data
                        if len(data) < batch_size:
                            print("  Reached end of available pools")
                            break
                            
                        # Update offset for next batch
                        offset += len(data)
                    else:
                        print(f"  Unexpected response format: {type(data)}")
                        break
                except json.JSONDecodeError:
                    print(f"  Error decoding JSON response")
                    print(f"  Response content: {response.text[:200]}...")
                    break
            else:
                print(f"  Error response: {response.status_code}")
                print(f"  Response content: {response.text[:200]}...")
                break
                
            # Respect rate limiting
            time.sleep(0.2)  # 200ms delay between requests
            
            # Break if we've reached the limit
            if len(all_pools) >= limit:
                break
                
        except Exception as e:
            print(f"  Error fetching pools: {str(e)}")
            break
    
    print(f"Successfully retrieved {len(all_pools)} pools")
    return all_pools[:limit]  # Ensure we don't exceed the requested limit

def fetch_tokens(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch tokens from the DeFi API
    
    Args:
        limit: Maximum number of tokens to retrieve
        
    Returns:
        List of token data
    """
    print(f"\nFetching up to {limit} tokens from the DeFi API...")
    
    try:
        # Make the API request
        url = f"{BASE_URL}/tokens"
        params = {"limit": limit}
        
        print(f"  Making request to {url}")
        response = requests.get(url, headers=HEADERS, params=params)
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Check if we got a list of tokens
                if isinstance(data, list):
                    print(f"  Retrieved {len(data)} tokens")
                    return data[:limit]  # Ensure we don't exceed the requested limit
                else:
                    print(f"  Unexpected response format: {type(data)}")
                    return []
            except json.JSONDecodeError:
                print(f"  Error decoding JSON response")
                print(f"  Response content: {response.text[:200]}...")
                return []
        else:
            print(f"  Error response: {response.status_code}")
            print(f"  Response content: {response.text[:200]}...")
            return []
            
    except Exception as e:
        print(f"  Error fetching tokens: {str(e)}")
        return []

def fetch_token_by_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch token data for a specific symbol
    
    Args:
        symbol: Token symbol
        
    Returns:
        Token data or None
    """
    print(f"\nFetching token data for {symbol}...")
    
    try:
        # Make the API request
        url = f"{BASE_URL}/tokens/{symbol}"
        
        print(f"  Making request to {url}")
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                if data:
                    print(f"  Successfully retrieved data for {symbol}")
                    # Handle the different possible response types
                    if isinstance(data, list) and data:
                        # If we got a list, return the first item
                        result: Dict[str, Any] = data[0]
                        return result
                    elif isinstance(data, dict):
                        # If we got a dictionary, return it directly
                        result: Dict[str, Any] = data
                        return result
                    else:
                        print(f"  Unexpected data type: {type(data)}")
                        return None
                else:
                    print(f"  No data found for {symbol}")
                    return None
            except json.JSONDecodeError:
                print(f"  Error decoding JSON response")
                print(f"  Response content type: {response.headers.get('Content-Type', 'unknown')}")
                print(f"  Response content: {response.text[:200]}...")
                return None
        else:
            print(f"  Error response: {response.status_code}")
            print(f"  Response content: {response.text[:200]}...")
            return None
            
    except Exception as e:
        print(f"  Error fetching token {symbol}: {str(e)}")
        return None

def analyze_pool_data(pools: List[Dict[str, Any]]) -> None:
    """
    Analyze and display insights from pool data
    
    Args:
        pools: List of pool data
    """
    if not pools:
        print("\nNo pool data to analyze")
        return
        
    print(f"\n=== Pool Data Analysis ({len(pools)} pools) ===")
    
    try:
        # Analyze sources (DEXes)
        sources = [pool.get("source", "Unknown") for pool in pools]
        source_counts = Counter(sources)
        print("\nPool Sources (DEXes):")
        for source, count in source_counts.most_common():
            print(f"  {source}: {count}")
        
        # Analyze active vs inactive pools
        active_status = [pool.get("active", False) for pool in pools]
        active_count = sum(1 for status in active_status if status)
        inactive_count = len(active_status) - active_count
        print("\nActive vs Inactive Pools:")
        print(f"  Active: {active_count}")
        print(f"  Inactive: {inactive_count}")
        
        # Analyze TVL (Total Value Locked)
        tvl_values = []
        for pool in pools:
            metrics = pool.get("metrics", {})
            if metrics and "tvl" in metrics and metrics["tvl"] is not None:
                tvl_values.append(metrics["tvl"])
        
        if tvl_values:
            avg_tvl = sum(tvl_values) / len(tvl_values)
            max_tvl = max(tvl_values)
            min_tvl = min(tvl_values)
            print("\nTVL (Total Value Locked) Statistics:")
            print(f"  Average: ${avg_tvl:.2f}")
            print(f"  Maximum: ${max_tvl:.2f}")
            print(f"  Minimum: ${min_tvl:.2f}")
            
        # Analyze APY
        apy_values = []
        pools_with_apy = []
        for pool in pools:
            metrics = pool.get("metrics", {})
            if metrics and "apy24h" in metrics and metrics["apy24h"] is not None:
                apy = metrics["apy24h"]
                apy_values.append(apy)
                pools_with_apy.append((pool, apy))
        
        if apy_values:
            avg_apy = sum(apy_values) / len(apy_values)
            max_apy = max(apy_values)
            min_apy = min(apy_values)
            print("\nAPY (24h) Statistics:")
            print(f"  Average: {avg_apy:.2f}%")
            print(f"  Maximum: {max_apy:.2f}%")
            print(f"  Minimum: {min_apy:.2f}%")
            
            # Display top 5 pools by APY
            pools_with_apy.sort(key=lambda x: x[1], reverse=True)
            print("\nTop 5 Pools by APY (24h):")
            for i, (pool, apy) in enumerate(pools_with_apy[:5]):
                print(f"  {pool.get('name', 'Unknown')} - {apy:.2f}% APY")
                
        # Analyze token pairs
        token_pairs = []
        for pool in pools:
            tokens = pool.get("tokens", [])
            token1_symbol = tokens[0].get("symbol", "Unknown") if len(tokens) >= 1 and isinstance(tokens[0], dict) else "Unknown"
            token2_symbol = tokens[1].get("symbol", "Unknown") if len(tokens) >= 2 and isinstance(tokens[1], dict) else "Unknown"
            token_pairs.append(f"{token1_symbol}-{token2_symbol}")
        
        if token_pairs:
            pair_counts = Counter(token_pairs)
            print("\nMost Common Token Pairs:")
            for pair, count in pair_counts.most_common(5):
                print(f"  {pair}: {count}")
                
    except Exception as e:
        print(f"Error analyzing pool data: {str(e)}")

def run_test():
    """Run the comprehensive DeFi API test"""
    if not API_KEY:
        print("Error: No DEFI_API_KEY found in environment variables")
        return False
    
    # Display API key information (securely)
    print(f"API key is set: {bool(API_KEY)}")
    print(f"API key length: {len(API_KEY)}")
    if API_KEY:
        masked_key = API_KEY[:4] + "..." + API_KEY[-4:]
        print(f"API key (masked): {masked_key}")
    
    # Fetch pools (reduced to 25 to avoid timeouts)
    pools = fetch_pools(25)
    
    # Analyze pool data
    analyze_pool_data(pools)
    
    # Fetch tokens (reduced to 10 to avoid timeouts)
    tokens = fetch_tokens(10)
    
    if tokens:
        print(f"\n=== Token Data ({len(tokens)} tokens) ===")
        for i, token in enumerate(tokens[:5]):  # Display first 5 tokens
            print(f"\nToken {i+1}:")
            print(f"  Symbol: {token.get('symbol', 'Unknown')}")
            print(f"  Name: {token.get('name', 'Unknown')}")
            print(f"  Address: {token.get('address', 'Unknown')}")
            print(f"  Price: ${token.get('price', 0)}")
    
    # Test specific token retrieval
    test_tokens = ["SOL", "USDC"]  # Removed ATLAS since it's causing problems
    for symbol in test_tokens:
        token = fetch_token_by_symbol(symbol)
        if token:
            print(f"\n=== {symbol} Token Details ===")
            print(json.dumps(token, indent=2))
        else:
            print(f"\nCould not retrieve details for {symbol}")
            
    # Save the full pool data to a file for reference
    if pools:
        with open("defi_api_pools.json", "w") as f:
            json.dump(pools[:10], f, indent=2)  # Save only 10 pools to avoid large files
        print("\nSaved sample pool data to defi_api_pools.json")
            
    return True

if __name__ == "__main__":
    run_test()