#!/usr/bin/env python3
"""
Script to examine the DeFi API response structure
"""

import os
import json
import requests
from pprint import pprint

def main():
    # Get API key from environment
    api_key = os.getenv("DEFI_API_KEY")
    if not api_key:
        print("Error: DEFI_API_KEY environment variable is not set")
        return 1
    
    # Make request to API
    url = "https://filotdefiapi.replit.app/api/v1/pools"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    params = {
        "limit": 3  # Just get a few pools for examination
    }
    
    print("Making API request...")
    response = requests.get(url, headers=headers, params=params, timeout=15)
    
    # Check if request was successful
    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        print(f"Response text: {response.text}")
        return 1
    
    # Parse JSON response
    data = response.json()
    
    # Count the number of pools
    if isinstance(data, list):
        pool_count = len(data)
        first_pool = data[0] if data else None
    elif isinstance(data, dict) and "pools" in data:
        pool_count = len(data["pools"])
        first_pool = data["pools"][0] if data["pools"] else None
    else:
        pool_count = 0
        first_pool = None
    
    print(f"Retrieved {pool_count} pools from API")
    
    if first_pool:
        print("\n=== First Pool Structure ===")
        print(f"Pool ID: {first_pool.get('poolId', 'N/A')}")
        print(f"Name: {first_pool.get('name', 'N/A')}")
        print(f"Source (DEX): {first_pool.get('source', 'N/A')}")
        
        # Dump the full structure of the first pool for inspection
        print("\n=== Full Structure (First Pool) ===")
        print(json.dumps(first_pool, indent=2))
        
        # Specifically look for liquidity/TVL and APR fields
        print("\n=== Key Fields ===")
        print(f"tvl field: {first_pool.get('tvl', 'Not found')}")
        print(f"metrics.tvl field: {first_pool.get('metrics', {}).get('tvl', 'Not found')}")
        print(f"apr24h field: {first_pool.get('apr24h', 'Not found')}")
        print(f"metrics.apr24h field: {first_pool.get('metrics', {}).get('apr24h', 'Not found')}")
        print(f"apy.24h field: {first_pool.get('apy', {}).get('24h', 'Not found')}")
        
        # List all available keys at the top level
        print("\n=== Available Keys ===")
        print(f"Top-level keys: {list(first_pool.keys())}")
        if 'metrics' in first_pool:
            print(f"Metrics-level keys: {list(first_pool['metrics'].keys())}")
        
        # Save sample JSON for future reference
        with open("api_response_sample.json", "w") as f:
            json.dump(data, f, indent=2)
            print("\nSaved sample response to api_response_sample.json")
    
    return 0

if __name__ == "__main__":
    exit(main())