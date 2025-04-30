"""
Test script to fetch and display API data for diagnosis
"""

import os
import json
import requests
from datetime import datetime

# Define output file
OUTPUT_FILE = "api_test_results.md"

def make_request(endpoint, params=None):
    """Make a request to the DeFi API"""
    api_key = os.getenv("DEFI_API_KEY")
    base_url = "https://filotdefiapi.replit.app/api/v1"
    
    url = f"{base_url}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print(f"Making request to {url}")
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def write_to_file(content):
    """Write content to the output file"""
    with open(OUTPUT_FILE, "a") as f:
        f.write(content + "\n\n")

def main():
    """Run the API tests"""
    # Start fresh output file
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"# DeFi API Test Results\n\n")
        f.write(f"Test run at: {datetime.now().isoformat()}\n\n")

    # Test 1: Get tokens
    write_to_file("## Test 1: Get Tokens\n")
    tokens = make_request("tokens")
    
    if tokens:
        write_to_file(f"Retrieved {len(tokens)} tokens")
        write_to_file("```json\n" + json.dumps(tokens, indent=2) + "\n```")
    else:
        write_to_file("Failed to retrieve tokens")
        
    # Test 2: Get pools (first 2 pools only for brevity)
    write_to_file("## Test 2: Get Pools (First 2)\n")
    pools = make_request("pools", {"limit": 2})
    
    if pools:
        # Write raw JSON response
        write_to_file(f"Retrieved {len(pools)} pools")
        write_to_file("```json\n" + json.dumps(pools, indent=2) + "\n```")
        
        # Extract and display token information for each pool
        for i, pool in enumerate(pools):
            write_to_file(f"### Pool {i+1}: {pool.get('name', 'Unnamed')}")
            write_to_file(f"- ID: {pool.get('id')}")
            write_to_file(f"- Pool ID: {pool.get('poolId')}")
            write_to_file(f"- Source: {pool.get('source')}")
            
            tokens = pool.get('tokens', [])
            write_to_file(f"- Token count: {len(tokens)}")
            
            if tokens:
                write_to_file("- Tokens:")
                for token in tokens:
                    write_to_file(f"  - {token.get('symbol', 'Unknown')} ({token.get('address', 'No address')}): ${token.get('price', 0)}")
            else:
                write_to_file("- No tokens found in response")
    else:
        write_to_file("Failed to retrieve pools")
    
    # Test 3: Get a single pool
    write_to_file("## Test 3: Get Specific Pool\n")
    # Try to get a specific pool (using first pool ID from previous response)
    if pools and len(pools) > 0:
        pool_id = pools[0].get('poolId', '')
        write_to_file(f"Fetching pool with ID: {pool_id}")
        
        pool = make_request(f"pools/{pool_id}")
        if pool:
            write_to_file("```json\n" + json.dumps(pool, indent=2) + "\n```")
        else:
            write_to_file("Failed to retrieve specific pool")
    
    # Test 4: Test with hard-coded pool ID from the docs
    write_to_file("## Test 4: Get Pool with Hard-coded ID\n")
    hard_coded_id = "7UF3m8hDGZ6bNnHzaT2YHrhp7A7n9qFfBj6QEpHPv5S8"  # Example from docs
    write_to_file(f"Fetching pool with hard-coded ID: {hard_coded_id}")
    
    pool = make_request(f"pools/{hard_coded_id}")
    if pool:
        write_to_file("```json\n" + json.dumps(pool, indent=2) + "\n```")
    else:
        write_to_file("Failed to retrieve hard-coded pool")
        
    print(f"Tests completed. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()