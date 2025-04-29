"""
Test the raw response from the DeFi API to see why tokens are not being retrieved
"""

import os
import json
import requests
import time
from defi_aggregation_api import DefiAggregationAPI
import pprint

# Initialize the API client
api = DefiAggregationAPI()

# Get the raw API base URL and headers
base_url = api.base_url
headers = api.headers

# Make a direct API request to get raw pool data
url = f"{base_url}/pools"
params = {"limit": 1}

start_time = time.time()
print(f"Making API request to {url}")
response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    # Parse and print response
    data = response.json()
    print("\nAPI Response Time:", time.time() - start_time, "seconds")
    print("\nRESPONSE DATA STRUCTURE:")
    if isinstance(data, list) and len(data) > 0:
        # Print structure of first item
        first_pool = data[0]
        print("\nFirst pool keys:", list(first_pool.keys()))
        
        # Check for tokens field
        if 'tokens' in first_pool:
            print("\nTOKENS FIELD CONTENT:")
            pprint.pprint(first_pool['tokens'])
            
            # Check token structure
            if isinstance(first_pool['tokens'], list):
                for i, token in enumerate(first_pool['tokens']):
                    print(f"\nToken {i+1} structure:")
                    pprint.pprint(token)
        else:
            print("\nNo 'tokens' field found in the API response!")
            
        # Check the actual name field
        if 'name' in first_pool:
            print("\nPool name:", first_pool['name'])
            # Try to parse token names from the pool name
            name_parts = first_pool['name'].split('-')
            if len(name_parts) >= 2:
                print(f"Potential token symbols from name: {name_parts[0]} and {name_parts[1]}")
    else:
        print("API returned empty or non-list response")
        print("Raw response:", data)
else:
    print(f"API request failed with status code: {response.status_code}")
    print("Response content:", response.text)

# Now check how the transformation is happening in our app
print("\nCHECKING TRANSFORMATION FUNCTION:")

# Get one pool using our API client
pools = api.get_pools(limit=1)
if pools and len(pools) > 0:
    # Get the first pool
    raw_pool = pools[0]
    
    # Transform using our function
    transformed = api.transform_pool_data(raw_pool)
    
    # Check token fields in transformed data
    print("\nTransformed Pool data:")
    print(f"token1_symbol: {transformed.get('token1_symbol', 'Not found')}")
    print(f"token2_symbol: {transformed.get('token2_symbol', 'Not found')}")
    print(f"token1_address: {transformed.get('token1_address', 'Not found')}")
    print(f"token2_address: {transformed.get('token2_address', 'Not found')}")
    
    # If we still have issues, try to extract token info manually
    if 'name' in raw_pool:
        name = raw_pool['name']
        print(f"\nPool name: {name}")
        
        # Try to parse token names
        if '-' in name:
            symbols = name.split('-')
            if len(symbols) >= 2:
                token1 = symbols[0].strip()
                token2 = symbols[1].split(' ')[0].strip()
                print(f"Extracted symbols from name: {token1} and {token2}")
    
    # Check tokens array directly
    if 'tokens' in raw_pool:
        tokens = raw_pool['tokens']
        print(f"\nRaw tokens: {tokens}")
else:
    print("Failed to get pool data from the API client")