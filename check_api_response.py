"""
Check the raw response from the DeFi API
"""
import json
import os
import requests

# Get API key from environment
api_key = os.getenv("DEFI_API_KEY")
if not api_key:
    print("Error: DEFI_API_KEY not found in environment variables")
    exit(1)
    
# Set up headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Make request
url = "https://filotdefiapi.replit.app/api/v1/pools"
params = {"limit": 2}

try:
    print(f"Sending request to {url} with API key {api_key[:5]}...")
    response = requests.get(url, headers=headers, params=params)
    
    print(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("\nRaw API Response:")
        print(json.dumps(data, indent=2))
        
        # Check the type and keys
        print(f"\nResponse type: {type(data)}")
        if isinstance(data, list):
            print(f"List length: {len(data)}")
            if data:
                print("\nFirst item keys:")
                print(json.dumps(list(data[0].keys()), indent=2))
                
                print("\nTokens structure:")
                tokens = data[0].get("tokens", [])
                print(json.dumps(tokens, indent=2))
        elif isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
    else:
        print(f"Error response: {response.text}")
except Exception as e:
    print(f"Exception: {str(e)}")