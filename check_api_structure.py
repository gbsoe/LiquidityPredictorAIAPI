"""
Check the structure of the DeFi API response 
"""
import json
import requests

# API Key and Base URL
API_KEY = "defi_WyJ71mVrIDzEkzwauPu_FpnRh__W83_l"
BASE_URL = "https://filotdefiapi.replit.app/api/v1"

# Headers with Bearer token
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

def main():
    # Get pools
    print("Fetching pools from API...")
    response = requests.get(f"{BASE_URL}/pools", headers=HEADERS)
    
    print(f"Response status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response type: {type(data)}")
        
        if len(data) > 0:
            print("\nFirst pool structure:")
            first_pool = data[0]
            print(f"Keys: {list(first_pool.keys())}")
            
            # Print the full structure of the first pool with sample values
            print("\nDetailed structure with sample values:")
            for key, value in first_pool.items():
                if isinstance(value, dict):
                    print(f"{key}: {type(value)} with keys {list(value.keys())}")
                elif isinstance(value, list):
                    if len(value) > 0:
                        print(f"{key}: List with {len(value)} items, first item type: {type(value[0])}")
                        if isinstance(value[0], dict):
                            print(f"  First item keys: {list(value[0].keys())}")
                    else:
                        print(f"{key}: Empty list")
                else:
                    print(f"{key}: {type(value)} = {value}")
            
            # Check for tokens specifically
            print("\nToken structure:")
            if 'tokens' in first_pool:
                tokens = first_pool['tokens']
                print(f"Tokens type: {type(tokens)}")
                if isinstance(tokens, list) and len(tokens) > 0:
                    print(f"First token keys: {list(tokens[0].keys())}")
                    print(f"First token values: {tokens[0]}")
            else:
                print("No 'tokens' field found")
                
            # Check for metrics
            print("\nMetrics structure:")
            if 'metrics' in first_pool:
                metrics = first_pool['metrics']
                print(f"Metrics type: {type(metrics)}")
                if isinstance(metrics, dict):
                    print(f"Metrics keys: {list(metrics.keys())}")
                    print(f"Sample metric values: {metrics}")
            else:
                print("No 'metrics' field found")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    main()