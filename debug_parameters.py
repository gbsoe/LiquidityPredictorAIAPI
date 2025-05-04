"""
Debug script to test the API with different parameter combinations
"""
import requests
import json
import time

# Constants
API_KEY = "9feae0d0af47e4948e061f2d7820461e374e040c21cf65c087166d7ed18f5ed6"
BASE_URL = "https://raydium-trader-filot.replit.app"

def test_api_with_params():
    """Test the API with different parameter combinations"""
    
    # Standard headers that we know work based on our debug_api_specific.py
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }
    
    # Define parameter combinations to test
    param_sets = [
        {
            "name": "No parameters",
            "params": {}
        },
        {
            "name": "Just limit",
            "params": {"limit": 5}
        },
        {
            "name": "With raydium source",
            "params": {"source": "raydium"}
        },
        {
            "name": "With raydium dex",
            "params": {"dex": "raydium"}
        },
        {
            "name": "With type=amm",
            "params": {"type": "amm"}
        }
    ]
    
    url = f"{BASE_URL}/api/pools"
    
    print(f"Testing API with different parameter combinations: {url}")
    
    for param_set in param_sets:
        print(f"\nUsing {param_set['name']}:")
        
        try:
            print(f"Parameters: {param_set['params']}")
            response = requests.get(url, headers=headers, params=param_set['params'], timeout=10)
            
            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                print("✅ SUCCESS!")
                try:
                    data = response.json()
                    print(f"Response keys: {list(data.keys())}")
                    if "pools" in data:
                        pools_data = data["pools"]
                        if isinstance(pools_data, dict):
                            print(f"Pools data categories: {list(pools_data.keys())}")
                            for category, pools in pools_data.items():
                                if isinstance(pools, list) and pools:
                                    print(f" - {category}: {len(pools)} pools")
                        elif isinstance(pools_data, list):
                            print(f"Pools data is a list with {len(pools_data)} items")
                except:
                    print(f"Response (text): {response.text[:200]}")
            else:
                print("❌ FAILED")
                try:
                    error = response.json()
                    print(f"Error: {json.dumps(error, indent=2)}")
                except:
                    print(f"Error (text): {response.text}")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
        
        # Add a delay to avoid rate limiting
        time.sleep(1)

def test_tokens_endpoint():
    """Test the tokens endpoint specifically"""
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }
    
    url = f"{BASE_URL}/api/tokens"
    
    print(f"\nTesting tokens endpoint: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("✅ SUCCESS!")
            try:
                data = response.json()
                if isinstance(data, list):
                    print(f"Response is a list with {len(data)} tokens")
                    if data:
                        print(f"Sample token: {json.dumps(data[0], indent=2)}")
                else:
                    print(f"Response keys: {list(data.keys())}")
            except:
                print(f"Response (text): {response.text[:200]}")
        else:
            print("❌ FAILED")
            try:
                error = response.json()
                print(f"Error: {json.dumps(error, indent=2)}")
            except:
                print(f"Error (text): {response.text}")
    except Exception as e:
        print(f"Exception occurred: {str(e)}")

if __name__ == "__main__":
    print("=== API PARAMETER DEBUGGING ===")
    test_api_with_params()
    test_tokens_endpoint()
    print("\n=== DEBUG COMPLETE ===")