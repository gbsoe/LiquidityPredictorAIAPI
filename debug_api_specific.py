"""
Debugging script specifically for the API authentication issue
"""
import requests
import json
import time

# Constants
API_KEY = "9feae0d0af47e4948e061f2d7820461e374e040c21cf65c087166d7ed18f5ed6"
BASE_URL = "https://raydium-trader-filot.replit.app"

def test_api_headers():
    """Test different API header variations with specific debugging"""
    
    # Standard headers used in our application
    headers_to_test = [
        {
            "name": "X-API-KEY all uppercase",
            "headers": {
                "X-API-KEY": API_KEY,
                "Content-Type": "application/json"
            }
        },
        {
            "name": "x-api-key all lowercase",
            "headers": {
                "x-api-key": API_KEY,
                "Content-Type": "application/json"
            }
        }
    ]
    
    url = f"{BASE_URL}/health"
    
    print(f"Testing health endpoint with different headers: {url}")
    
    for header_config in headers_to_test:
        print(f"\nUsing {header_config['name']}:")
        
        try:
            print(f"Headers: {header_config['headers']}")
            response = requests.get(url, headers=header_config['headers'], timeout=10)
            
            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                print("✅ SUCCESS!")
                try:
                    data = response.json()
                    print(f"Response: {json.dumps(data, indent=2)}")
                except:
                    print(f"Response (text): {response.text}")
            else:
                print("❌ FAILED")
                try:
                    error = response.json()
                    print(f"Error: {json.dumps(error, indent=2)}")
                except:
                    print(f"Error (text): {response.text}")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
    
    # Now test actual API endpoint
    url = f"{BASE_URL}/api/pools"
    
    print(f"\nTesting API pools endpoint with different headers: {url}")
    
    for header_config in headers_to_test:
        print(f"\nUsing {header_config['name']}:")
        
        try:
            print(f"Headers: {header_config['headers']}")
            response = requests.get(url, headers=header_config['headers'], timeout=10)
            
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
                                    sample_pool = pools[0]
                                    print(f"   Sample pool keys: {list(sample_pool.keys())}")
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

def check_url_connectivity():
    """Basic connectivity check to the API host"""
    print(f"\nChecking basic connectivity to {BASE_URL}...")
    
    try:
        # No headers to test DNS resolution and routing
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Basic connectivity: {'✅ SUCCESS' if response.status_code == 200 else '❌ FAILED'}")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text[:100]}")
    except Exception as e:
        print(f"❌ FAILED to connect: {str(e)}")

if __name__ == "__main__":
    print("=== API AUTHENTICATION DEBUGGING ===")
    check_url_connectivity()
    test_api_headers()
    print("\n=== DEBUG COMPLETE ===")