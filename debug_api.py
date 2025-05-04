import requests
import json
import time

# API Configuration
API_KEY = "9feae0d0af47e4948e061f2d7820461e374e040c21cf65c087166d7ed18f5ed6"
BASE_URL = "https://raydium-trader-filot.replit.app"

def test_api_pools_endpoint():
    """Test the pools endpoint with various header configurations"""
    
    # Test different header formats
    header_formats = [
        {
            "name": "Standard Header (lowercase)",
            "headers": {
                "x-api-key": API_KEY,
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Uppercase Header",
            "headers": {
                "X-API-KEY": API_KEY,
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Mixed Case Header",
            "headers": {
                "X-Api-Key": API_KEY,
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Authorization Bearer",
            "headers": {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Both Authorization Bearer and X-API-KEY",
            "headers": {
                "Authorization": f"Bearer {API_KEY}",
                "X-API-KEY": API_KEY,
                "Content-Type": "application/json"
            }
        }
    ]
    
    for header_config in header_formats:
        print(f"\nTesting with {header_config['name']}:")
        url = f"{BASE_URL}/api/pools"
        headers = header_config["headers"]
        
        try:
            print(f"Request URL: {url}")
            print(f"Headers: {headers}")
            
            response = requests.get(url, headers=headers)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ SUCCESS!")
                data = response.json()
                # Print a sample of the response to verify data
                print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")
                if isinstance(data, dict) and "pools" in data:
                    pools_data = data["pools"]
                    if isinstance(pools_data, dict):
                        print(f"Pools data contains: {list(pools_data.keys())}")
                    elif isinstance(pools_data, list):
                        print(f"Pools data is a list with {len(pools_data)} items")
                        if pools_data:
                            print(f"First pool: {json.dumps(pools_data[0], indent=2)[:200]}...")
            else:
                print("❌ FAILED")
                try:
                    error_data = response.json()
                    print(f"Error response: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Raw response: {response.text}")
        
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
        
        # Add a delay between tests
        time.sleep(1)

def test_nested_endpoints():
    """Test specific pool endpoint structures"""
    
    # Try standard header format (most likely to work based on test_api_auth.py results)
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }
    
    endpoints = [
        "/api/pools",
        "/api/pools?limit=5",
        "/api/tokens",
        "/api/dexes",
        "/health"
    ]
    
    for endpoint in endpoints:
        print(f"\nTesting endpoint: {endpoint}")
        
        url = f"{BASE_URL}{endpoint}"
        
        try:
            print(f"Request URL: {url}")
            response = requests.get(url, headers=headers)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ SUCCESS!")
                try:
                    data = response.json()
                    print(f"Response structure: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")
                except:
                    print(f"Response (non-JSON): {response.text[:200]}")
            else:
                print("❌ FAILED")
                try:
                    error_data = response.json()
                    print(f"Error response: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Raw response: {response.text}")
        
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
        
        # Add a delay between tests
        time.sleep(1)

if __name__ == "__main__":
    print("===== API DEBUG SCRIPT =====")
    print(f"Testing API at {BASE_URL}")
    print(f"Using API key: {API_KEY[:5]}...{API_KEY[-5:]}")
    
    print("\n----- Testing Pool Endpoints with Different Headers -----")
    test_api_pools_endpoint()
    
    print("\n----- Testing Various API Endpoints -----")
    test_nested_endpoints()
    
    print("\nDebug complete!")