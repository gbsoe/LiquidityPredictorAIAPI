"""
Test script to verify the correct API authentication method
"""
import requests
import json

API_KEY = "9feae0d0af47e4948e061f2d7820461e374e040c21cf65c087166d7ed18f5ed6"
BASE_URL = "https://raydium-trader-filot.replit.app"

def test_auth_methods():
    """Test various authentication header formats"""
    
    # Test endpoints
    endpoints = [
        "/health",
        "/api/pools",
    ]
    
    # Authentication methods to test
    auth_methods = [
        {
            "name": "x-api-key lowercase",
            "headers": {
                "x-api-key": API_KEY,
                "Content-Type": "application/json"
            }
        },
        {
            "name": "X-API-KEY uppercase",
            "headers": {
                "X-API-KEY": API_KEY,
                "Content-Type": "application/json"
            }
        },
        {
            "name": "X-Api-Key mixed case",
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
            "name": "api-key dash format",
            "headers": {
                "api-key": API_KEY,
                "Content-Type": "application/json"
            }
        },
        {
            "name": "apikey no dash",
            "headers": {
                "apikey": API_KEY,
                "Content-Type": "application/json"
            }
        },
    ]
    
    # Test each endpoint with each auth method
    for endpoint in endpoints:
        print(f"\n===== Testing Endpoint: {endpoint} =====")
        
        for method in auth_methods:
            url = f"{BASE_URL}{endpoint}"
            print(f"\nTrying {method['name']}:")
            print(f"URL: {url}")
            print(f"Headers: {method['headers']}")
            
            try:
                response = requests.get(url, headers=method['headers'], timeout=10)
                
                print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    print("✅ SUCCESS!")
                    try:
                        data = response.json()
                        print(f"Response: {json.dumps(data, indent=2)[:200]}...")
                    except:
                        print(f"Response (text): {response.text[:200]}...")
                else:
                    print("❌ FAILED")
                    print(f"Response: {response.text}")
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_auth_methods()