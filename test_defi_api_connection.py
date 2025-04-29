"""
Test the connection to the DeFi API and retrieve pool data
"""
import os
import requests
import json

def test_defi_api_connection():
    # Get API key from environment variables
    api_key = os.getenv("DEFI_API_KEY")
    
    if not api_key:
        print("Error: No DEFI_API_KEY found in environment variables")
        return False
    
    # Display API key information (securely)
    print(f"API key is set: {bool(api_key)}")
    print(f"API key length: {len(api_key)}")
    if api_key:
        masked_key = api_key[:4] + "..." + api_key[-4:]
        print(f"API key (masked): {masked_key}")
    
    # Set up API request
    base_url = "https://filotdefiapi.replit.app/api/v1"
    endpoint = "pools"
    
    # Try both header formats to see which one works
    headers_bearer = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    headers_x_api_key = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    # Test with Bearer token authentication
    print("\nTesting with Bearer token authentication...")
    try:
        response = requests.get(
            f"{base_url}/{endpoint}", 
            headers=headers_bearer,
            params={"limit": 5}  # Limit to 5 pools for testing
        )
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                pool_count = len(data) if isinstance(data, list) else 0
                print(f"Successfully retrieved {pool_count} pools")
                
                # Display sample data
                if pool_count > 0:
                    print("\nSample pool data:")
                    print(json.dumps(data[0], indent=2))
                
                return True
            except json.JSONDecodeError:
                print("Error: Response is not valid JSON")
                print(f"Response content: {response.text[:200]}...")
        else:
            print(f"Error response: {response.text[:200]}...")
    except Exception as e:
        print(f"Request error: {str(e)}")
    
    # Test with X-API-KEY header
    print("\nTesting with X-API-KEY header...")
    try:
        response = requests.get(
            f"{base_url}/{endpoint}", 
            headers=headers_x_api_key,
            params={"limit": 5}  # Limit to 5 pools for testing
        )
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                pool_count = len(data) if isinstance(data, list) else 0
                print(f"Successfully retrieved {pool_count} pools")
                
                # Display sample data
                if pool_count > 0:
                    print("\nSample pool data:")
                    print(json.dumps(data[0], indent=2))
                
                return True
            except json.JSONDecodeError:
                print("Error: Response is not valid JSON")
                print(f"Response content: {response.text[:200]}...")
        else:
            print(f"Error response: {response.text[:200]}...")
    except Exception as e:
        print(f"Request error: {str(e)}")
    
    return False

if __name__ == "__main__":
    test_defi_api_connection()