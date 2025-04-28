"""
Test script for the DeFi Aggregation API to verify it works properly
before integrating it into the main application.

This script:
1. Tests API connectivity with provided authentication
2. Retrieves pools with proper rate limiting
3. Formats data according to our application's requirements
"""

import os
import time
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
API_BASE_URL = "https://filotdefiapi.replit.app/api/v1"
API_KEY = os.getenv("DEFI_API_KEY")

if not API_KEY:
    raise ValueError("DEFI_API_KEY not found in environment variables!")

print(f"API key: {API_KEY}")
# The API expects the key to start with "defi_"
if not API_KEY.startswith("defi_"):
    print("Warning: API key does not start with 'defi_' prefix which is required")

# Headers for API requests
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_api_connection():
    """Test basic connectivity to the API"""
    print("Testing API connection...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/pools", headers=headers, params={"limit": 1})
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response body: {response.text}")
        
        if response.status_code == 200:
            print("✅ API connection successful")
            data = response.json()
            # API may return data directly as a list or as an object with 'pools'
            if isinstance(data, list):
                pools = data
                print(f"Received {len(pools)} pools (list format)")
            else:
                pools = data.get('pools', [])
                print(f"Received {len(pools)} pools out of {data.get('total', 0)} total")
            return True
        elif response.status_code == 401:
            print("❌ Authentication failed. Check API key.")
            print(f"API Key being used: {API_KEY[:5]}...{API_KEY[-5:] if len(API_KEY) > 10 else ''}")
            return False
        elif response.status_code == 429:
            print("❌ Rate limit exceeded.")
            return False
        else:
            print(f"❌ API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False

def fetch_pools_with_rate_limiting(limit=50, page=1):
    """Fetch pools with rate limiting to respect the API's constraints"""
    print(f"Fetching pools (page {page}, limit {limit})...")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/pools", 
            headers=headers,
            params={"limit": limit, "page": page}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if the response is a list (direct pools) or an object with 'pools' key
            if isinstance(data, list):
                pools = data
                print(f"✅ Successfully fetched {len(pools)} pools (list format)")
            else:
                pools = data.get('pools', [])
                print(f"✅ Successfully fetched {len(pools)} pools from object")
                
            # Return a standardized format for consistency
            return {
                "pools": pools,
                "total": len(pools),
                "page": page
            }
        else:
            print(f"❌ Failed to fetch pools: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error fetching pools: {e}")
        return None

def get_all_pools(max_pools=500):
    """Get all pools with pagination and rate limiting"""
    print(f"Retrieving up to {max_pools} pools with rate limiting...")
    
    all_pools = []
    page = 1
    per_page = 10  # Small batch size to avoid rate limits
    
    while len(all_pools) < max_pools:
        data = fetch_pools_with_rate_limiting(limit=per_page, page=page)
        
        if not data or not data.get('pools'):
            break
            
        pools = data.get('pools', [])
        all_pools.extend(pools)
        
        print(f"Retrieved {len(pools)} pools (total so far: {len(all_pools)})")
        
        # If we've fetched all available pools, break
        if len(all_pools) >= data.get('total', 0):
            break
            
        # Rate limiting: 10 requests per second (with buffer)
        time.sleep(0.15)  # ~6 requests per second
        page += 1
    
    print(f"✅ Retrieved a total of {len(all_pools)} pools")
    return all_pools

def transform_pool_data(pool):
    """Transform API pool data to match our application's format"""
    try:
        # Extract token data
        tokens = pool.get('tokens', [])
        token1 = tokens[0] if len(tokens) > 0 else {}
        token2 = tokens[1] if len(tokens) > 1 else {}
        
        # Create transformed pool record
        transformed = {
            "id": pool.get('poolId', ''),
            "name": pool.get('name', ''),
            "dex": pool.get('source', 'Unknown'),
            "token1_symbol": token1.get('symbol', 'Unknown'),
            "token2_symbol": token2.get('symbol', 'Unknown'),
            "token1_address": token1.get('address', ''),
            "token2_address": token2.get('address', ''),
            "liquidity": pool.get('tvl', 0),
            "volume_24h": pool.get('volumeUsd', 0),
            "apr": pool.get('apr24h', 0),
            "apr_change_24h": 0,  # Would need historical data for change
            "apr_change_7d": pool.get('apr7d', 0) - pool.get('apr24h', 0) if pool.get('apr7d') and pool.get('apr24h') else 0,
            "prediction_score": 65,  # Default prediction score
            "category": "Stable" if "USD" in pool.get('name', '') or "stablecoin" in pool.get('name', '').lower() else "Standard",
            "token1_price": token1.get('price', 0),
            "token2_price": token2.get('price', 0),
            "fee_percentage": pool.get('fee', 0) * 100,  # Convert to percentage
        }
        
        return transformed
    except Exception as e:
        print(f"❌ Error transforming pool data: {e}")
        return None

def test_full_workflow():
    """Test the complete workflow from API to transformed data"""
    if not test_api_connection():
        print("API connectivity test failed. Exiting.")
        return False
    
    print("\nFetching and processing a small sample of pools...")
    pools = get_all_pools(max_pools=30)  # Limit to 30 for testing
    
    if not pools:
        print("Failed to retrieve pools. Exiting.")
        return False
    
    print("\nTransforming pool data to application format...")
    transformed_pools = []
    
    for pool in pools:
        transformed = transform_pool_data(pool)
        if transformed:
            transformed_pools.append(transformed)
    
    print(f"✅ Successfully transformed {len(transformed_pools)} pools")
    
    # Save a sample of the transformed data
    print("\nSaving sample transformed data...")
    try:
        with open('defi_api_test_sample.json', 'w') as f:
            json.dump(transformed_pools[:5], f, indent=2)
        print("✅ Sample data saved to defi_api_test_sample.json")
    except Exception as e:
        print(f"❌ Error saving sample data: {e}")
    
    return len(transformed_pools) > 0

if __name__ == "__main__":
    print("=== DeFi API Integration Test ===")
    success = test_full_workflow()
    
    if success:
        print("\n✅ All tests passed! The API is ready for integration.")
    else:
        print("\n❌ Tests failed. Please check the errors above.")