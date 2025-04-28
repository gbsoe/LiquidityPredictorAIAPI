"""
Test DeFi API Endpoints (with hardcoded API key for testing)

This script tests all the API endpoints from the DeFi API documentation and 
outputs the results to a Markdown file. The API key is hardcoded for easy testing.
"""

import json
import requests
import time
from typing import Dict, Any, Optional

# ============================================================
# HARD-CODED API KEY FOR TESTING - REPLACE WITH YOUR ACTUAL KEY
# ============================================================
API_KEY = "your_api_key_here"  
# ============================================================

BASE_URL = "https://filotdefiapi.replit.app/api/v1"

# Output file
OUTPUT_MD = "defi_api_test_results.md"

def make_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make a request to the API with proper headers"""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    
    headers = {
        "X-API-Key": API_KEY,  # Using X-API-Key format per docs
        "Content-Type": "application/json"
    }
    
    print(f"Making request to: {url}")
    print(f"With headers: {headers}")
    if params:
        print(f"With params: {params}")
    
    response = requests.get(url, headers=headers, params=params)
    
    # Sleep to respect rate limits
    time.sleep(0.1)
    
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response text: {response.text}")
        return {"error": str(e), "status_code": response.status_code, "text": response.text}
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

def write_to_markdown(title: str, description: str, request_info: str, response: Dict[str, Any]) -> None:
    """Append test results to the markdown file"""
    with open(OUTPUT_MD, "a") as f:
        f.write(f"## {title}\n\n")
        f.write(f"{description}\n\n")
        f.write("### Request\n\n")
        f.write(f"```\n{request_info}\n```\n\n")
        f.write("### Response\n\n")
        
        # Format the response nicely
        if isinstance(response, dict) and response.get("error"):
            f.write(f"**Error:** {response.get('error')}\n\n")
            if "status_code" in response:
                f.write(f"**Status Code:** {response.get('status_code')}\n\n")
            if "text" in response:
                f.write(f"**Response Text:**\n\n```\n{response.get('text')}\n```\n\n")
        else:
            f.write("```json\n")
            f.write(json.dumps(response, indent=2))
            f.write("\n```\n\n")
        
        f.write("---\n\n")

def test_all_endpoints():
    """Test all endpoints from the documentation"""
    
    # Start with a fresh markdown file
    with open(OUTPUT_MD, "w") as f:
        f.write("# DeFi API Test Results\n\n")
        f.write(f"Tests run on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Base URL: `{BASE_URL}`\n\n")
        f.write(f"API Key Format Used: `X-API-Key`\n\n")
        f.write("---\n\n")
    
    # 1. Get All Pools
    endpoint = "pools"
    request_info = f"GET {BASE_URL}/{endpoint}"
    response = make_request(endpoint)
    write_to_markdown(
        "Get All Pools", 
        "Retrieve a list of all available liquidity pools across supported DEXes.",
        request_info, 
        response
    )
    
    # 2. Get Pool by ID
    pool_id = "8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj"  # Example from docs
    endpoint = f"pools/{pool_id}"
    request_info = f"GET {BASE_URL}/{endpoint}"
    response = make_request(endpoint)
    write_to_markdown(
        "Get Pool by ID", 
        f"Retrieve detailed information for pool ID: {pool_id}",
        request_info, 
        response
    )
    
    # 3. Get Pools by DEX
    dex_name = "Meteora"  # Example from docs
    endpoint = "pools"
    params = {"source": dex_name}
    request_info = f"GET {BASE_URL}/{endpoint}?source={dex_name}"
    response = make_request(endpoint, params)
    write_to_markdown(
        "Get Pools by DEX", 
        f"Retrieve pools from {dex_name} DEX.",
        request_info, 
        response
    )
    
    # 4. Get Pools by Token
    token_symbol = "SOL"  # Example from docs
    endpoint = "pools"
    params = {"token": token_symbol}
    request_info = f"GET {BASE_URL}/{endpoint}?token={token_symbol}"
    response = make_request(endpoint, params)
    write_to_markdown(
        "Get Pools by Token", 
        f"Retrieve pools that include {token_symbol} token.",
        request_info, 
        response
    )
    
    # 5. Get Top Pools by APR
    endpoint = "pools"
    params = {"sort": "apr24h", "order": "desc", "limit": 10}
    request_info = f"GET {BASE_URL}/{endpoint}?sort=apr24h&order=desc&limit=10"
    response = make_request(endpoint, params)
    write_to_markdown(
        "Get Top Pools by APR", 
        "Retrieve top-performing pools ordered by APR.",
        request_info, 
        response
    )
    
    # 6. Get Token Information
    token_symbol = "SOL"  # Example from docs
    endpoint = f"tokens/{token_symbol}"
    request_info = f"GET {BASE_URL}/{endpoint}"
    response = make_request(endpoint)
    write_to_markdown(
        "Get Token Information", 
        f"Retrieve information about {token_symbol} token.",
        request_info, 
        response
    )
    
    print(f"All tests completed. Results saved to {OUTPUT_MD}")

# This allows you to easily test different header formats by modifying the code directly
def make_request_bearer(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Alternative version using Bearer token format"""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",  # Alternative Bearer format
        "Content-Type": "application/json"
    }
    
    print(f"Making request to: {url}")
    print(f"With headers: {headers}")
    if params:
        print(f"With params: {params}")
    
    response = requests.get(url, headers=headers, params=params)
    
    # Sleep to respect rate limits
    time.sleep(0.1)
    
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response text: {response.text}")
        return {"error": str(e), "status_code": response.status_code, "text": response.text}
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # UNCOMMENT THE LINE BELOW TO TEST WITH BEARER TOKEN FORMAT INSTEAD OF X-API-KEY
    # make_request = make_request_bearer
    
    test_all_endpoints()