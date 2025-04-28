"""
Test Different Authentication Methods for the DeFi API

This script tests the API endpoints with both X-API-Key and Bearer token authentication
formats to determine which one works with the API.
"""

import json
import requests
import time
from typing import Dict, Any, Optional

# ============================================================
# API Key from environment variable
# ============================================================
import os
API_KEY = os.environ.get("DEFI_API_KEY", "your_api_key_here")  
# ============================================================

BASE_URL = "https://filotdefiapi.replit.app/api/v1"

# Output file
OUTPUT_MD = "auth_method_test_results.md"

def make_request_xapikey(endpoint: str) -> Dict[str, Any]:
    """Make a request with X-API-Key header format"""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    
    headers = {
        "X-API-Key": API_KEY,  # X-API-Key format
        "Content-Type": "application/json"
    }
    
    print(f"Making request to: {url} with X-API-Key")
    response = requests.get(url, headers=headers)
    
    try:
        return {
            "status_code": response.status_code,
            "headers_used": headers,
            "body": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {
            "status_code": response.status_code if hasattr(response, 'status_code') else None,
            "headers_used": headers,
            "error": str(e),
            "body": response.text if hasattr(response, 'text') else "Error retrieving response"
        }

def make_request_bearer(endpoint: str) -> Dict[str, Any]:
    """Make a request with Bearer token header format"""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",  # Bearer token format
        "Content-Type": "application/json"
    }
    
    print(f"Making request to: {url} with Bearer token")
    response = requests.get(url, headers=headers)
    
    try:
        return {
            "status_code": response.status_code,
            "headers_used": headers,
            "body": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {
            "status_code": response.status_code if hasattr(response, 'status_code') else None,
            "headers_used": headers,
            "error": str(e),
            "body": response.text if hasattr(response, 'text') else "Error retrieving response"
        }

def make_request_apikey_lowercase(endpoint: str) -> Dict[str, Any]:
    """Make a request with x-api-key (lowercase) header format"""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    
    headers = {
        "x-api-key": API_KEY,  # lowercase x-api-key format
        "Content-Type": "application/json"
    }
    
    print(f"Making request to: {url} with x-api-key (lowercase)")
    response = requests.get(url, headers=headers)
    
    try:
        return {
            "status_code": response.status_code,
            "headers_used": headers,
            "body": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {
            "status_code": response.status_code if hasattr(response, 'status_code') else None,
            "headers_used": headers,
            "error": str(e),
            "body": response.text if hasattr(response, 'text') else "Error retrieving response"
        }

def make_request_apikey_uppercase(endpoint: str) -> Dict[str, Any]:
    """Make a request with X-API-KEY (uppercase KEY) header format"""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    
    headers = {
        "X-API-KEY": API_KEY,  # uppercase KEY format
        "Content-Type": "application/json"
    }
    
    print(f"Making request to: {url} with X-API-KEY (uppercase KEY)")
    response = requests.get(url, headers=headers)
    
    try:
        return {
            "status_code": response.status_code,
            "headers_used": headers,
            "body": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {
            "status_code": response.status_code if hasattr(response, 'status_code') else None,
            "headers_used": headers,
            "error": str(e),
            "body": response.text if hasattr(response, 'text') else "Error retrieving response"
        }

def test_auth_methods():
    """Test different authentication header formats"""
    
    # Start with a fresh markdown file
    with open(OUTPUT_MD, "w") as f:
        f.write("# DeFi API Authentication Method Test Results\n\n")
        f.write(f"Tests run on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Base URL: `{BASE_URL}`\n\n")
        f.write("This document compares different authentication header formats to determine which one works with the API.\n\n")
        f.write("---\n\n")
    
    # Test simple endpoint (pools) with different auth methods
    endpoint = "pools?limit=1"  # Limit to 1 pool for quick test
    
    # Method 1: X-API-Key
    xapikey_response = make_request_xapikey(endpoint)
    
    # Method 2: Bearer token
    bearer_response = make_request_bearer(endpoint)
    
    # Method 3: lowercase x-api-key
    lowercase_response = make_request_apikey_lowercase(endpoint)
    
    # Method 4: uppercase X-API-KEY
    uppercase_response = make_request_apikey_uppercase(endpoint)
    
    # Sleep between tests to respect rate limits
    time.sleep(0.5)
    
    with open(OUTPUT_MD, "a") as f:
        f.write("## Method 1: X-API-Key\n\n")
        f.write("```\n")
        f.write(f"Headers: {xapikey_response['headers_used']}\n")
        f.write(f"Status Code: {xapikey_response['status_code']}\n")
        f.write("```\n\n")
        
        if xapikey_response['status_code'] == 200:
            f.write("✅ **SUCCESS**\n\n")
            f.write("Response sample:\n")
            f.write("```json\n")
            if isinstance(xapikey_response['body'], dict) or isinstance(xapikey_response['body'], list):
                f.write(json.dumps(xapikey_response['body'], indent=2)[:500])  # First 500 chars only
                f.write("\n...(truncated)...")
            else:
                f.write(str(xapikey_response['body'])[:500])
                f.write("\n...(truncated)...")
            f.write("\n```\n\n")
        else:
            f.write("❌ **FAILED**\n\n")
            f.write("Error response:\n")
            f.write("```\n")
            f.write(str(xapikey_response['body']))
            f.write("\n```\n\n")
        
        f.write("---\n\n")
        
        f.write("## Method 2: Bearer token\n\n")
        f.write("```\n")
        f.write(f"Headers: {bearer_response['headers_used']}\n")
        f.write(f"Status Code: {bearer_response['status_code']}\n")
        f.write("```\n\n")
        
        if bearer_response['status_code'] == 200:
            f.write("✅ **SUCCESS**\n\n")
            f.write("Response sample:\n")
            f.write("```json\n")
            if isinstance(bearer_response['body'], dict) or isinstance(bearer_response['body'], list):
                f.write(json.dumps(bearer_response['body'], indent=2)[:500])  # First 500 chars only
                f.write("\n...(truncated)...")
            else:
                f.write(str(bearer_response['body'])[:500])
                f.write("\n...(truncated)...")
            f.write("\n```\n\n")
        else:
            f.write("❌ **FAILED**\n\n")
            f.write("Error response:\n")
            f.write("```\n")
            f.write(str(bearer_response['body']))
            f.write("\n```\n\n")
        
        f.write("---\n\n")
        
        f.write("## Method 3: lowercase x-api-key\n\n")
        f.write("```\n")
        f.write(f"Headers: {lowercase_response['headers_used']}\n")
        f.write(f"Status Code: {lowercase_response['status_code']}\n")
        f.write("```\n\n")
        
        if lowercase_response['status_code'] == 200:
            f.write("✅ **SUCCESS**\n\n")
            f.write("Response sample:\n")
            f.write("```json\n")
            if isinstance(lowercase_response['body'], dict) or isinstance(lowercase_response['body'], list):
                f.write(json.dumps(lowercase_response['body'], indent=2)[:500])  # First 500 chars only
                f.write("\n...(truncated)...")
            else:
                f.write(str(lowercase_response['body'])[:500])
                f.write("\n...(truncated)...")
            f.write("\n```\n\n")
        else:
            f.write("❌ **FAILED**\n\n")
            f.write("Error response:\n")
            f.write("```\n")
            f.write(str(lowercase_response['body']))
            f.write("\n```\n\n")
        
        f.write("---\n\n")
        
        f.write("## Method 4: uppercase X-API-KEY\n\n")
        f.write("```\n")
        f.write(f"Headers: {uppercase_response['headers_used']}\n")
        f.write(f"Status Code: {uppercase_response['status_code']}\n")
        f.write("```\n\n")
        
        if uppercase_response['status_code'] == 200:
            f.write("✅ **SUCCESS**\n\n")
            f.write("Response sample:\n")
            f.write("```json\n")
            if isinstance(uppercase_response['body'], dict) or isinstance(uppercase_response['body'], list):
                f.write(json.dumps(uppercase_response['body'], indent=2)[:500])  # First 500 chars only
                f.write("\n...(truncated)...")
            else:
                f.write(str(uppercase_response['body'])[:500])
                f.write("\n...(truncated)...")
            f.write("\n```\n\n")
        else:
            f.write("❌ **FAILED**\n\n")
            f.write("Error response:\n")
            f.write("```\n")
            f.write(str(uppercase_response['body']))
            f.write("\n```\n\n")
        
        f.write("---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("| Method | Status Code | Result |\n")
        f.write("|--------|-------------|--------|\n")
        f.write(f"| X-API-Key | {xapikey_response['status_code']} | {'✅ Success' if xapikey_response['status_code'] == 200 else '❌ Failed'} |\n")
        f.write(f"| Bearer token | {bearer_response['status_code']} | {'✅ Success' if bearer_response['status_code'] == 200 else '❌ Failed'} |\n")
        f.write(f"| x-api-key (lowercase) | {lowercase_response['status_code']} | {'✅ Success' if lowercase_response['status_code'] == 200 else '❌ Failed'} |\n")
        f.write(f"| X-API-KEY (uppercase KEY) | {uppercase_response['status_code']} | {'✅ Success' if uppercase_response['status_code'] == 200 else '❌ Failed'} |\n")
        
    print(f"Authentication method tests completed. Results saved to {OUTPUT_MD}")

if __name__ == "__main__":
    test_auth_methods()