"""
Simple test for the DeFi Aggregation API.
"""
import os
import requests

# API configuration
API_BASE_URL = "https://filotdefiapi.replit.app/api/v1"
API_KEY = os.getenv("DEFI_API_KEY")

print(f"API KEY found: {bool(API_KEY)}")
print(f"API KEY length: {len(API_KEY) if API_KEY else 0}")
print(f"API KEY prefix: {API_KEY[:5] if API_KEY and len(API_KEY) >= 5 else 'N/A'}")

# Let's try both header variations
headers_standard = {
    "X-API-KEY": API_KEY
}

headers_auth = {
    "Authorization": f"Bearer {API_KEY}"
}

# Test standard headers
print("\nTesting with X-API-KEY header...")
try:
    response = requests.get(f"{API_BASE_URL}/pools", headers=headers_standard, params={"limit": 1})
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text[:100]}")
except Exception as e:
    print(f"Error: {e}")

# Test Authorization header
print("\nTesting with Authorization header...")
try:
    response = requests.get(f"{API_BASE_URL}/pools", headers=headers_auth, params={"limit": 1})
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text[:100]}")
except Exception as e:
    print(f"Error: {e}")