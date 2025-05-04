"""
Simplified test script to find the correct authentication method
for the new API endpoint and save the result to .env file
"""
import requests
import os
import json
import time
import dotenv

# Constants
API_KEY = "9feae0d0af47e4948e061f2d7820461e374e040c21cf65c087166d7ed18f5ed6"
BASE_URL = "https://raydium-trader-filot.replit.app"

def test_authentication_format():
    """
    Test different header formats for authentication 
    and identify which one works reliably
    """
    # Define header variations to test
    header_variations = [
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
            "name": "X-API-KEY + Authorization Bearer",
            "headers": {
                "X-API-KEY": API_KEY,
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
        }
    ]
    
    # Test endpoints
    test_endpoints = [
        "/health",  # Basic health check
        "/api/pools"  # Main data endpoint
    ]
    
    # Track results
    results = {}
    best_header = None
    best_score = 0
    
    for endpoint in test_endpoints:
        print(f"\n=== Testing endpoint: {endpoint} ===")
        url = f"{BASE_URL}{endpoint}"
        
        for variation in header_variations:
            name = variation["name"]
            headers = variation["headers"]
            
            try:
                print(f"Trying {name}...")
                
                # Make the API request
                response = requests.get(url, headers=headers, timeout=10)
                
                # Check if successful
                if response.status_code == 200:
                    print(f"✅ SUCCESS with {name} (Status: {response.status_code})")
                    
                    # Store the result
                    if name not in results:
                        results[name] = {"success": 0, "total": 0}
                    results[name]["success"] += 1
                    results[name]["total"] += 1
                else:
                    print(f"❌ FAILED with {name} (Status: {response.status_code})")
                    if name not in results:
                        results[name] = {"success": 0, "total": 0}
                    results[name]["total"] += 1
                
            except Exception as e:
                print(f"❌ ERROR with {name}: {str(e)}")
                if name not in results:
                    results[name] = {"success": 0, "total": 0}
                results[name]["total"] += 1
            
            # Avoid rate limiting
            time.sleep(1)
    
    # Analyze results
    print("\n=== Results Summary ===")
    for name, stats in results.items():
        success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{name}: {stats['success']}/{stats['total']} successful ({success_rate*100:.1f}%)")
        
        # Track best performing header
        if success_rate > best_score:
            best_score = success_rate
            best_header = next(v for v in header_variations if v["name"] == name)
    
    if best_header:
        print(f"\nBest performing header format: {best_header['name']}")
        return best_header
    else:
        print("\nNo successful header format found")
        return None

def update_env_file(best_header):
    """Update .env file with the best authentication method"""
    if not best_header:
        print("No successful authentication method found, not updating .env file")
        return False
    
    # Load existing environment variables
    dotenv.load_dotenv()
    
    # Generate header format string
    if "Authorization" in best_header["headers"]:
        auth_format = "bearer"
    else:
        # Extract the exact case format from the header keys
        auth_format = next(k for k in best_header["headers"].keys() if k.lower() == "x-api-key")
    
    # Update or add the auth format environment variable
    env_updates = {
        "DEFI_API_URL": BASE_URL,
        "DEFI_API_KEY": API_KEY,
        "DEFI_API_AUTH_FORMAT": auth_format
    }
    
    # Update .env file
    with open(".env", "a+") as env_file:
        env_file.seek(0)
        current_content = env_file.read()
        
        for key, value in env_updates.items():
            # Check if the key already exists
            if f"{key}=" in current_content:
                # Update existing value
                lines = current_content.split("\n")
                updated_lines = []
                for line in lines:
                    if line.startswith(f"{key}="):
                        updated_lines.append(f"{key}={value}")
                    else:
                        updated_lines.append(line)
                
                # Write back the updated content
                env_file.seek(0)
                env_file.truncate()
                env_file.write("\n".join(updated_lines))
            else:
                # Add new value
                env_file.write(f"\n{key}={value}")
    
    print(f"Updated .env file with best authentication format: {auth_format}")
    return True

def create_auth_helper():
    """Create a helper module for consistent authentication"""
    code = """
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

def get_api_headers():
    \"\"\"
    Get the appropriate API headers based on the determined best format.
    This ensures consistent authentication across all API calls.
    \"\"\"
    api_key = os.getenv("DEFI_API_KEY")
    auth_format = os.getenv("DEFI_API_AUTH_FORMAT", "X-API-KEY")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Use the appropriate authentication format
    if auth_format.lower() == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        # Use the exact case format from the environment variable
        headers[auth_format] = api_key
    
    return headers
"""
    
    # Write the helper module
    with open("api_auth_helper.py", "w") as f:
        f.write(code.strip())
    
    print("Created api_auth_helper.py module for consistent authentication")

if __name__ == "__main__":
    print("=== Testing API Authentication Methods ===")
    print(f"API URL: {BASE_URL}")
    print(f"API Key: {API_KEY[:5]}...{API_KEY[-5:]}")
    
    best_header = test_authentication_format()
    
    if best_header:
        update_env_file(best_header)
        create_auth_helper()
        
        print("\n=== Success! ===")
        print("Now update defi_aggregation_api.py to use the api_auth_helper module")
    else:
        print("\n=== Failed to find working authentication method ===")
        print("Please verify the API URL and key manually")
    