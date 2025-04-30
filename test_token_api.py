import requests
import json
import sys

def test_token_api():
    print("Testing token API...")
    try:
        # Get tokens
        response = requests.get("https://filotdefiapi.replit.app/api/v1/tokens")
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return
        
        tokens = response.json()
        print(f"Retrieved {len(tokens)} tokens")
        
        # Print first few tokens
        for i, token in enumerate(tokens[:5]):
            print(f"\nToken #{i+1}:")
            print(json.dumps(token, indent=2))
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_token_api()