"""
Test the token data service
"""
import os
from token_data_service import TokenDataService

def test_token_data_service():
    # Get API key from environment variables
    api_key = os.getenv("DEFI_API_KEY")
    
    if not api_key:
        print("Error: No DEFI_API_KEY found in environment variables")
        return False
    
    # Initialize token data service
    token_service = TokenDataService(api_key=api_key)
    
    # Test getting a token by symbol
    token_symbols = ["SOL", "USDC", "ATLAS"]
    
    for symbol in token_symbols:
        print(f"\nTesting token data for {symbol}...")
        try:
            token_data = token_service.get_token_by_symbol(symbol)
            if token_data:
                print(f"Successfully retrieved data for {symbol}")
                print(f"Token data sample: {token_data}")
            else:
                print(f"No data found for {symbol}")
        except Exception as e:
            print(f"Error fetching token {symbol}: {str(e)}")
    
    # Test getting all tokens
    print("\nTesting getting all tokens...")
    try:
        all_tokens = token_service.get_all_tokens()
        print(f"Successfully retrieved {len(all_tokens)} tokens")
        if all_tokens:
            print(f"First token sample: {all_tokens[0]}")
    except Exception as e:
        print(f"Error fetching all tokens: {str(e)}")

if __name__ == "__main__":
    test_token_data_service()