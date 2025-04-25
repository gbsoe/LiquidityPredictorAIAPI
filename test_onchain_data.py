import os
import json
import sys
from datetime import datetime

# Import the OnChainExtractor
from onchain_extractor import OnChainExtractor

def test_data_collection():
    """Test function to collect data from Solana blockchain and save to file"""
    print(f"[{datetime.now()}] Starting test data collection...")
    
    # Get the RPC endpoint from environment variables
    rpc_endpoint = os.getenv("SOLANA_RPC_ENDPOINT")
    if not rpc_endpoint:
        print(f"[{datetime.now()}] ERROR: SOLANA_RPC_ENDPOINT environment variable not set")
        return False
    
    print(f"[{datetime.now()}] Using RPC endpoint (masked): {rpc_endpoint[:10]}...{rpc_endpoint[-10:]}")
    
    try:
        # Initialize the extractor
        print(f"[{datetime.now()}] Initializing OnChainExtractor...")
        extractor = OnChainExtractor(rpc_endpoint=rpc_endpoint)
        
        # Extract pool data (limited number for testing)
        print(f"[{datetime.now()}] Extracting pool data from blockchain...")
        pools = extractor.extract_and_enrich_pools(max_per_dex=5)
        
        # Check if we got any data
        if not pools or len(pools) == 0:
            print(f"[{datetime.now()}] ERROR: No pool data retrieved")
            return False
        
        print(f"[{datetime.now()}] Successfully retrieved {len(pools)} pools")
        
        # Save to test file
        test_file = "test_extracted_pools.json"
        print(f"[{datetime.now()}] Saving data to {test_file}...")
        
        with open(test_file, "w") as f:
            json.dump(pools, f, indent=2)
        
        # Print sample of data for verification
        print(f"[{datetime.now()}] Data sample:")
        sample_pools = pools[:2]  # Just show first 2 pools
        print(json.dumps(sample_pools, indent=2))
        
        print(f"[{datetime.now()}] Test completed successfully")
        return True
    
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_collection()
    sys.exit(0 if success else 1)