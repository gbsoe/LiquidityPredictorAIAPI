import os
import json
import sys
from datetime import datetime

# Import the OnChainExtractor
from onchain_extractor import OnChainExtractor

def test_data_collection():
    """Test function to collect data from Solana blockchain and save to file"""
    print(f"[{datetime.now()}] Starting test data collection with public RPC endpoint...")
    
    # Explicitly use the public Solana RPC endpoint
    rpc_endpoint = "https://api.mainnet-beta.solana.com"
    print(f"[{datetime.now()}] Using public RPC endpoint: {rpc_endpoint}")
    
    try:
        # Initialize the extractor with explicit endpoint
        print(f"[{datetime.now()}] Initializing OnChainExtractor...")
        extractor = OnChainExtractor(rpc_endpoint=rpc_endpoint)
        
        # Extract pool data (limited number for testing)
        print(f"[{datetime.now()}] Extracting pool data from blockchain...")
        pools = extractor.extract_and_enrich_pools(max_per_dex=3)  # Only get 3 pools per DEX for faster test
        
        # Check if we got any data
        if not pools or len(pools) == 0:
            print(f"[{datetime.now()}] ERROR: No pool data retrieved")
            return False
        
        print(f"[{datetime.now()}] Successfully retrieved {len(pools)} pools")
        
        # Save to test file
        test_file = "public_rpc_pools.json"
        print(f"[{datetime.now()}] Saving data to {test_file}...")
        
        with open(test_file, "w") as f:
            json.dump(pools, f, indent=2)
        
        # Print summary of data for verification
        print(f"[{datetime.now()}] Data summary:")
        dex_counts = {}
        for pool in pools:
            dex = pool.get('dex', 'unknown')
            dex_counts[dex] = dex_counts.get(dex, 0) + 1
        
        for dex, count in dex_counts.items():
            print(f"  - {dex}: {count} pools")
        
        # Print first pool as sample
        if pools:
            print(f"[{datetime.now()}] Sample pool:")
            sample_pool = pools[0]
            print(f"  - ID: {sample_pool.get('id', 'unknown')}")
            print(f"  - Name: {sample_pool.get('name', 'unknown')}")
            print(f"  - DEX: {sample_pool.get('dex', 'unknown')}")
            print(f"  - Liquidity: ${sample_pool.get('liquidity', 0):,.2f}")
            print(f"  - APR: {sample_pool.get('apr', 0):.2f}%")
        
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