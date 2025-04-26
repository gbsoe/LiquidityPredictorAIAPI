"""
This script tests various Solana RPC endpoints to find ones that allow access to liquidity pool data
"""
import os
import json
import logging
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load environment variables
load_dotenv()

# List of public Solana RPC endpoints to try
PUBLIC_ENDPOINTS = [
    "https://api.mainnet-beta.solana.com",           # Solana public RPC
    "https://solana-api.projectserum.com",           # Project Serum
    "https://rpc.ankr.com/solana",                   # Ankr
    "https://mainnet.rpcpool.com",                   # RPCPool
    "https://sol.getblock.io/mainnet",               # GetBlock
]

# Raydium v4 program ID - a major Solana DEX
RAYDIUM_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

def test_endpoint(endpoint, max_pools=5):
    """
    Test if an endpoint can retrieve pool data
    
    Args:
        endpoint: RPC endpoint URL
        max_pools: Maximum number of pools to request
        
    Returns:
        Dict with success status, message, and pool count
    """
    logger.info(f"Testing endpoint: {endpoint}")
    start_time = time.time()
    
    try:
        # Create session
        session = requests.Session()
        
        # First test basic connection with getVersion
        version_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getVersion"
        }
        
        version_response = session.post(endpoint, json=version_payload, timeout=10)
        version_response.raise_for_status()
        version_result = version_response.json()
        
        if "result" not in version_result:
            return {
                "success": False,
                "message": f"Invalid response from endpoint: {version_result}",
                "endpoint": endpoint,
                "pools": 0,
                "time": time.time() - start_time
            }
            
        # Now try to get program accounts
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getProgramAccounts",
            "params": [
                RAYDIUM_PROGRAM_ID,
                {
                    "encoding": "base64",
                    "commitment": "confirmed",
                    "filters": [
                        {
                            "dataSize": 1400  # Approximate size for Raydium pool accounts
                        }
                    ],
                    "limit": max_pools
                }
            ]
        }
        
        logger.info(f"Requesting program accounts from {endpoint}")
        response = session.post(endpoint, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        
        if "result" not in result:
            return {
                "success": False,
                "message": f"Invalid program accounts response: {result}",
                "endpoint": endpoint,
                "pools": 0,
                "time": time.time() - start_time
            }
            
        pools = result["result"]
        
        return {
            "success": True,
            "message": f"Successfully fetched {len(pools)} pools in {time.time() - start_time:.2f}s",
            "endpoint": endpoint,
            "pools": len(pools),
            "time": time.time() - start_time
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "endpoint": endpoint,
            "pools": 0,
            "time": time.time() - start_time
        }

def check_all_endpoints():
    """Test all available endpoints"""
    results = []
    
    # Test current endpoint from environment first
    current_endpoint = os.getenv("SOLANA_RPC_ENDPOINT")
    if current_endpoint:
        # If it's just a UUID (Helius API key)
        if len(current_endpoint) == 36 and current_endpoint.count('-') == 4:
            # Convert to Helius URL
            current_endpoint = f"https://mainnet.helius-rpc.com/?api-key={current_endpoint}"
        
        result = test_endpoint(current_endpoint)
        results.append(result)
    
    # Test all public endpoints
    for endpoint in PUBLIC_ENDPOINTS:
        result = test_endpoint(endpoint)
        results.append(result)
    
    # Print results
    logger.info("\n===== RESULTS =====")
    for result in results:
        status = "✅" if result["success"] and result["pools"] > 0 else "❌"
        logger.info(f"{status} {result['endpoint']}: {result['message']}")
    
    # Find best endpoint
    successful = [r for r in results if r["success"] and r["pools"] > 0]
    if successful:
        best = max(successful, key=lambda x: x["pools"])
        logger.info(f"\n🏆 Best endpoint: {best['endpoint']} with {best['pools']} pools in {best['time']:.2f}s")
        
        # Save to .env if it's different from current
        if best["endpoint"] != current_endpoint:
            logger.info(f"Updating .env with new best endpoint")
            env_path = ".env"
            
            # Read current .env
            with open(env_path, "r") as f:
                env_content = f.read()
            
            # Update or add SOLANA_RPC_ENDPOINT
            if "SOLANA_RPC_ENDPOINT=" in env_content:
                # Replace existing value
                lines = env_content.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("SOLANA_RPC_ENDPOINT="):
                        lines[i] = f"SOLANA_RPC_ENDPOINT={best['endpoint']}"
                        break
                
                # Write back
                with open(env_path, "w") as f:
                    f.write("\n".join(lines))
            else:
                # Append to file
                with open(env_path, "a") as f:
                    f.write(f"\nSOLANA_RPC_ENDPOINT={best['endpoint']}\n")
                    
            logger.info(f"Updated .env file with new endpoint")
    else:
        logger.warning("⚠️ No successful endpoints found that can retrieve pool data")
        logger.info("You might need to use a paid RPC service with higher limits for production use")

if __name__ == "__main__":
    check_all_endpoints()