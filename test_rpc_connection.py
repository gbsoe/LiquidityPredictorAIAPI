"""
Test script to validate Solana RPC connection and retrieve pool data
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

# Load environment variables from .env file
load_dotenv()

def test_rpc_connection(rpc_endpoint=None):
    """
    Test if we can connect to the RPC endpoint and get a valid response
    
    Args:
        rpc_endpoint: Solana RPC endpoint (defaults to value from .env)
    
    Returns:
        True if connection was successful, False otherwise
    """
    # Get endpoint from parameter or environment
    endpoint = rpc_endpoint or os.getenv("SOLANA_RPC_ENDPOINT")
    
    if not endpoint:
        logger.error("No RPC endpoint provided. Please set SOLANA_RPC_ENDPOINT in .env file")
        return False
    
    # Use the endpoint directly as provided
    # No specific provider formatting needed
    
    logger.info(f"Testing connection to: {endpoint[:20]}...{endpoint[-20:] if len(endpoint) > 40 else ''}")
    
    try:
        # Create a simple session
        session = requests.Session()
        
        # Simple request to get the Solana version
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getVersion"
        }
        
        # Make request with timeout
        response = session.post(endpoint, json=payload, timeout=15)
        response.raise_for_status()  # Raise for HTTP errors
        
        # Check if we got a valid response
        result = response.json()
        if "result" in result:
            version = result["result"]["solana-core"]
            logger.info(f"Successfully connected! Solana version: {version}")
            return True
        else:
            logger.error(f"Invalid response from RPC endpoint: {result}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to RPC endpoint: {str(e)}")
        return False

def test_get_raydium_pools(rpc_endpoint=None, max_pools=5):
    """
    Test fetching Raydium pools from the blockchain
    
    Args:
        rpc_endpoint: Solana RPC endpoint (defaults to value from .env)
        max_pools: Maximum number of pools to fetch
    
    Returns:
        List of pools if successful, empty list otherwise
    """
    # Get endpoint from parameter or environment
    endpoint = rpc_endpoint or os.getenv("SOLANA_RPC_ENDPOINT")
    
    if not endpoint:
        logger.error("No RPC endpoint provided. Please set SOLANA_RPC_ENDPOINT in .env file")
        return []
    
    # Use the endpoint directly as provided
    # No specific provider formatting needed
    
    logger.info(f"Testing Raydium pools fetch from: {endpoint[:20]}...{endpoint[-20:] if len(endpoint) > 40 else ''}")
    
    try:
        # Create a simple session
        session = requests.Session()
        
        # Raydium v4 program ID
        raydium_program_id = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
        
        # Request to get program accounts
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getProgramAccounts",
            "params": [
                raydium_program_id,
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
        
        # Make request with timeout
        start_time = time.time()
        logger.info(f"Fetching Raydium pools (max: {max_pools})...")
        response = session.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()  # Raise for HTTP errors
        
        # Check if we got a valid response
        result = response.json()
        elapsed = time.time() - start_time
        
        if "result" in result:
            pools = result["result"]
            logger.info(f"Successfully fetched {len(pools)} Raydium pools in {elapsed:.2f}s")
            
            # Save raw data to file for inspection
            with open('raydium_pools_raw.json', 'w') as f:
                json.dump(pools, f, indent=2)
                
            logger.info(f"Saved raw data to raydium_pools_raw.json")
            return pools
        else:
            logger.error(f"Invalid response from RPC endpoint: {result}")
            return []
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Raydium pools: {str(e)}")
        return []

def run_tests():
    """Run all tests"""
    logger.info("Starting RPC connection tests")
    
    # Test basic connection
    if not test_rpc_connection():
        logger.error("Connection test failed. Please check your RPC endpoint")
        return
    
    # Test fetching Raydium pools
    pools = test_get_raydium_pools(max_pools=5)
    if pools:
        logger.info(f"Successfully fetched {len(pools)} pools")
        
        # Print some basic info about the first pool
        if len(pools) > 0:
            first_pool = pools[0]
            logger.info(f"First pool pubkey: {first_pool.get('pubkey', 'Unknown')}")
            logger.info(f"First pool data length: {len(first_pool.get('account', {}).get('data', ['']))}") 
    else:
        logger.error("Failed to fetch pools. Please check your RPC endpoint permissions")

if __name__ == "__main__":
    run_tests()