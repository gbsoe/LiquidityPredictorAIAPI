#!/usr/bin/env python3
"""
Test script for Helius API connection and functionality.
This script verifies which methods work with our current API key.
"""

import os
import json
import time
import requests
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("helius_tester")

# Load environment variables
load_dotenv()

# Constants
DEFAULT_TIMEOUT = 10  # seconds
MAX_RETRIES = 2  # Keep it low for testing
RETRY_DELAY = 1.0  # seconds

# Get API endpoint from environment
ENDPOINT = os.getenv("SOLANA_RPC_ENDPOINT")
if not ENDPOINT:
    logger.error("SOLANA_RPC_ENDPOINT not found in environment")
    exit(1)

# Override for testing - use the specific value from .env file
ENDPOINT = "https://mainnet.helius-rpc.com/?api-key=1d54c390-7463-4f14-9995-f264140a5993"
logger.info(f"Using Helius endpoint: {ENDPOINT[:30]}...{ENDPOINT[-15:]}")

# Methods to test
METHODS_TO_TEST = [
    # Core account info methods
    "getAccountInfo",
    "getProgramAccounts",
    "getBalance",
    
    # Token methods
    "getTokenAccountBalance",
    "getTokenSupply",
    
    # Block and transaction methods
    "getBlock",
    "getTransaction",
    "getSignaturesForAddress",
    
    # General network methods
    "getVersion",
    "getHealth",
    "getSlot",
    
    # Additional methods
    "getEpochInfo",
    "getInflationRate",
    "getLatestBlockhash",
    "getMinimumBalanceForRentExemption",
    "getClusterNodes"
]

# Test addresses and parameters
TEST_ADDRESSES = {
    "solana": "So11111111111111111111111111111111111111112",  # Wrapped SOL
    "usdc": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC token mint
    "raydium_program": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium v4
    "raydium_pool": "CS1qzNMiAUNRLJys7exabzPhZMwzMfwZUmzNEDmYcRY3",  # SOL-USDC pool
    "wallet": "3PwAUGXfGwDy9PdGhwXJDhLZXS9PVNR24GhkD9rY9xdq"  # Example wallet with transactions
}

# Create a session
session = requests.Session()
session.headers.update({
    "Content-Type": "application/json",
    "User-Agent": "SolanaPoolAnalysis/1.0"
})

def make_rpc_request(method, params):
    """Make an RPC request to the Solana API endpoint"""
    data = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Testing method: {method}")
            
            response = session.post(
                ENDPOINT,
                json=data,
                timeout=DEFAULT_TIMEOUT
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                error = result["error"]
                logger.error(f"RPC error for {method}: {error}")
                return False, result
            
            return True, result
            
        except requests.RequestException as e:
            logger.warning(f"Request failed for {method} (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    logger.error(f"All retries failed for method {method}")
    return False, {"error": "All retries failed"}

def test_methods():
    """Test all methods and return results"""
    results = {}
    
    # Test getAccountInfo
    success, result = make_rpc_request(
        "getAccountInfo",
        [TEST_ADDRESSES["solana"], {"encoding": "jsonParsed"}]
    )
    results["getAccountInfo"] = {
        "success": success,
        "sample_result": result.get("result", {}) if success else "Failed"
    }
    
    # Test getBalance
    success, result = make_rpc_request(
        "getBalance",
        [TEST_ADDRESSES["wallet"]]
    )
    results["getBalance"] = {
        "success": success,
        "sample_result": result.get("result", {}) if success else "Failed"
    }
    
    # Test getProgramAccounts (with limit to avoid timeout)
    success, result = make_rpc_request(
        "getProgramAccounts",
        [TEST_ADDRESSES["raydium_program"], {"limit": 2, "encoding": "base64"}]
    )
    results["getProgramAccounts"] = {
        "success": success,
        "sample_result": result.get("result", {}) if success else "Failed"
    }
    
    # Test getTokenAccountBalance
    success, result = make_rpc_request(
        "getTokenAccountBalance",
        ["AjAXQhXxqGe7erWDCQVrVWwEFJqTjT2H8e9yRvavSdfg"]  # Example USDC token account
    )
    results["getTokenAccountBalance"] = {
        "success": success,
        "sample_result": result.get("result", {}) if success else "Failed"
    }
    
    # Test getTokenSupply
    success, result = make_rpc_request(
        "getTokenSupply",
        [TEST_ADDRESSES["usdc"]]
    )
    results["getTokenSupply"] = {
        "success": success,
        "sample_result": result.get("result", {}) if success else "Failed"
    }
    
    # Test getBlock (recent block)
    success, result = make_rpc_request(
        "getBlock",
        [200000000, {"encoding": "json", "maxSupportedTransactionVersion": 0}]
    )
    results["getBlock"] = {
        "success": success, 
        "sample_result": "Success" if success else "Failed"  # Block result too large to display
    }
    
    # Test getSignaturesForAddress
    success, result = make_rpc_request(
        "getSignaturesForAddress",
        [TEST_ADDRESSES["raydium_pool"], {"limit": 2}]
    )
    results["getSignaturesForAddress"] = {
        "success": success,
        "sample_result": result.get("result", {}) if success else "Failed"
    }
    
    # Test getTransaction (requires a valid signature)
    # Skipping for now as we'd need a recent signature
    
    # Test getVersion
    success, result = make_rpc_request(
        "getVersion",
        []
    )
    results["getVersion"] = {
        "success": success,
        "sample_result": result.get("result", {}) if success else "Failed"
    }
    
    # Test getHealth
    success, result = make_rpc_request(
        "getHealth",
        []
    )
    results["getHealth"] = {
        "success": success,
        "sample_result": result.get("result", {}) if success else "Failed"
    }
    
    # Test getSlot
    success, result = make_rpc_request(
        "getSlot",
        []
    )
    results["getSlot"] = {
        "success": success,
        "sample_result": result.get("result", {}) if success else "Failed"
    }
    
    # Test getLatestBlockhash
    success, result = make_rpc_request(
        "getLatestBlockhash",
        []
    )
    results["getLatestBlockhash"] = {
        "success": success,
        "sample_result": result.get("result", {}) if success else "Failed"
    }
    
    # Test getEpochInfo
    success, result = make_rpc_request(
        "getEpochInfo",
        []
    )
    results["getEpochInfo"] = {
        "success": success,
        "sample_result": result.get("result", {}) if success else "Failed"
    }
    
    return results

def main():
    logger.info(f"Testing Helius API endpoint: {ENDPOINT[:20]}...{ENDPOINT[-20:] if len(ENDPOINT) > 40 else ENDPOINT}")
    
    # Test the connection
    success, result = make_rpc_request("getHealth", [])
    if not success:
        logger.error("Failed to connect to Helius API")
        return
    
    logger.info("Connection successful, testing methods...")
    
    # Test all methods
    results = test_methods()
    
    # Print summary
    logger.info("\n--- Method Test Results ---")
    for method, data in results.items():
        status = "✅ Success" if data["success"] else "❌ Failed"
        logger.info(f"{method}: {status}")
    
    # Save detailed results to file
    with open("helius_api_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Detailed results saved to helius_api_test_results.json")

if __name__ == "__main__":
    main()