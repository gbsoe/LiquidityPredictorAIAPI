#!/usr/bin/env python3
"""
Test script to validate Helius RPC endpoint and API key
"""
import json
import requests
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def test_helius_endpoint(api_key=None):
    """
    Test Helius RPC endpoint with various methods to determine what works
    
    Args:
        api_key: Helius API key to use
    """
    # Get API key from parameter or environment
    api_key = api_key or os.getenv("HELIUS_API_KEY", "1d54c390-7463-4f14-9995-f264140a5993")
    
    # Different possible endpoint formats to test
    endpoints = [
        f"https://mainnet.helius-rpc.com/?api-key={api_key}",
        f"https://rpc.helius.xyz/?api-key={api_key}",
        f"https://api.helius.xyz/v0/rpc?api-key={api_key}"
    ]
    
    # Test each endpoint format
    for endpoint_url in endpoints:
        logger.info(f"\n===== TESTING ENDPOINT: {endpoint_url[:30]}...{endpoint_url[-15:]} =====")
        test_endpoint_with_methods(endpoint_url)

def test_endpoint_with_methods(endpoint):
    
    logger.info(f"Testing Helius endpoint: {endpoint[:30]}...{endpoint[-15:]}")
    
    # List of methods to test
    test_methods = [
        {
            "name": "getVersion",
            "params": [],
            "description": "Basic RPC connection"
        },
        {
            "name": "getBalance",
            "params": ["So11111111111111111111111111111111111111112"],
            "description": "Get SOL token balance"
        },
        {
            "name": "getAccountInfo",
            "params": ["EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", {"encoding": "jsonParsed"}],
            "description": "Get USDC token info"
        },
        {
            "name": "getProgramAccounts",
            "params": ["675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8", {"encoding": "base64", "limit": 3}],
            "description": "Get Raydium program accounts (limited)"
        },
        {
            "name": "getClusterNodes",
            "params": [],
            "description": "Get Solana cluster nodes"
        }
    ]
    
    # Run each test
    results = {}
    
    for test in test_methods:
        method = test["name"]
        params = test["params"]
        description = test["description"]
        
        logger.info(f"Testing method: {method} - {description}")
        
        try:
            # Set up request payload
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params
            }
            
            # Make request
            response = requests.post(
                endpoint,
                json=payload,
                timeout=20,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            result = response.json()
            
            # Check for RPC errors
            if "error" in result:
                error = result["error"]
                results[method] = {
                    "success": False,
                    "error": f"RPC error: {error.get('message', str(error))}"
                }
                logger.error(f"RPC error: {error}")
            else:
                # Success - show a sample of the result
                result_str = str(result)
                if len(result_str) > 100:
                    result_str = result_str[:100] + "..."
                    
                results[method] = {
                    "success": True,
                    "result_sample": result_str
                }
                logger.info(f"Success: {result_str}")
                
        except Exception as e:
            results[method] = {
                "success": False,
                "error": f"Request error: {str(e)}"
            }
            logger.error(f"Error: {str(e)}")
    
    # Summarize results
    logger.info("\n===== TEST RESULTS =====")
    for method, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        details = result.get("result_sample", "") if result["success"] else result.get("error", "Unknown error")
        logger.info(f"{status}: {method} - {details}")
    
    successful_tests = sum(1 for r in results.values() if r["success"])
    logger.info(f"\nSUMMARY: {successful_tests}/{len(results)} tests passed")
    
    return successful_tests == len(results)

if __name__ == "__main__":
    test_helius_endpoint()