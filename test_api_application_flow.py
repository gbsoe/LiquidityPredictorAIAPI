"""
Test script that simulates the application's flow for retrieving pool data

This script makes an API request using the Bearer token authentication method,
captures the exact API response, and tests the transformation process that's 
causing errors in the application.
"""

import os
import json
import requests
import time
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Key from environment variable
API_KEY = os.environ.get("DEFI_API_KEY", "your_api_key_here")
BASE_URL = "https://filotdefiapi.replit.app/api/v1"

# Output files
RESPONSE_JSON_FILE = "api_response_data.json"
TRANSFORMED_JSON_FILE = "transformed_pool_data.json"
TEST_RESULTS_MD = "api_application_flow_test_results.md"

class LiquidityPool:
    """Simplified mock of the application's LiquidityPool class"""
    
    def __init__(self, id, name, dex, token1_symbol, token2_symbol, liquidity, 
                volume_24h, apr, category=None, **kwargs):
        self.id = id
        self.name = name
        self.dex = dex
        self.token1_symbol = token1_symbol
        self.token2_symbol = token2_symbol
        self.liquidity = liquidity
        self.volume_24h = volume_24h
        self.apr = apr
        self.category = category or "Other"
        
        # Store any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {attr: getattr(self, attr) for attr in dir(self) 
                if not attr.startswith('_') and not callable(getattr(self, attr))}


def get_pools_from_api() -> List[Dict[str, Any]]:
    """
    Get pools from the API using Bearer token authentication
    
    Returns:
        List of pool data dictionaries
    """
    url = f"{BASE_URL}/pools"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",  # Bearer token format
        "Content-Type": "application/json"
    }
    
    params = {"limit": 20}  # Get a reasonable number of pools for testing
    
    logger.info(f"Making request to: {url} with Bearer token")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        # Save raw API response for inspection
        pool_data = response.json()
        with open(RESPONSE_JSON_FILE, 'w') as f:
            json.dump(pool_data, f, indent=2)
            
        logger.info(f"Saved raw API response to {RESPONSE_JSON_FILE}")
        
        # Check what type of response we got (list or dict with 'pools' key)
        if isinstance(pool_data, dict) and 'pools' in pool_data:
            return pool_data['pools']
        elif isinstance(pool_data, list):
            return pool_data
        else:
            logger.error(f"Unexpected API response format: {type(pool_data)}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching pools: {str(e)}")
        return []


def transform_pool_data(pool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Transform API pool data to the format expected by the application
    
    Args:
        pool: Pool data from the API
        
    Returns:
        Transformed pool data dictionary or None if transformation fails
    """
    try:
        # Get the pool ID
        pool_id = pool.get("poolId", "")
        if not pool_id:
            logger.error(f"Pool missing ID: {pool}")
            return None
        
        # Get pool name
        pool_name = pool.get("name", "")
        if pool_name.endswith(" LP"):
            pool_name = pool_name[:-3]
        
        # Extract token data
        tokens = pool.get("tokens", [])
        token1 = tokens[0] if len(tokens) > 0 else {}
        token2 = tokens[1] if len(tokens) > 1 else {}
        
        token1_symbol = token1.get("symbol", "")
        token2_symbol = token2.get("symbol", "")
        
        # Skip if token symbols are missing
        if not token1_symbol or not token2_symbol:
            logger.warning(f"Pool with ID {pool_id} has unknown or missing token symbols: {token1_symbol}-{token2_symbol}")
            return None
        
        # If name is empty, use token symbols
        if not pool_name:
            pool_name = f"{token1_symbol}-{token2_symbol}"
        
        # Determine category (simplified)
        if "USDC" in token1_symbol or "USDC" in token2_symbol or "USDT" in token1_symbol or "USDT" in token2_symbol:
            category = "Stablecoin"
        elif "SOL" in token1_symbol or "SOL" in token2_symbol:
            category = "SOL-based"
        else:
            category = "Other"
        
        # Basic metrics
        tvl = pool.get("tvl", 0)
        apr_24h = pool.get("apr24h", 0)
        volume_24h = pool.get("volumeUsd", 0)
        
        # Return transformed data
        return {
            "id": pool_id,
            "name": pool_name,
            # Note: We're excluding display_name as it appears to be causing issues
            "dex": pool.get("source", "Unknown"),
            "token1_symbol": token1_symbol,
            "token2_symbol": token2_symbol,
            "liquidity": tvl,
            "volume_24h": volume_24h,
            "apr": apr_24h,
            "category": category,
            "last_updated": pool.get("updatedAt", ""),
        }
        
    except Exception as e:
        logger.error(f"Error transforming pool data: {str(e)}")
        return None


def create_pool_objects(transformed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Try to create pool objects from transformed data
    
    Args:
        transformed_data: List of transformed pool data dictionaries
        
    Returns:
        Dictionary with success/error counts and sample objects
    """
    successful_pools = []
    failed_pools = []
    
    for pool_data in transformed_data:
        try:
            # Try to create a pool object with the transformed data
            pool_obj = LiquidityPool(**pool_data)
            
            # If successful, add to the list
            successful_pools.append({
                "id": pool_data["id"],
                "name": pool_data["name"],
                "created": True,
                "object_dict": pool_obj.to_dict()
            })
            
        except Exception as e:
            # If creation fails, record the error
            failed_pools.append({
                "id": pool_data["id"],
                "name": pool_data["name"],
                "error": str(e),
                "data": pool_data
            })
    
    return {
        "successful_count": len(successful_pools),
        "failed_count": len(failed_pools),
        "successful_pools": successful_pools[:3],  # Show just the first 3
        "failed_pools": failed_pools[:3]  # Show just the first 3 failures
    }


def test_application_flow():
    """Run the full test of the application's data flow"""
    
    # Start with a fresh markdown file for results
    with open(TEST_RESULTS_MD, "w") as f:
        f.write("# DeFi API Application Flow Test Results\n\n")
        f.write(f"Tests run on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Base URL: `{BASE_URL}`\n\n")
        f.write("This document tests the full application flow for retrieving and processing pool data.\n\n")
        f.write("---\n\n")
    
    # Step 1: Fetch pools from API
    logger.info("Step 1: Fetching pools from API")
    pool_data = get_pools_from_api()
    
    with open(TEST_RESULTS_MD, "a") as f:
        f.write("## Step 1: API Data Retrieval\n\n")
        f.write(f"Retrieved {len(pool_data)} pools from the API.\n\n")
        
        if pool_data:
            f.write("Sample of raw API data for the first pool:\n\n")
            f.write("```json\n")
            if len(pool_data) > 0:
                f.write(json.dumps(pool_data[0], indent=2))
            f.write("\n```\n\n")
        else:
            f.write("❌ **FAILED to retrieve pool data**\n\n")
        
        f.write("---\n\n")
    
    if not pool_data:
        logger.error("No pool data retrieved. Stopping test.")
        return
    
    # Step 2: Transform pool data
    logger.info("Step 2: Transforming pool data")
    transformed_data = []
    
    for pool in pool_data:
        transformed = transform_pool_data(pool)
        if transformed:
            transformed_data.append(transformed)
    
    # Save transformed data for inspection
    with open(TRANSFORMED_JSON_FILE, 'w') as f:
        json.dump(transformed_data, f, indent=2)
    
    with open(TEST_RESULTS_MD, "a") as f:
        f.write("## Step 2: Data Transformation\n\n")
        f.write(f"Successfully transformed {len(transformed_data)} out of {len(pool_data)} pools.\n\n")
        
        if transformed_data:
            f.write("Sample of transformed data for the first pool:\n\n")
            f.write("```json\n")
            if len(transformed_data) > 0:
                f.write(json.dumps(transformed_data[0], indent=2))
            f.write("\n```\n\n")
        
        f.write("The full transformed dataset is saved to: `" + TRANSFORMED_JSON_FILE + "`\n\n")
        f.write("---\n\n")
    
    # Step 3: Try to create pool objects
    logger.info("Step 3: Creating pool objects")
    pool_creation_results = create_pool_objects(transformed_data)
    
    with open(TEST_RESULTS_MD, "a") as f:
        f.write("## Step 3: Object Creation\n\n")
        f.write(f"Successfully created {pool_creation_results['successful_count']} pool objects.\n")
        f.write(f"Failed to create {pool_creation_results['failed_count']} pool objects.\n\n")
        
        if pool_creation_results['successful_count'] > 0:
            f.write("### Successful Pool Objects (First 3)\n\n")
            for pool in pool_creation_results['successful_pools']:
                f.write(f"- Pool **{pool['name']}** (ID: `{pool['id']}`)\n")
            f.write("\n")
        
        if pool_creation_results['failed_count'] > 0:
            f.write("### Failed Pool Objects (First 3)\n\n")
            for pool in pool_creation_results['failed_pools']:
                f.write(f"- Pool **{pool['name']}** (ID: `{pool['id']}`)\n")
                f.write(f"  - Error: `{pool['error']}`\n")
            f.write("\n")
        
        f.write("---\n\n")
    
    # Step 4: Analysis of the API response structure
    logger.info("Step 4: Analyzing API response structure")
    
    with open(TEST_RESULTS_MD, "a") as f:
        f.write("## Step 4: API Response Structure Analysis\n\n")
        
        # Analyze the first pool's structure
        if pool_data:
            first_pool = pool_data[0]
            f.write("### API Pool Object Structure\n\n")
            f.write("Top-level fields in API response:\n\n")
            f.write("```\n")
            for key in first_pool.keys():
                f.write(f"- {key}: {type(first_pool[key]).__name__}\n")
            f.write("```\n\n")
            
            # Look at metrics if present
            if "metrics" in first_pool:
                f.write("Metrics fields:\n\n")
                f.write("```\n")
                for key in first_pool["metrics"].keys():
                    f.write(f"- {key}: {type(first_pool['metrics'][key]).__name__}\n")
                f.write("```\n\n")
            
            # Look at tokens if present
            if "tokens" in first_pool and len(first_pool["tokens"]) > 0:
                f.write("Token fields:\n\n")
                f.write("```\n")
                for key in first_pool["tokens"][0].keys():
                    f.write(f"- {key}: {type(first_pool['tokens'][0][key]).__name__}\n")
                f.write("```\n\n")
        
        f.write("---\n\n")
    
    # Step 5: Analysis of the error
    logger.info("Step 5: Analyzing error patterns")
    
    with open(TEST_RESULTS_MD, "a") as f:
        f.write("## Step 5: Error Analysis\n\n")
        
        if "display_name" in str(pool_creation_results['failed_pools']):
            f.write("### 'display_name' Error\n\n")
            f.write("The error `'display_name' is an invalid keyword argument for LiquidityPool` suggests that:\n\n")
            f.write("1. The transformation function is adding a `display_name` field to the pool data\n")
            f.write("2. The `LiquidityPool` class doesn't accept this parameter in its constructor\n\n")
            f.write("**Potential fixes:**\n\n")
            f.write("1. Remove `display_name` from the transformed data before creating pool objects\n")
            f.write("2. Update the `LiquidityPool` class to accept the `display_name` parameter\n")
            f.write("3. Rename `display_name` to an accepted parameter name\n\n")
        
        f.write("### Conclusion\n\n")
        if pool_creation_results['successful_count'] > 0:
            f.write("✅ **API data retrieval is working correctly with Bearer token authentication**\n\n")
        else:
            f.write("❌ **API data retrieval is still not working correctly**\n\n")
        
        if pool_creation_results['failed_count'] > 0:
            f.write("❌ **Object creation is failing due to data structure mismatch**\n\n")
        else:
            f.write("✅ **Object creation is working correctly**\n\n")
        
        f.write("See the full API response in `" + RESPONSE_JSON_FILE + "` and transformed data in `" + TRANSFORMED_JSON_FILE + "` for more details.\n\n")
    
    logger.info(f"Test completed. Results saved to {TEST_RESULTS_MD}")

if __name__ == "__main__":
    test_application_flow()