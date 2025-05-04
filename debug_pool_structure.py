import requests
import json
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('debug_pool_structure')

# API Configuration
API_KEY = "9feae0d0af47e4948e061f2d7820461e374e040c21cf65c087166d7ed18f5ed6"
BASE_URL = "https://raydium-trader-filot.replit.app"

def save_response_to_file(data, filename="api_response_debug.json"):
    """Save API response data to a JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Response saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save response: {str(e)}")

def get_pools_with_parameters(params=None):
    """Get pools with specific parameters and analyze the structure"""
    if params is None:
        params = {}
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    url = f"{BASE_URL}/api/pools"
    
    try:
        logger.info(f"Making request to {url} with params: {params}")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Request successful. Status code: {response.status_code}")
            
            # Analyze response structure
            if isinstance(data, dict):
                logger.info(f"Response top-level keys: {list(data.keys())}")
                
                if "pools" in data:
                    pools_data = data["pools"]
                    
                    if isinstance(pools_data, dict):
                        # If pools is a dictionary with categories
                        logger.info(f"Pools data is a dictionary with categories: {list(pools_data.keys())}")
                        
                        for category, pools in pools_data.items():
                            logger.info(f"Category '{category}' contains {len(pools)} pools")
                            
                            if pools:
                                # Analyze the first pool in this category
                                first_pool = pools[0]
                                logger.info(f"First pool in '{category}' has keys: {list(first_pool.keys())}")
                                
                                # Check for tokens
                                if "tokens" in first_pool:
                                    logger.info(f"Pool has {len(first_pool['tokens'])} tokens")
                                else:
                                    logger.info("Pool does not have 'tokens' field")
                                
                                # Check for nested objects
                                if "metrics" in first_pool:
                                    logger.info(f"Pool metrics has keys: {list(first_pool['metrics'].keys())}")
                    
                    elif isinstance(pools_data, list):
                        # If pools is a direct list of pools
                        logger.info(f"Pools data is a list with {len(pools_data)} items")
                        
                        if pools_data:
                            first_pool = pools_data[0]
                            logger.info(f"First pool has keys: {list(first_pool.keys())}")
                            
                            # Check for tokens
                            if "tokens" in first_pool:
                                logger.info(f"Pool has {len(first_pool['tokens'])} tokens")
                            else:
                                logger.info("Pool does not have 'tokens' field")
                            
                            # Check for nested objects
                            if "metrics" in first_pool:
                                logger.info(f"Pool metrics has keys: {list(first_pool['metrics'].keys())}")
            
            # Save full response for detailed analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"api_response_{timestamp}.json"
            save_response_to_file(data, filename)
            
            return data
        else:
            logger.error(f"Request failed. Status code: {response.status_code}")
            try:
                error_data = response.json()
                logger.error(f"Error response: {json.dumps(error_data, indent=2)}")
            except:
                logger.error(f"Raw response: {response.text}")
            
            return None
    
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        return None

def analyze_pool_categories():
    """Analyze different parameter combinations to find all pools"""
    
    # Test different parameter combinations
    param_sets = [
        # Basic query with no parameters
        {},
        
        # Limited results
        {"limit": 50},
        
        # Filtering by DEX/source
        {"source": "raydium"},
        {"source": "meteora"},
        {"source": "orca"},
        
        # Combined parameters
        {"limit": 50, "source": "raydium"},
        
        # Testing different field names from API docs
        {"dex": "raydium"},
        {"type": "amm"},
        
        # Testing pagination
        {"offset": 10, "limit": 10},
        
        # Testing sorting options
        {"sort": "tvl", "order": "desc"},
        {"sort": "apy", "order": "desc"},
    ]
    
    results = {}
    
    for params in param_sets:
        param_key = "&".join([f"{k}={v}" for k, v in params.items()]) if params else "default"
        logger.info(f"\n=== Testing with parameters: {param_key} ===")
        
        response = get_pools_with_parameters(params)
        
        if response:
            # Count pools by category if categorized
            pool_counts = {}
            
            if isinstance(response, dict) and "pools" in response:
                pools_data = response["pools"]
                
                if isinstance(pools_data, dict):
                    # Categorized pools
                    for category, pools in pools_data.items():
                        pool_counts[category] = len(pools)
                elif isinstance(pools_data, list):
                    # Direct list of pools
                    pool_counts["uncategorized"] = len(pools_data)
            
            results[param_key] = {
                "total_pool_count": sum(pool_counts.values()),
                "pool_counts_by_category": pool_counts
            }
    
    # Print summary
    logger.info("\n\n=== SUMMARY OF RESULTS ===")
    for param_key, result in results.items():
        logger.info(f"Parameters: {param_key}")
        logger.info(f"  Total pools: {result['total_pool_count']}")
        for category, count in result['pool_counts_by_category'].items():
            logger.info(f"    {category}: {count} pools")

if __name__ == "__main__":
    logger.info("===== API POOL STRUCTURE ANALYSIS =====")
    logger.info(f"Analyzing API at {BASE_URL}")
    
    analyze_pool_categories()
    
    logger.info("Analysis complete!")