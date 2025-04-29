"""
DeFi API Data Test Script

This script demonstrates fetching liquidity pool data from the DeFi API
using Bearer token authentication and processes it to extract the necessary
data fields required for prediction models.
"""

import requests
import json
import os
import pandas as pd
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('defi_api_test')

# DeFi API configuration
API_KEY = "defi_WyJ71mVrIDzEkzwauPu_FpnRh__W83_l"
BASE_URL = "https://filotdefiapi.replit.app/api/v1"
# Try both header formats - different files in the codebase use different auth methods
HEADERS_BEARER = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

HEADERS_X_API_KEY = {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json'
}

# Output file for results
RESULTS_FILE = "defi_api_test_results.md"

def fetch_pools(limit=10):
    """
    Fetch liquidity pool data from the DeFi API
    
    Args:
        limit: Maximum number of pools to fetch
        
    Returns:
        List of pool data dictionaries
    """
    url = f"{BASE_URL}/pools"
    params = {
        "limit": limit
    }
    
    # Try with Bearer token authentication first
    try:
        logger.info(f"Fetching pool data from {url} using Bearer token")
        response = requests.get(url, headers=HEADERS_BEARER, params=params)
        
        if response.status_code == 200:
            data = response.json()
            # Check if the response is a list directly
            if isinstance(data, list):
                logger.info(f"Successfully fetched {len(data)} pools (directly as list)")
                return data
            # Check if the response is a dict with a 'pools' key
            elif isinstance(data, dict) and 'pools' in data:
                logger.info(f"Successfully fetched {len(data['pools'])} pools")
                return data['pools']
            else:
                logger.info(f"Response structure: {type(data)}")
                logger.info(f"Sample response: {str(data)[:200]}")
                return None
        else:
            logger.warning(f"Error fetching pools with Bearer token: {response.status_code} - {response.text}")
            # Fall through to try with X-API-Key
    except Exception as e:
        logger.warning(f"Exception fetching pools with Bearer token: {str(e)}")
        # Fall through to try with X-API-Key
    
    # Try with X-API-Key authentication
    try:
        logger.info(f"Fetching pool data from {url} using X-API-Key")
        response = requests.get(url, headers=HEADERS_X_API_KEY, params=params)
        
        if response.status_code == 200:
            data = response.json()
            # Check if the response is a list directly
            if isinstance(data, list):
                logger.info(f"Successfully fetched {len(data)} pools (directly as list)")
                return data
            # Check if the response is a dict with a 'pools' key
            elif isinstance(data, dict) and 'pools' in data:
                logger.info(f"Successfully fetched {len(data['pools'])} pools")
                return data['pools']
            else:
                logger.info(f"Response structure: {type(data)}")
                logger.info(f"Sample response: {str(data)[:200]}")
                return None
        else:
            logger.error(f"Error fetching pools with X-API-Key: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Exception fetching pools with X-API-Key: {str(e)}")
        return None

def fetch_pool_history(pool_id):
    """
    Fetch historical data for a specific pool
    
    Args:
        pool_id: Pool identifier
        
    Returns:
        Dictionary with historical data
    """
    url = f"{BASE_URL}/pools/{pool_id}/history"
    params = {
        "days": 30,
        "interval": "day"
    }
    
    # Try with Bearer token first
    try:
        logger.info(f"Fetching historical data for pool {pool_id} with Bearer token")
        response = requests.get(url, headers=HEADERS_BEARER, params=params)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Successfully fetched historical data for pool {pool_id}")
            return data
        else:
            logger.warning(f"Error fetching pool history with Bearer token: {response.status_code}")
            # Fall through to try X-API-Key
    except Exception as e:
        logger.warning(f"Exception fetching pool history with Bearer token: {str(e)}")
        # Fall through to try X-API-Key
    
    # Try with X-API-Key
    try:
        logger.info(f"Fetching historical data for pool {pool_id} with X-API-Key")
        response = requests.get(url, headers=HEADERS_X_API_KEY, params=params)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Successfully fetched historical data for pool {pool_id}")
            return data
        else:
            logger.error(f"Error fetching pool history with X-API-Key: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Exception fetching pool history with X-API-Key: {str(e)}")
        return None

def fetch_token_prices(token_addresses):
    """
    Fetch current token prices
    
    Args:
        token_addresses: List of token addresses
        
    Returns:
        Dictionary mapping token addresses to prices
    """
    url = f"{BASE_URL}/tokens/prices"
    params = {
        "addresses": ",".join(token_addresses),
        "network": "solana"
    }
    
    # Try with Bearer token first
    try:
        logger.info(f"Fetching token prices for {len(token_addresses)} tokens with Bearer token")
        response = requests.get(url, headers=HEADERS_BEARER, params=params)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Successfully fetched token prices")
            return data.get('prices', {})
        else:
            logger.warning(f"Error fetching token prices with Bearer token: {response.status_code}")
            # Fall through to try X-API-Key
    except Exception as e:
        logger.warning(f"Exception fetching token prices with Bearer token: {str(e)}")
        # Fall through to try X-API-Key
        
    # Try with X-API-Key
    try:
        logger.info(f"Fetching token prices for {len(token_addresses)} tokens with X-API-Key")
        response = requests.get(url, headers=HEADERS_X_API_KEY, params=params)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Successfully fetched token prices")
            return data.get('prices', {})
        else:
            logger.error(f"Error fetching token prices with X-API-Key: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        logger.error(f"Exception fetching token prices with X-API-Key: {str(e)}")
        return {}

def calculate_pool_metrics(pool, historical_data):
    """
    Calculate additional metrics for a pool based on historical data
    
    Args:
        pool: Basic pool data
        historical_data: Historical metrics for the pool
        
    Returns:
        Pool data with additional calculated metrics
    """
    enhanced_pool = pool.copy()
    
    # Default values in case we can't calculate
    enhanced_pool['apr_change_24h'] = 0.0
    enhanced_pool['apr_change_7d'] = 0.0
    enhanced_pool['apr_change_30d'] = 0.0
    enhanced_pool['tvl_change_24h'] = 0.0
    enhanced_pool['tvl_change_7d'] = 0.0
    enhanced_pool['tvl_change_30d'] = 0.0
    enhanced_pool['volume_change_24h'] = 0.0
    enhanced_pool['liquidity_change_24h'] = 0.0
    
    # Assign category based on tokens
    token1 = enhanced_pool.get('token1', {}).get('symbol', '').upper()
    token2 = enhanced_pool.get('token2', {}).get('symbol', '').upper()
    
    if token1 in ['USDC', 'USDT', 'DAI'] and token2 in ['USDC', 'USDT', 'DAI']:
        enhanced_pool['category'] = 'Stablecoin'
    elif token1 in ['SOL', 'ETH', 'BTC'] or token2 in ['SOL', 'ETH', 'BTC']:
        enhanced_pool['category'] = 'Major'
    elif token1 in ['BONK', 'SAMO', 'PEPE'] or token2 in ['BONK', 'SAMO', 'PEPE']:
        enhanced_pool['category'] = 'Meme'
    elif token1 in ['GMT', 'ATLAS', 'POLIS'] or token2 in ['GMT', 'ATLAS', 'POLIS']:
        enhanced_pool['category'] = 'Gaming'
    elif token1 in ['JUP', 'RAY', 'SRM', 'ORCA'] or token2 in ['JUP', 'RAY', 'SRM', 'ORCA']:
        enhanced_pool['category'] = 'DeFi'
    else:
        enhanced_pool['category'] = 'Other'
    
    # If we have historical data, calculate changes
    if historical_data and 'history' in historical_data and len(historical_data['history']) > 0:
        history = historical_data['history']
        
        # Sort by timestamp to ensure order
        history.sort(key=lambda x: x.get('timestamp', 0))
        
        # Get latest and reference points
        latest = history[-1]
        
        # 24h change
        if len(history) > 1:
            prev_24h = history[-2]
            if 'apr' in latest and 'apr' in prev_24h and prev_24h['apr'] != 0:
                enhanced_pool['apr_change_24h'] = ((latest['apr'] - prev_24h['apr']) / prev_24h['apr']) * 100
            
            if 'tvl' in latest and 'tvl' in prev_24h and prev_24h['tvl'] != 0:
                enhanced_pool['tvl_change_24h'] = ((latest['tvl'] - prev_24h['tvl']) / prev_24h['tvl']) * 100
                
            if 'volume' in latest and 'volume' in prev_24h and prev_24h['volume'] != 0:
                enhanced_pool['volume_change_24h'] = ((latest['volume'] - prev_24h['volume']) / prev_24h['volume']) * 100
        
        # 7d change
        if len(history) > 7:
            prev_7d = history[-8]
            if 'apr' in latest and 'apr' in prev_7d and prev_7d['apr'] != 0:
                enhanced_pool['apr_change_7d'] = ((latest['apr'] - prev_7d['apr']) / prev_7d['apr']) * 100
            
            if 'tvl' in latest and 'tvl' in prev_7d and prev_7d['tvl'] != 0:
                enhanced_pool['tvl_change_7d'] = ((latest['tvl'] - prev_7d['tvl']) / prev_7d['tvl']) * 100
        
        # 30d change
        if len(history) > 29:
            prev_30d = history[0]
            if 'apr' in latest and 'apr' in prev_30d and prev_30d['apr'] != 0:
                enhanced_pool['apr_change_30d'] = ((latest['apr'] - prev_30d['apr']) / prev_30d['apr']) * 100
            
            if 'tvl' in latest and 'tvl' in prev_30d and prev_30d['tvl'] != 0:
                enhanced_pool['tvl_change_30d'] = ((latest['tvl'] - prev_30d['tvl']) / prev_30d['tvl']) * 100
    
    # Calculate a simple prediction score based on available metrics
    prediction_score = 50  # Base score
    
    # APR contributes up to 20 points
    apr = enhanced_pool.get('apr', 0)
    prediction_score += min(20, apr / 2)
    
    # Positive trends add points
    if enhanced_pool['apr_change_7d'] > 0:
        prediction_score += 5
    if enhanced_pool['tvl_change_7d'] > 0:
        prediction_score += 5
    if enhanced_pool['apr_change_24h'] > 0:
        prediction_score += 3
    if enhanced_pool['tvl_change_24h'] > 0:
        prediction_score += 3
    
    # Category adjustments
    if enhanced_pool['category'] == 'Meme':
        prediction_score += 7  # Higher volatility, higher potential
    elif enhanced_pool['category'] == 'DeFi':
        prediction_score += 5  # Good potential
    elif enhanced_pool['category'] == 'Major':
        prediction_score += 3  # Stable but lower ceiling
    
    # Cap at 100
    enhanced_pool['prediction_score'] = min(100, prediction_score)
    
    return enhanced_pool

def standardize_pool_data(pool_data):
    """
    Standardize the pool data structure to match our prediction requirements
    
    Args:
        pool_data: Raw pool data from API
        
    Returns:
        Standardized pool data dictionary
    """
    try:
        standardized = {}
        
        # Basic pool information
        standardized['id'] = pool_data.get('poolId', '')
        
        # Handle tokens data
        tokens = pool_data.get('tokens', [])
        token1 = tokens[0] if len(tokens) > 0 else {}
        token2 = tokens[1] if len(tokens) > 1 else {}
        
        # Extract token symbols for name
        token1_symbol = token1.get('symbol', '')
        token2_symbol = token2.get('symbol', '')
        standardized['name'] = f"{token1_symbol}/{token2_symbol}"
        
        # Extract token details
        standardized['dex'] = pool_data.get('source', 'Unknown')
        standardized['token1_symbol'] = token1_symbol
        standardized['token2_symbol'] = token2_symbol
        standardized['token1_address'] = token1.get('address', '')
        standardized['token2_address'] = token2.get('address', '')
        
        # Financial metrics from the metrics field
        metrics = pool_data.get('metrics', {})
        standardized['liquidity'] = metrics.get('tvl', 0)
        standardized['volume_24h'] = metrics.get('volumeUsd', 0)
        standardized['apr'] = metrics.get('apy24h', 0)  # Use 24h APY as the current APR
        standardized['fee'] = metrics.get('fee', 0) * 100  # Convert to percentage
        
        # Historical trend data - calculated as percent changes between periods
        apy24h = metrics.get('apy24h', 0)
        apy7d = metrics.get('apy7d', 0)
        apy30d = metrics.get('apy30d', 0)
        
        # Calculate APR changes if we have the values
        if apy7d != 0 and apy24h != 0:
            standardized['apr_change_24h'] = ((apy24h - apy7d) / apy7d) * 100
        else:
            standardized['apr_change_24h'] = 0
            
        if apy30d != 0 and apy7d != 0:
            standardized['apr_change_7d'] = ((apy7d - apy30d) / apy30d) * 100
        else:
            standardized['apr_change_7d'] = 0
            
        standardized['apr_change_30d'] = 0  # No data for previous 30d period
        
        # No TVL change data in API, will be calculated from historical records
        standardized['tvl_change_24h'] = 0
        standardized['tvl_change_7d'] = 0
        standardized['tvl_change_30d'] = 0
        
        # Categorization based on token names
        token_symbols = [token1_symbol, token2_symbol]
        if any(symbol in ['USDC', 'USDT', 'DAI'] for symbol in token_symbols):
            if all(symbol in ['USDC', 'USDT', 'DAI'] for symbol in token_symbols):
                standardized['category'] = 'Stablecoin'
            elif any(symbol in ['SOL', 'ETH', 'BTC'] for symbol in token_symbols):
                standardized['category'] = 'Major'
            elif any(symbol in ['BONK', 'SAMO', 'PEPE'] for symbol in token_symbols):
                standardized['category'] = 'Meme'
            elif any(symbol in ['GMT', 'ATLAS', 'POLIS'] for symbol in token_symbols):
                standardized['category'] = 'Gaming'
            elif any(symbol in ['JUP', 'RAY', 'SRM', 'ORCA'] for symbol in token_symbols):
                standardized['category'] = 'DeFi'
            else:
                standardized['category'] = 'Other'
        else:
            standardized['category'] = 'Other'
            
        standardized['version'] = 'v1'  # Default version
        
        # Time-based data
        standardized['created_at'] = pool_data.get('createdAt', datetime.now().isoformat())
        standardized['updated_at'] = pool_data.get('updatedAt', datetime.now().isoformat())
        
        # Calculate a basic prediction score
        prediction_score = 50  # Base score
        
        # APR contributes up to 20 points
        prediction_score += min(20, standardized['apr'] / 2)
        
        # Positive trends add points
        if standardized['apr_change_7d'] > 0:
            prediction_score += 5
        if standardized['apr_change_24h'] > 0:
            prediction_score += 3
            
        # Volume to liquidity ratio adds points (high volume relative to liquidity is good)
        if standardized['liquidity'] > 0:
            vol_liq_ratio = standardized['volume_24h'] / standardized['liquidity']
            if vol_liq_ratio > 0.5:
                prediction_score += 5
            elif vol_liq_ratio > 0.2:
                prediction_score += 3
                
        # Category adjustments
        if standardized['category'] == 'Meme':
            prediction_score += 7  # Higher volatility, higher potential
        elif standardized['category'] == 'DeFi':
            prediction_score += 5  # Good potential
        elif standardized['category'] == 'Major':
            prediction_score += 3  # Stable but lower ceiling
            
        # Cap at 100
        standardized['prediction_score'] = min(100, prediction_score)
        
        return standardized
    except Exception as e:
        logger.error(f"Error processing pool {pool_data.get('poolId', 'unknown')}: {str(e)}")
        return None

def write_markdown_results(pools):
    """
    Write pools data to markdown file
    
    Args:
        pools: List of processed pool data
    """
    with open(RESULTS_FILE, 'w') as f:
        f.write("# DeFi API Data Test Results\n\n")
        f.write(f"Data retrieved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Summary\n")
        f.write(f"Successfully retrieved data for {len(pools)} liquidity pools from the DeFi API.\n\n")
        
        f.write("## Sample Pool Data\n\n")
        
        # Write detailed info for first 3 pools
        for i, pool in enumerate(pools[:3]):
            f.write(f"### Pool {i+1}: {pool['name']}\n\n")
            
            f.write("#### Basic Pool Information\n")
            f.write(f"- **ID**: `{pool['id']}`\n")
            f.write(f"- **Name**: {pool['name']}\n")
            f.write(f"- **DEX**: {pool['dex']}\n")
            f.write(f"- **Token 1**: {pool['token1_symbol']} (`{pool['token1_address']}`)\n")
            f.write(f"- **Token 2**: {pool['token2_symbol']} (`{pool['token2_address']}`)\n")
            f.write(f"- **Category**: {pool['category']}\n")
            f.write(f"- **Version**: {pool['version']}\n\n")
            
            f.write("#### Financial Metrics\n")
            f.write(f"- **Liquidity**: ${pool['liquidity']:,.2f}\n")
            f.write(f"- **24h Volume**: ${pool['volume_24h']:,.2f}\n")
            f.write(f"- **APR**: {pool['apr']:.2f}%\n")
            f.write(f"- **Fee**: {pool['fee']:.2f}%\n\n")
            
            f.write("#### Historical Trend Data\n")
            f.write(f"- **APR Change (24h)**: {pool['apr_change_24h']:.2f}%\n")
            f.write(f"- **APR Change (7d)**: {pool['apr_change_7d']:.2f}%\n")
            f.write(f"- **APR Change (30d)**: {pool['apr_change_30d']:.2f}%\n")
            f.write(f"- **TVL Change (24h)**: {pool['tvl_change_24h']:.2f}%\n")
            f.write(f"- **TVL Change (7d)**: {pool['tvl_change_7d']:.2f}%\n")
            f.write(f"- **TVL Change (30d)**: {pool['tvl_change_30d']:.2f}%\n\n")
            
            f.write("#### Prediction Data\n")
            f.write(f"- **Prediction Score**: {pool['prediction_score']:.2f}\n\n")
            
        # Write table with all pools
        f.write("## All Retrieved Pools\n\n")
        f.write("| Name | DEX | Liquidity | APR | Volume 24h | Prediction Score |\n")
        f.write("|------|-----|-----------|-----|------------|------------------|\n")
        
        for pool in pools:
            f.write(f"| {pool['name']} | {pool['dex']} | ${pool['liquidity']:,.2f} | {pool['apr']:.2f}% | ${pool['volume_24h']:,.2f} | {pool['prediction_score']:.2f} |\n")
        
        f.write("\n\n## Data Fields for Prediction Models\n\n")
        f.write("The following data fields were successfully retrieved for use in prediction models:\n\n")
        
        field_categories = {
            "Basic Pool Information": [
                "id", "name", "dex", "token1_symbol", "token2_symbol", 
                "token1_address", "token2_address"
            ],
            "Financial Metrics": [
                "liquidity", "volume_24h", "apr", "fee"
            ],
            "Historical Trend Data": [
                "apr_change_24h", "apr_change_7d", "apr_change_30d",
                "tvl_change_24h", "tvl_change_7d", "tvl_change_30d"
            ],
            "Categorization": [
                "category", "version"
            ],
            "Time-based Data": [
                "created_at", "updated_at"
            ],
            "Prediction Results": [
                "prediction_score"
            ]
        }
        
        for category, fields in field_categories.items():
            f.write(f"### {category}\n")
            for field in fields:
                f.write(f"- `{field}`\n")
            f.write("\n")

def main():
    """Main function to run the test"""
    logger.info("Starting DeFi API data test")
    
    # Fetch pool data
    pools = fetch_pools(limit=10)
    
    if not pools:
        logger.error("Failed to fetch pool data. Exiting.")
        return
    
    processed_pools = []
    
    # Process each pool directly with the pool data from API
    for pool in pools:
        standardized_pool = standardize_pool_data(pool)
        if standardized_pool:
            processed_pools.append(standardized_pool)
    
    # Filter out None values
    processed_pools = [p for p in processed_pools if p is not None]
    
    if not processed_pools:
        logger.error("No pools were successfully processed. Exiting.")
        return
    
    # Write results to markdown file
    write_markdown_results(processed_pools)
    
    logger.info(f"Test completed. Results written to {RESULTS_FILE}")
    
    # Store the processed pools for database ingestion
    with open('processed_pools.json', 'w') as f:
        json.dump(processed_pools, f, indent=2)
    
    logger.info(f"Processed pools also saved to processed_pools.json")

if __name__ == "__main__":
    main()