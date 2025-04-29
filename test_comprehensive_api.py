"""
Comprehensive test of DeFi API with both pool data and token metadata
"""

import os
import json
import time
import requests
from defi_aggregation_api import DefiAggregationAPI
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import pprint

class ComprehensiveDefiAPI(DefiAggregationAPI):
    """Enhanced API client with comprehensive data retrieval"""
    
    def transform_pool_data(self, pool: Dict[str, Any]) -> Dict[str, Any]:
        """Transform pool data and extract token symbols from name if needed"""
        # Call the original transform function
        transformed = super().transform_pool_data(pool)
        
        # If token symbols are unknown, try to extract from name
        if transformed.get('token1_symbol') == 'Unknown' or transformed.get('token2_symbol') == 'Unknown':
            name = pool.get('name', '')
            if name and '-' in name:
                # Extract token symbols from the name
                parts = name.split('-')
                if len(parts) >= 2:
                    # First part is usually token1
                    token1 = parts[0].strip()
                    
                    # Second part might have extra text (like "LP" or DEX name)
                    # Try to extract just the token symbol
                    token2_parts = parts[1].split(' ')
                    token2 = token2_parts[0].strip()
                    
                    # Update the transformed data
                    transformed['token1_symbol'] = token1
                    transformed['token2_symbol'] = token2
        
        return transformed
    
    def get_token_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get token metadata from the API
        
        Args:
            symbol: Token symbol
            
        Returns:
            Token metadata or None if not found
        """
        try:
            url = f"{self.base_url}/tokens/{symbol}"
            
            # Make API request with rate limiting
            response = requests.get(url, headers=self.headers)
            
            # Introduce delay to respect rate limits
            time.sleep(self.request_delay)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"Error getting token metadata for {symbol}: {str(e)}")
            return None
    
    def enrich_pool_with_token_data(self, pool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich pool data with token metadata
        
        Args:
            pool: Transformed pool data
            
        Returns:
            Pool data with additional token metadata
        """
        enriched_pool = pool.copy()
        
        # Get token symbols
        token1_symbol = pool.get('token1_symbol')
        token2_symbol = pool.get('token2_symbol')
        
        # Only proceed if we have actual token symbols
        if token1_symbol and token1_symbol != 'Unknown':
            token1_data = self.get_token_metadata(token1_symbol)
            if token1_data:
                enriched_pool['token1_metadata'] = token1_data
                enriched_pool['token1_price'] = token1_data.get('price', 0)
                enriched_pool['token1_coingecko_id'] = token1_data.get('coingeckoId', '')
        
        if token2_symbol and token2_symbol != 'Unknown':
            token2_data = self.get_token_metadata(token2_symbol)
            if token2_data:
                enriched_pool['token2_metadata'] = token2_data
                enriched_pool['token2_price'] = token2_data.get('price', 0)
                enriched_pool['token2_coingecko_id'] = token2_data.get('coingeckoId', '')
        
        return enriched_pool

def test_comprehensive_api():
    """Run a comprehensive test of the DeFi API"""
    
    print("Starting Comprehensive DeFi API Test...")
    start_time = time.time()
    max_time = 10  # 10 seconds maximum
    
    # Initialize results dictionary
    results = {
        "success": False,
        "time_taken": 0,
        "pools_count": 0,
        "unique_tokens": set(),
        "token_metadata": {},
        "token_pairs": [],
        "most_common_tokens": [],
        "most_common_dexes": [],
        "error": None
    }
    
    try:
        # Initialize the comprehensive API client
        print("Initializing Comprehensive DeFi API client...")
        api = ComprehensiveDefiAPI()
        
        # Get a limited number of pools
        print("Fetching pools from API...")
        pools = api.get_transformed_pools(max_pools=10)  # Limit to 10 pools for speed
        
        if pools and time.time() - start_time <= max_time:
            results["success"] = True
            results["pools_count"] = len(pools)
            
            print(f"Successfully retrieved {len(pools)} pools")
            
            # Extract token symbols and track metadata
            token_symbols = []
            token_pairs = []
            dexes = []
            
            print("Analyzing pool data...")
            for pool in pools:
                # Extract token symbols
                token1 = pool.get("token1_symbol", "Unknown")
                token2 = pool.get("token2_symbol", "Unknown")
                
                if token1 != "Unknown":
                    results["unique_tokens"].add(token1)
                    token_symbols.append(token1)
                
                if token2 != "Unknown":
                    results["unique_tokens"].add(token2)
                    token_symbols.append(token2)
                
                # Create token pair
                token_pairs.append(f"{token1}-{token2}")
                
                # Extract DEX
                dex = pool.get("dex", "Unknown")
                if dex != "Unknown":
                    dexes.append(dex)
            
            # Try to get metadata for a couple of tokens if time permits
            unique_tokens = list(results["unique_tokens"])
            if time.time() - start_time <= max_time - 3 and unique_tokens:  # Leave 3 seconds for API calls
                print(f"Getting metadata for {min(3, len(unique_tokens))} tokens...")
                token_sample = unique_tokens[:3]  # Take up to 3 tokens for demonstration
                
                for token in token_sample:
                    token_metadata = api.get_token_metadata(token)
                    if token_metadata:
                        results["token_metadata"][token] = token_metadata
                        print(f"Retrieved metadata for token: {token}")
            
            # Convert set to list for JSON serialization
            results["unique_tokens"] = list(results["unique_tokens"])
            
            # Count frequency of tokens and find most common
            token_counts = Counter(token_symbols)
            results["most_common_tokens"] = token_counts.most_common(5)
            
            # Count frequency of token pairs and add to results
            pair_counts = Counter(token_pairs)
            results["token_pairs"] = pair_counts.most_common(5)
            
            # Count frequency of DEXes and add to results
            dex_counts = Counter(dexes)
            results["most_common_dexes"] = dex_counts.most_common(5)
        
        # Add timing information
        results["time_taken"] = time.time() - start_time
        
    except Exception as e:
        results["error"] = str(e)
        results["time_taken"] = time.time() - start_time
        print(f"Error during API test: {str(e)}")
    
    return results

def print_results(results):
    """Print the results in a formatted way"""
    print("\n" + "="*50)
    print("COMPREHENSIVE DeFi API TEST RESULTS")
    print("="*50)
    
    if results["success"]:
        print(f"\n✅ Successfully connected to DeFi API and retrieved data in {results['time_taken']:.2f} seconds")
        
        # Display summary information
        print("\nSUMMARY:")
        print(f"🏊 Total Pools: {results['pools_count']}")
        print(f"🪙 Unique Tokens: {len(results['unique_tokens'])}")
        
        # Display token information
        print("\nTOP TOKENS:")
        for token, count in results["most_common_tokens"]:
            print(f"  {token}: {count}")
        
        # Display token pair information
        print("\nTOP TOKEN PAIRS:")
        for pair, count in results["token_pairs"]:
            print(f"  {pair}: {count}")
        
        # Display DEX information
        print("\nTOP DEXes:")
        for dex, count in results["most_common_dexes"]:
            print(f"  {dex}: {count}")
        
        # Display token metadata
        if results["token_metadata"]:
            print("\nTOKEN METADATA:")
            for token, metadata in results["token_metadata"].items():
                print(f"\n  {token}:")
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        print(f"    {key}: {value}")
                else:
                    print(f"    {metadata}")
        
        # Show all tokens
        print("\nALL UNIQUE TOKENS:")
        print(", ".join(sorted(results["unique_tokens"])))
    else:
        print(f"\n❌ Failed to connect to DeFi API: {results['error']}")
        print(f"Time taken: {results['time_taken']:.2f} seconds")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    print("Running Comprehensive DeFi API test...")
    results = test_comprehensive_api()
    print_results(results)