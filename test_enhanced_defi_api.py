"""
Enhanced test for DeFi API with improved token extraction
"""

import os
import json
import time
from defi_aggregation_api import DefiAggregationAPI
from typing import Dict, List, Any
from collections import Counter
import re

class EnhancedDefiAPI(DefiAggregationAPI):
    """Enhanced version of DefiAggregationAPI with better token extraction"""
    
    def transform_pool_data(self, pool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced transform function that extracts token symbols from pool name
        if the tokens array is empty
        """
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
                    
                    print(f"Extracted token symbols from name '{name}': {token1} and {token2}")
        
        return transformed

def test_enhanced_defi_api():
    """Test connection to the DeFi API with enhanced token extraction"""
    
    print("Starting Enhanced DeFi API test...")
    start_time = time.time()
    max_time = 10  # 10 seconds maximum
    
    results = {
        "success": False,
        "time_taken": 0,
        "pools_count": 0,
        "unique_tokens": set(),
        "token_pairs": [],
        "most_common_tokens": [],
        "most_common_dexes": [],
        "error": None
    }
    
    try:
        # Initialize the enhanced API client
        print("Initializing Enhanced DeFi API client...")
        defi_api = EnhancedDefiAPI()
        
        # Get pools with enhanced token extraction
        print("Fetching pools from API with enhanced token extraction...")
        pools = defi_api.get_transformed_pools(max_pools=20)  # Limit to 20 pools
        
        # Check if we have data and it's still within our time limit
        if pools and time.time() - start_time <= max_time:
            results["success"] = True
            results["pools_count"] = len(pools)
            
            print(f"Successfully retrieved {len(pools)} pools")
            
            # Extract token information
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
    print("ENHANCED DeFi API TEST RESULTS")
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
        
        # Show all tokens
        print("\nALL UNIQUE TOKENS:")
        print(", ".join(sorted(results["unique_tokens"])))
    else:
        print(f"\n❌ Failed to connect to DeFi API: {results['error']}")
        print(f"Time taken: {results['time_taken']:.2f} seconds")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    print("Running Enhanced DeFi API test...")
    results = test_enhanced_defi_api()
    print_results(results)