"""
Lighter version of the 60-second DeFi API test that should complete within the available execution time.
"""

import os
import json
import time
import sys
from datetime import datetime
from collections import Counter
from typing import Dict, List, Any, Set
from defi_aggregation_api import DefiAggregationAPI

class LightDefiAPITest:
    """Lightweight test class for DeFi API test for up to 60 seconds"""
    
    def __init__(self):
        # Initialize the API client
        self.api = DefiAggregationAPI()
        
        # Stats
        self.start_time = time.time()
        self.max_runtime = 60  # Run for at most 60 seconds
        self.api_calls = 0
        self.pools_retrieved = 0
        self.errors = []
        self.pools = []
        self.unique_pools = set()  # Pool IDs to track duplicates
        self.unique_tokens = set()
        self.token_pairs = []
        self.dexes = []
        
    def extract_token_symbols_from_name(self, name: str) -> (str, str):
        """Extract token symbols from pool name"""
        if not name or '-' not in name:
            return "Unknown", "Unknown"
            
        parts = name.split('-')
        if len(parts) < 2:
            return "Unknown", "Unknown"
            
        # First part is usually token1
        token1 = parts[0].strip()
        
        # Second part might have extra text (like "LP" or DEX name)
        # Try to extract just the token symbol
        token2_parts = parts[1].split(' ')
        token2 = token2_parts[0].strip()
        
        return token1, token2
        
    def process_pool(self, pool: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single pool and extract token information"""
        # Skip duplicates (same pool ID)
        pool_id = pool.get('poolId', '')
        if pool_id in self.unique_pools:
            return None
            
        # Mark as processed
        self.unique_pools.add(pool_id)
        
        # Extract pool metrics
        processed_pool = {
            'id': pool.get('id', 'Unknown'),
            'pool_id': pool_id,
            'dex': pool.get('source', 'Unknown'),
            'name': pool.get('name', 'Unknown'),
            'liquidity': pool.get('metrics', {}).get('tvl', 0),
            'volume_24h': pool.get('metrics', {}).get('volumeUsd', 0),
            'apr': pool.get('metrics', {}).get('apy24h', 0),
            'fee': pool.get('metrics', {}).get('fee', 0),
        }
        
        # Extract token symbols from name
        token1, token2 = self.extract_token_symbols_from_name(processed_pool['name'])
        processed_pool['token1_symbol'] = token1
        processed_pool['token2_symbol'] = token2
        
        # Track tokens
        if token1 != "Unknown":
            self.unique_tokens.add(token1)
        if token2 != "Unknown":
            self.unique_tokens.add(token2)
            
        # Track token pair
        token_pair = f"{token1}-{token2}"
        self.token_pairs.append(token_pair)
        
        # Track DEX
        if processed_pool['dex'] != "Unknown":
            self.dexes.append(processed_pool['dex'])
        
        return processed_pool
    
    def retrieve_pools(self, limit: int = 100):
        """Retrieve pools sequentially until time runs out"""
        print(f"Starting pool retrieval, running for up to {self.max_runtime} seconds...")
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
        
        offset = 0
        last_progress_count = 0
        
        while time.time() < self.start_time + self.max_runtime:
            try:
                # Make API request
                self.api_calls += 1
                print(f"API call #{self.api_calls} - offset: {offset}, limit: {limit}")
                
                # Call the API
                pools = self.api._make_request(f"pools", params={
                    'limit': limit,
                    'offset': offset
                })
                
                # Process response
                if not pools or not isinstance(pools, list):
                    self.errors.append(f"Invalid response for offset {offset}: {pools}")
                    break
                
                # Process each pool
                new_pools_count = 0
                for pool in pools:
                    processed_pool = self.process_pool(pool)
                    if processed_pool:
                        self.pools.append(processed_pool)
                        self.pools_retrieved += 1
                        new_pools_count += 1
                
                # Show progress
                if self.pools_retrieved > last_progress_count:
                    print(f"Retrieved {self.pools_retrieved} unique pools so far ({new_pools_count} new)")
                    last_progress_count = self.pools_retrieved
                
                # Advance offset or break if we hit the end
                if len(pools) < limit:
                    print(f"Reached end of data at offset {offset}")
                    break
                
                offset += limit
                
                # Brief sleep to avoid hammering the API
                elapsed = time.time() - self.start_time
                print(f"Elapsed time: {elapsed:.2f} seconds")
                
                # If we're about to exceed our time limit, break
                if time.time() + 2 > self.start_time + self.max_runtime:
                    print("Approaching time limit, stopping retrieval")
                    break
                    
                time.sleep(0.1)  # Small delay
                
            except Exception as e:
                self.errors.append(f"Error retrieving pools at offset {offset}: {str(e)}")
                break
    
    def print_summary(self):
        """Print a summary of the test results"""
        print("\n" + "="*60)
        print("DEFI API TEST RESULTS")
        print("="*60)
        
        total_time = time.time() - self.start_time
        
        print(f"\nTest duration: {total_time:.2f} seconds")
        print(f"API calls: {self.api_calls}")
        print(f"Unique pools: {self.pools_retrieved}")
        
        if self.api_calls > 0:
            print(f"Avg. time per API call: {total_time / self.api_calls:.2f} seconds")
            print(f"Avg. pools per API call: {self.pools_retrieved / self.api_calls:.2f}")
        
        print(f"Unique tokens: {len(self.unique_tokens)}")
        
        if self.errors:
            print(f"\nEncountered {len(self.errors)} errors:")
            for i, error in enumerate(self.errors[:3], 1):
                print(f"  {i}. {error}")
            if len(self.errors) > 3:
                print(f"  ... and {len(self.errors) - 3} more")
        
        if not self.pools:
            print("No pools were retrieved!")
            return
        
        # Pool metrics
        total_liquidity = sum(p.get('liquidity', 0) for p in self.pools)
        total_volume = sum(p.get('volume_24h', 0) for p in self.pools)
        avg_apr = sum(p.get('apr', 0) for p in self.pools) / len(self.pools) if self.pools else 0
        
        print(f"\nPOOL METRICS:")
        print(f"Total Liquidity: ${total_liquidity:,.0f}")
        print(f"Total 24h Volume: ${total_volume:,.0f}")
        print(f"Average APR: {avg_apr:.2f}%")
        
        # Token analysis
        token_counts = Counter([p.get('token1_symbol') for p in self.pools] + 
                              [p.get('token2_symbol') for p in self.pools])
        token_pair_counts = Counter(self.token_pairs)
        dex_counts = Counter(self.dexes)
        
        print("\nTOP TOKENS:")
        for token, count in token_counts.most_common(10):
            if token != "Unknown":
                print(f"  {token}: {count} occurrences")
        
        print("\nTOP TOKEN PAIRS:")
        for pair, count in token_pair_counts.most_common(10):
            if "Unknown" not in pair:
                print(f"  {pair}: {count} pools")
        
        print("\nDEXES:")
        for dex, count in dex_counts.most_common():
            print(f"  {dex}: {count} pools")
            
        # All unique tokens
        print("\nALL UNIQUE TOKENS:")
        tokens_sorted = sorted(list(self.unique_tokens))
        tokens_display = ", ".join(tokens_sorted)
        if len(tokens_display) > 100:
            print(f"  {tokens_display[:100]}... (and {len(tokens_sorted) - len(tokens_display[:100].split(', '))} more)")
        else:
            print(f"  {tokens_display}")
        
        print("\n" + "="*60)
    
    def run_test(self):
        """Run the complete test"""
        try:
            self.retrieve_pools(limit=50)  # We can get more pools with a higher limit
            self.print_summary()
            return True
        except Exception as e:
            print(f"Test failed: {str(e)}")
            return False

if __name__ == "__main__":
    # Run the light test
    test = LightDefiAPITest()
    test.run_test()