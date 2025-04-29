"""
Extended test of the DeFi API that runs continuously for 60 seconds,
retrieving as much pool and token data as possible.
"""

import os
import json
import time
import threading
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any, Set
from defi_aggregation_api import DefiAggregationAPI

class ExtendedDefiAPITest:
    """Test class for running continuous DeFi API tests for 60 seconds"""
    
    def __init__(self):
        # Initialize the API client
        self.api = DefiAggregationAPI()
        
        # Stats
        self.start_time = time.time()
        self.end_time = self.start_time + 60  # Run for 60 seconds
        self.api_calls = 0
        self.pools_retrieved = 0
        self.errors = []
        self.pools = []
        self.unique_pools = set()  # Pool IDs to track duplicates
        self.unique_tokens = set()
        self.token_pairs = []
        self.dexes = []
        self.pool_retrieval_times = []
        
        # Concurrent tracking
        self.lock = threading.Lock()
        
        # For token extraction from pool names
        self.token_name_mapping = {}  # Maps names from pool titles to actual tokens
        
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
    
    def retrieve_pools_batch(self, limit: int = 20, offset: int = 0):
        """Retrieve a batch of pools from the API"""
        try:
            start_batch = time.time()
            
            # Call the API
            with self.lock:
                self.api_calls += 1
                
            pools = self.api._make_request(f"pools", params={
                'limit': limit,
                'offset': offset
            })
            
            if not pools or not isinstance(pools, list):
                with self.lock:
                    self.errors.append(f"Invalid response for offset {offset}: {pools}")
                return 0
                
            # Process pools
            processed_count = 0
            for pool in pools:
                if time.time() > self.end_time:
                    # Stop if we've exceeded our time limit
                    break
                
                processed_pool = self.process_pool(pool)
                if processed_pool:
                    with self.lock:
                        self.pools.append(processed_pool)
                        self.pools_retrieved += 1
                        processed_count += 1
            
            # Track timing
            batch_time = time.time() - start_batch
            with self.lock:
                self.pool_retrieval_times.append({
                    'offset': offset,
                    'limit': limit,
                    'count': processed_count,
                    'time': batch_time
                })
            
            return processed_count
        except Exception as e:
            with self.lock:
                self.errors.append(f"Error retrieving pools at offset {offset}: {str(e)}")
            return 0
    
    def run_continuous_retrieval(self):
        """Run continuous pool retrieval for 60 seconds"""
        print(f"Starting continuous pool retrieval for 60 seconds at {datetime.now().strftime('%H:%M:%S')}...")
        print("Retrieving pools in batches...")
        
        offset = 0
        batch_size = 20  # Adjust as needed
        
        while time.time() < self.end_time:
            # Check if we've been running for at least a certain period
            elapsed = time.time() - self.start_time
            
            # Print progress
            if int(elapsed) % 5 == 0:  # Every 5 seconds
                print(f"Elapsed: {int(elapsed)}s, Pools: {self.pools_retrieved}, API calls: {self.api_calls}")
            
            # Retrieve a batch of pools
            count = self.retrieve_pools_batch(limit=batch_size, offset=offset)
            
            # If we got a full batch, advance offset
            if count == batch_size:
                offset += batch_size
            else:
                # Otherwise, start from the beginning again
                offset = 0
            
            # Small delay to avoid hammering the API
            time.sleep(0.1)
        
        # Final stats
        total_elapsed = time.time() - self.start_time
        print(f"\nCompleted after {total_elapsed:.2f} seconds.")
        print(f"Made {self.api_calls} API calls")
        print(f"Retrieved {self.pools_retrieved} unique pools")
        print(f"Found {len(self.unique_tokens)} unique tokens")
        
        if self.errors:
            print(f"\nEncountered {len(self.errors)} errors:")
            for i, error in enumerate(self.errors[:5], 1):  # Show only first 5 errors
                print(f"  {i}. {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
    
    def print_summary(self):
        """Print a summary of the test results"""
        if not self.pools:
            print("No pools were retrieved!")
            return
            
        print("\n" + "="*60)
        print("DEFI API 60-SECOND TEST RESULTS")
        print("="*60)
        
        total_time = time.time() - self.start_time
        avg_time_per_api_call = total_time / self.api_calls if self.api_calls > 0 else 0
        avg_pools_per_call = self.pools_retrieved / self.api_calls if self.api_calls > 0 else 0
        
        print(f"\nTotal test duration: {total_time:.2f} seconds")
        print(f"Total API calls: {self.api_calls}")
        print(f"Total unique pools: {self.pools_retrieved}")
        print(f"Avg. time per API call: {avg_time_per_api_call:.2f} seconds")
        print(f"Avg. pools per API call: {avg_pools_per_call:.2f}")
        print(f"Unique tokens found: {len(self.unique_tokens)}")
        
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
        
        # API performance
        print("\nAPI PERFORMANCE:")
        # Calculate average time per batch
        avg_batch_time = sum(batch['time'] for batch in self.pool_retrieval_times) / len(self.pool_retrieval_times) if self.pool_retrieval_times else 0
        max_batch_time = max(batch['time'] for batch in self.pool_retrieval_times) if self.pool_retrieval_times else 0
        min_batch_time = min(batch['time'] for batch in self.pool_retrieval_times) if self.pool_retrieval_times else 0
        
        print(f"Average batch retrieval time: {avg_batch_time:.2f} seconds")
        print(f"Fastest batch: {min_batch_time:.2f} seconds")
        print(f"Slowest batch: {max_batch_time:.2f} seconds")
        
        print("\n" + "="*60)
        
    def save_results(self, filename: str = "defi_api_60sec_results.json"):
        """Save results to file"""
        try:
            results = {
                "test_info": {
                    "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                    "duration": time.time() - self.start_time,
                    "api_calls": self.api_calls,
                    "pools_retrieved": self.pools_retrieved,
                    "unique_tokens": list(self.unique_tokens),
                    "error_count": len(self.errors)
                },
                "pools": self.pools[:100],  # Save only first 100 pools to avoid huge files
                "token_counts": dict(Counter([p.get('token1_symbol') for p in self.pools] + 
                                    [p.get('token2_symbol') for p in self.pools]).most_common(20)),
                "token_pairs": dict(Counter(self.token_pairs).most_common(20)),
                "dexes": dict(Counter(self.dexes).most_common()),
                "performance": {
                    "retrieval_times": self.pool_retrieval_times[:20]  # Save only sample
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"\nResults saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    # Run the extended test
    test = ExtendedDefiAPITest()
    test.run_continuous_retrieval()
    test.print_summary()
    test.save_results()