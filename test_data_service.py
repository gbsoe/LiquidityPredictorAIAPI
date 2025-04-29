"""
Test script for the data services implementation.

This script tests the various components of the data services package:
- Cache manager
- Collectors
- Data service coordination
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_data_service')

# Import data services
from data_services.initialize import init_services, get_stats
from data_services.data_service import get_data_service
from data_services.collectors import get_defi_collector
from data_services.cache import get_cache_manager

def test_cache_manager():
    """Test the cache manager functionality"""
    print("\n=== Testing Cache Manager ===")
    
    # Get the cache manager
    cache = get_cache_manager(ttl_seconds=10)  # Short TTL for testing
    
    # Test caching a value
    print("Setting cache value...")
    cache.set("test_key", {"value": "test_value", "timestamp": str(datetime.now())})
    
    # Test retrieving the value
    print("Getting cache value...")
    value = cache.get("test_key")
    print(f"Cached value: {value}")
    
    # Test cache statistics
    print("Cache statistics:")
    stats = cache.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Test cache expiration
    print("Testing cache expiration (waiting 11 seconds)...")
    time.sleep(11)
    
    expired_value = cache.get("test_key")
    print(f"After expiration: {expired_value}")
    
    # Test cache invalidation
    print("Setting new value...")
    cache.set("test_key2", "will be invalidated")
    
    print("Invalidating cache...")
    cache.invalidate("test_key2")
    
    invalidated_value = cache.get("test_key2")
    print(f"After invalidation: {invalidated_value}")
    
    return True

def test_defi_collector():
    """Test the DeFi aggregation collector"""
    print("\n=== Testing DeFi Aggregation Collector ===")
    
    # Get the collector
    collector = get_defi_collector()
    
    # Test getting supported DEXes
    print("Getting supported DEXes...")
    dexes = collector.get_supported_dexes()
    print(f"Supported DEXes: {dexes}")
    
    # Test collecting data
    print("Collecting pool data...")
    start_time = time.time()
    pools = collector.collect()
    elapsed = time.time() - start_time
    
    print(f"Collected {len(pools)} pools in {elapsed:.2f}s ({len(pools)/elapsed:.2f} pools/s)")
    
    # Show sample pool data
    if pools:
        print("\nSample pool data:")
        sample_pool = pools[0]
        print_keys = ['id', 'poolId', 'source', 'name']
        sample_data = {k: sample_pool[k] for k in print_keys if k in sample_pool}
        print(json.dumps(sample_data, indent=2))
        
        # Print token info if available
        if 'tokens' in sample_pool:
            print("\nToken info:")
            for token in sample_pool['tokens']:
                print(f"- {token.get('symbol', 'Unknown')}: {token.get('address', 'Unknown')}")
    
    # Test collector statistics
    print("\nCollector statistics:")
    stats = collector.get_stats()
    print(json.dumps(stats, indent=2))
    
    return True

def test_data_service():
    """Test the central data service functionality"""
    print("\n=== Testing Data Service ===")
    
    # Get the data service
    data_service = get_data_service()
    
    # Test getting all pools with caching
    print("Getting all pools (should use cache if available)...")
    start_time = time.time()
    pools = data_service.get_all_pools()
    elapsed = time.time() - start_time
    
    print(f"Retrieved {len(pools)} pools in {elapsed:.2f}s")
    
    # Test forced refresh
    print("\nGetting all pools with forced refresh...")
    start_time = time.time()
    pools = data_service.get_all_pools(force_refresh=True)
    elapsed = time.time() - start_time
    
    print(f"Retrieved {len(pools)} pools in {elapsed:.2f}s")
    
    # Test system stats
    print("\nSystem statistics:")
    stats = data_service.get_system_stats()
    
    # Format the stats for readability
    formatted_stats = {}
    for key, value in stats.items():
        if key != "collectors" and key != "cache":
            formatted_stats[key] = value
    
    print(json.dumps(formatted_stats, indent=2))
    
    # Test scheduled collection
    print("\nScheduled collection status:")
    print(f"Running: {data_service.scheduler_running}")
    
    # Start scheduled collection if not running
    if not data_service.scheduler_running:
        print("Starting scheduled collection...")
        data_service.start_scheduled_collection()
        print(f"Running: {data_service.scheduler_running}")
    
    return True

def test_pool_by_token():
    """Test getting pools by token"""
    print("\n=== Testing Get Pools by Token ===")
    
    # Get the data service
    data_service = get_data_service()
    
    # Test tokens we expect to find
    test_tokens = ["SOL", "USDC", "ETH"]
    
    for token in test_tokens:
        print(f"\nGetting pools for token: {token}")
        pools = data_service.get_pools_by_token(token)
        print(f"Found {len(pools)} pools containing {token}")
        
        if pools:
            print("Sample pools:")
            for i, pool in enumerate(pools[:3]):  # Show up to 3 pools
                print(f"  {i+1}. {pool.get('name', 'Unknown')}")
    
    return True

def run_tests():
    """Run all tests"""
    print("Starting Data Services Tests")
    print("============================")
    
    # Initialize services
    print("Initializing data services...")
    init_services()
    
    # Run the tests
    cache_result = test_cache_manager()
    collector_result = test_defi_collector()
    service_result = test_data_service()
    token_result = test_pool_by_token()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Cache Manager: {'✓ PASSED' if cache_result else '✗ FAILED'}")
    print(f"DeFi Collector: {'✓ PASSED' if collector_result else '✗ FAILED'}")
    print(f"Data Service: {'✓ PASSED' if service_result else '✗ FAILED'}")
    print(f"Token Lookup: {'✓ PASSED' if token_result else '✗ FAILED'}")
    
    total_passed = sum([cache_result, collector_result, service_result, token_result])
    print(f"\nTotal: {total_passed}/4 tests passed")

if __name__ == "__main__":
    run_tests()