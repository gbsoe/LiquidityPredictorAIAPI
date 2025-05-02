"""
Token Loading Performance Test

This script measures the performance of token data loading,
focusing specifically on the token service's ability to load
token data from the API and resolve token symbols and addresses.
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('token_load_test')

# Initialize test
logger.info("=== TOKEN LOADING PERFORMANCE TEST STARTED ===")
start_time = time.time()

# Import performance monitoring
from performance_monitor import PerformanceMonitor
monitor = PerformanceMonitor()
monitor.start_tracking("token_load_test")
monitor.mark_checkpoint("test_started")

# Import token service
from token_data_service import get_token_service
from defi_aggregation_api import DefiAggregationAPI

# Initialize API client
api_client = DefiAggregationAPI()

# Initialize token service with time tracking
monitor.start_tracking("token_service_initialization")
token_service = get_token_service()
initial_cache_size = len(token_service.token_cache)
monitor.stop_tracking("token_service_initialization")
logger.info(f"Token service initialized with {initial_cache_size} initial tokens")

# Load token basics (symbols, names, addresses) with time tracking
monitor.start_tracking("token_basics_loading")
try:
    logger.info("Loading token basics...")
    token_service.preload_token_basics()
    logger.info("Token basics loading initiated")
except Exception as e:
    logger.error(f"Error loading token basics: {str(e)}")

# Wait a bit for async operations to complete
time.sleep(2)
basics_cache_size = len(token_service.token_cache)
monitor.stop_tracking("token_basics_loading")
logger.info(f"Token basics loaded, cache now has {basics_cache_size} tokens (+{basics_cache_size - initial_cache_size})")

# Test token symbol resolution
monitor.start_tracking("token_symbol_resolution")
test_symbols = ["SOL", "USDC", "BTC", "ETH", "UNKNOWN_TOKEN"]
logger.info(f"Testing resolution of {len(test_symbols)} token symbols")

symbol_success = 0
for symbol in test_symbols:
    try:
        start = time.time()
        token = token_service.get_token_by_symbol(symbol)
        duration = time.time() - start
        
        if token:
            symbol_success += 1
            logger.info(f"Resolved '{symbol}' to {token.get('name', 'Unknown')} ({token.get('address', 'No address')}) in {duration:.4f}s")
        else:
            logger.info(f"Failed to resolve '{symbol}' in {duration:.4f}s")
            
    except Exception as e:
        logger.error(f"Error resolving symbol '{symbol}': {str(e)}")

monitor.stop_tracking("token_symbol_resolution")
logger.info(f"Symbol resolution complete: {symbol_success}/{len(test_symbols)} successful")

# Test token address resolution
monitor.start_tracking("token_address_resolution")
test_addresses = [
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "So11111111111111111111111111111111111111112",   # SOL
    "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E", # BTC 
    "UNKNOWN_ADDRESS"
]
logger.info(f"Testing resolution of {len(test_addresses)} token addresses")

address_success = 0
for address in test_addresses:
    try:
        start = time.time()
        token = token_service.get_token_by_address(address)
        duration = time.time() - start
        
        if token:
            address_success += 1
            logger.info(f"Resolved '{address}' to {token.get('symbol', 'Unknown')} ({token.get('name', 'No name')}) in {duration:.4f}s")
        else:
            logger.info(f"Failed to resolve '{address}' in {duration:.4f}s")
            
    except Exception as e:
        logger.error(f"Error resolving address '{address}': {str(e)}")

monitor.stop_tracking("token_address_resolution")
logger.info(f"Address resolution complete: {address_success}/{len(test_addresses)} successful")

# Check stats
stats = token_service.get_stats() if hasattr(token_service, 'get_stats') else {}
monitor.update_system_stats({
    "token_cache_size": len(token_service.token_cache),
    "tokens_by_symbol": stats.get("tokens_by_symbol", 0),
    "tokens_by_address": stats.get("tokens_by_address", 0),
    "cache_hits": stats.get("cache_hits", 0),
    "cache_misses": stats.get("cache_misses", 0)
})

# Complete the test
monitor.stop_tracking("token_load_test")
monitor.mark_checkpoint("test_completed")

# Calculate metrics
total_time = monitor.time_between_checkpoints('test_started', 'test_completed')
final_cache_size = len(token_service.token_cache)

# Generate and save the report
report = monitor.get_report()
report_file = monitor.save_final_report()

# Print summary
logger.info("=== TOKEN LOADING PERFORMANCE TEST COMPLETED ===")
logger.info(f"Total test duration: {total_time:.2f}s")
logger.info(f"Initial token cache size: {initial_cache_size}")
logger.info(f"Final token cache size: {final_cache_size} (+{final_cache_size - initial_cache_size})")
logger.info(f"Symbol resolution success: {symbol_success}/{len(test_symbols)}")
logger.info(f"Address resolution success: {address_success}/{len(test_addresses)}")
logger.info(f"Full report saved to: {report_file}")