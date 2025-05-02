"""
Simple Load Time Test Script for SolPool Insight

This is a simplified version of the load time test to focus only on
the essential metrics without connecting to external services.
"""

import os
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('simple_load_test')

# Initialize test
logger.info("=== SIMPLE LOAD TIME TEST STARTED ===")
logger.info(f"Test started at: {datetime.now().isoformat()}")
start_time = time.time()

# Import performance monitoring
logger.info("Importing performance monitor...")
from performance_monitor import PerformanceMonitor
monitor = PerformanceMonitor()
monitor.start_tracking("simple_load_test")
monitor.mark_checkpoint("test_started")

# Import core modules with timing
logger.info("Importing core modules...")
import_start = time.time()
import os
import sys
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
logger.info(f"Core modules imported in {time.time() - import_start:.2f}s")

# Import app-specific modules
logger.info("Importing app modules...")
monitor.start_tracking("import_app_modules")
from token_data_service import get_token_service
from defi_aggregation_api import DefiAggregationAPI
from token_price_service import TokenPriceService
monitor.stop_tracking("import_app_modules")
logger.info("App modules imported")

# Initialize API client
logger.info("Initializing API client...")
monitor.start_tracking("api_client_initialization")
api_client = DefiAggregationAPI()
monitor.stop_tracking("api_client_initialization")
logger.info("API client initialized")

# Initialize token service
logger.info("Initializing token service...")
monitor.start_tracking("token_service_initialization")
token_service = get_token_service()
monitor.stop_tracking("token_service_initialization")
logger.info("Token service initialized")

# Initialize token price service
logger.info("Initializing price service...")
monitor.start_tracking("price_service_initialization")
price_service = TokenPriceService()
monitor.stop_tracking("price_service_initialization")
logger.info("Price service initialized")

# Load token data
logger.info("Loading token data...")
monitor.start_tracking("token_data_loading")
token_count = len(token_service.token_cache)
logger.info(f"Initial token cache has {token_count} tokens")

# Trigger token preloading (without waiting for completion)
try:
    token_service.preload_token_basics()
    logger.info("Token preloading initiated")
except Exception as e:
    logger.error(f"Error preloading tokens: {str(e)}")

monitor.stop_tracking("token_data_loading")
monitor.mark_checkpoint("token_data_loaded")

# Mark data loading completion
monitor.mark_checkpoint("data_loaded")
logger.info(f"All data loaded in {monitor.time_between_checkpoints('test_started', 'data_loaded'):.2f}s")

# Complete the test
monitor.stop_tracking("simple_load_test")
monitor.mark_checkpoint("test_completed")

# Calculate final metrics
total_time = monitor.time_between_checkpoints('test_started', 'test_completed')
time_to_data = monitor.time_between_checkpoints('test_started', 'data_loaded')

# Generate and save the final report
logger.info("Generating performance report...")
report = monitor.get_report()

# Print summary
logger.info("=== SIMPLE LOAD TIME TEST COMPLETED ===")
logger.info(f"Total test duration: {total_time:.2f}s")
logger.info(f"Time to data loaded: {time_to_data:.2f}s")

# Save simplified report
simplified_report = {
    "timestamp": datetime.now().isoformat(),
    "total_time": f"{total_time:.2f}s",
    "time_to_data_loaded": f"{time_to_data:.2f}s",
    "key_segments": {
        "api_client_initialization": f"{monitor.segments.get('api_client_initialization', 0):.2f}s",
        "token_service_initialization": f"{monitor.segments.get('token_service_initialization', 0):.2f}s",
        "token_data_loading": f"{monitor.segments.get('token_data_loading', 0):.2f}s"
    }
}

simplified_report_file = os.path.join("logs", "simple_load_time.json")
os.makedirs(os.path.dirname(simplified_report_file), exist_ok=True)

with open(simplified_report_file, 'w') as f:
    json.dump(simplified_report, f, indent=2)

logger.info(f"Simplified report saved to: {simplified_report_file}")