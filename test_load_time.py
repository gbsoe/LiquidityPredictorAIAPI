"""
Test script to measure loading time from system initialization to prediction delivery.

This script tracks and reports:
1. Time to initialize components
2. Time to load token data
3. Time to load pool data
4. Time to generate first prediction
5. Total time from start to prediction availability

Usage:
    python test_load_time.py
"""

import time
import logging
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("load_time_test")

# Import performance monitor
from performance_monitor import get_performance_monitor

# Start performance tracking
monitor = get_performance_monitor()
monitor.start_tracking("full_system_test")
monitor.mark_checkpoint("test_started")

# Log the test start
logger.info("=== LOAD TIME TEST STARTED ===")
logger.info(f"Test started at: {datetime.now().isoformat()}")

# Track initialization time
monitor.start_tracking("system_initialization")
logger.info("Initializing system components...")

# Import necessary components with timing
import_start = time.time()
try:
    # Import core modules with timing
    monitor.start_tracking("import_modules")
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import streamlit as st  # For testing UI components
    monitor.stop_tracking("import_modules")
    logger.info(f"Core modules imported in {time.time() - import_start:.2f}s")
    
    # Import app-specific modules
    monitor.start_tracking("import_app_modules")
    from token_data_service import get_token_service
    from defi_aggregation_api import DefiAggregationAPI
    from token_price_service import TokenPriceService
    from data_services.data_service import get_data_service
    
    # Try to import prediction modules if available
    try:
        from advanced_prediction_engine import PredictionEngine
        has_prediction_engine = True
    except ImportError:
        logger.warning("Advanced prediction engine not available")
        has_prediction_engine = False
        
    monitor.stop_tracking("import_app_modules")
    
except Exception as e:
    logger.error(f"Error during imports: {str(e)}")
    monitor.mark_checkpoint("import_error")
    raise

# Initialize main components with timing
monitor.mark_checkpoint("begin_service_initialization")

# Initialize API client
monitor.start_tracking("api_client_initialization")
api_client = DefiAggregationAPI()
monitor.stop_tracking("api_client_initialization")

# Initialize token service
monitor.start_tracking("token_service_initialization")
token_service = get_token_service()
monitor.stop_tracking("token_service_initialization")

# Initialize token price service
monitor.start_tracking("price_service_initialization")
price_service = TokenPriceService(token_service)
monitor.stop_tracking("price_service_initialization")

# Initialize data service
monitor.start_tracking("data_service_initialization")
data_service = get_data_service()
monitor.stop_tracking("data_service_initialization")

# Initialize prediction engine if available
if has_prediction_engine:
    monitor.start_tracking("prediction_engine_initialization")
    prediction_engine = PredictionEngine()
    monitor.stop_tracking("prediction_engine_initialization")

# Mark completion of initialization phase
monitor.stop_tracking("system_initialization")
monitor.mark_checkpoint("system_initialized")
logger.info(f"System initialized in {monitor.time_between_checkpoints('test_started', 'system_initialized'):.2f}s")

# Load token data
monitor.start_tracking("token_data_loading")
logger.info("Loading token data...")
token_count = len(token_service.token_cache)
logger.info(f"Initial token cache has {token_count} tokens")

# Trigger token preloading
token_service.preload_all_tokens()
monitor.stop_tracking("token_data_loading")
monitor.mark_checkpoint("token_data_loaded")

new_token_count = len(token_service.token_cache)
logger.info(f"Token data loaded in {monitor.time_between_checkpoints('token_data_loading_start', 'token_data_loaded'):.2f}s")
logger.info(f"Token cache now has {new_token_count} tokens (+{new_token_count - token_count} tokens)")

# Get token stats for reporting
token_stats = token_service.get_stats() if hasattr(token_service, 'get_stats') else {}
monitor.update_system_stats({
    "token_cache_size": token_stats.get("cache_size", new_token_count),
    "tokens_by_symbol": token_stats.get("tokens_by_symbol", 0),  
    "tokens_by_address": token_stats.get("tokens_by_address", 0)
})

# Load pool data
monitor.start_tracking("pool_data_loading")
logger.info("Loading pool data...")

try:
    # Use the data service to get pools
    monitor.start_tracking("get_pools")
    pool_collector = data_service.get_collector()
    # Get pools from supported DEXes
    dexes = ["Raydium", "Orca", "Meteora"]
    pools = []
    for dex in dexes:
        pools.extend(pool_collector.get_pools_by_dex(dex, limit=20))
    pool_count = len(pools)
    monitor.stop_tracking("get_pools")
    logger.info(f"Successfully loaded {pool_count} pools")
    
    # Update pool data with token information
    monitor.start_tracking("pool_token_enrichment")
    enriched_pools = []
    for pool in pools:
        if hasattr(token_service, 'update_pool_token_data'):
            enriched_pool = token_service.update_pool_token_data(pool)
        else:
            enriched_pool = pool
        enriched_pools.append(enriched_pool)
    monitor.stop_tracking("pool_token_enrichment")
    
    monitor.update_system_stats({
        "total_pools_loaded": pool_count
    })
    
except Exception as e:
    logger.error(f"Error loading pool data: {str(e)}")
    pool_count = 0
    enriched_pools = []

monitor.stop_tracking("pool_data_loading")
monitor.mark_checkpoint("pool_data_loaded")
logger.info(f"Pool data loaded and enriched in {monitor.time_between_checkpoints('pool_data_loading_start', 'pool_data_loaded'):.2f}s")

# Mark data loading completion
monitor.mark_checkpoint("data_loaded")
logger.info(f"All data loaded in {monitor.time_between_checkpoints('test_started', 'data_loaded'):.2f}s")

# For testing purposes only, implement a simple prediction function
def simple_predict_performance(pool):
    """
    A very simple prediction function for testing purposes only.
    Returns a mock prediction score based on pool data.
    """
    try:
        # Extract metrics if available
        liquidity = pool.get('metrics', {}).get('tvl', 0)
        volume = pool.get('metrics', {}).get('volume', {}).get('h24', 0)
        apr = pool.get('metrics', {}).get('apr', {}).get('h24', 0)
        
        # Basic prediction (just for testing load times)
        score = min(100, (liquidity * 0.0001) + (volume * 0.001) + (apr * 0.1))
        return {
            'score': score,
            'confidence': 0.85,
            'recommendation': 'HOLD',
            'potential_apy': apr * 0.9
        }
    except Exception:
        # Fallback for testing
        return {
            'score': 50,
            'confidence': 0.5,
            'recommendation': 'NEUTRAL',
            'potential_apy': 5.0
        }

# Run predictions for testing
if pool_count > 0:
    monitor.start_tracking("prediction_generation")
    logger.info("Generating predictions...")
    
    try:
        # Sample 5 pools for prediction to simulate real usage
        test_pools = enriched_pools[:5] if len(enriched_pools) >= 5 else enriched_pools
        
        for i, pool in enumerate(test_pools):
            pred_start = time.time()
            # Use simple prediction function or PredictionEngine if available
            if has_prediction_engine:
                prediction = prediction_engine.predict_pool_performance(pool)
            else:
                prediction = simple_predict_performance(pool)
            pred_time = time.time() - pred_start
            
            logger.info(f"Generated prediction for pool {i+1}/{len(test_pools)} in {pred_time:.4f}s")
            
            # Add first prediction time as a checkpoint
            if i == 0:
                monitor.mark_checkpoint("first_prediction_complete")
        
        monitor.update_system_stats({
            "predictions_generated": len(test_pools)
        })
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
    
    monitor.stop_tracking("prediction_generation")
    monitor.mark_checkpoint("predictions_ready")
    
    time_to_first_prediction = monitor.time_between_checkpoints('test_started', 'first_prediction_complete')
    logger.info(f"Time to first prediction: {time_to_first_prediction:.2f}s")
    logger.info(f"All predictions generated in {monitor.time_between_checkpoints('prediction_generation_start', 'predictions_ready'):.2f}s")
else:
    logger.warning("Skipping prediction tests (no pools loaded)")
    # Still mark predictions_ready for consistency in reporting
    monitor.mark_checkpoint("predictions_ready")

# Simulate UI readiness (since we can't actually measure Streamlit rendering)
monitor.mark_checkpoint("ui_ready")

# Complete the test
monitor.stop_tracking("full_system_test")
monitor.mark_checkpoint("test_completed")

# Calculate final metrics
total_time = monitor.time_between_checkpoints('test_started', 'test_completed')
time_to_data = monitor.time_between_checkpoints('test_started', 'data_loaded')
time_to_predictions = monitor.time_between_checkpoints('test_started', 'predictions_ready')

# Generate and save the final report
logger.info("Generating performance report...")
report = monitor.get_report()

# Print summary
logger.info("=== LOAD TIME TEST COMPLETED ===")
logger.info(f"Total test duration: {total_time:.2f}s")
logger.info(f"Time to data loaded: {time_to_data:.2f}s")
logger.info(f"Time to predictions ready: {time_to_predictions:.2f}s")
logger.info(f"Token cache size: {new_token_count} tokens")
logger.info(f"Pools loaded: {pool_count} pools")

# Save report to file
report_file = monitor.save_final_report()
logger.info(f"Full performance report saved to: {report_file}")

# Also save a simplified report for quick reference
simplified_report = {
    "timestamp": datetime.now().isoformat(),
    "total_time": f"{total_time:.2f}s",
    "time_to_data_loaded": f"{time_to_data:.2f}s",
    "time_to_predictions": f"{time_to_predictions:.2f}s",
    "token_cache_size": new_token_count,
    "pools_loaded": pool_count,
    "key_segments": {
        "system_initialization": f"{monitor.segments.get('system_initialization', 0):.2f}s",
        "token_data_loading": f"{monitor.segments.get('token_data_loading', 0):.2f}s",
        "pool_data_loading": f"{monitor.segments.get('pool_data_loading', 0):.2f}s",
        "prediction_generation": f"{monitor.segments.get('prediction_generation', 0):.2f}s"
    }
}

simplified_report_file = os.path.join("logs", "load_time_summary.json")
os.makedirs(os.path.dirname(simplified_report_file), exist_ok=True)

with open(simplified_report_file, 'w') as f:
    json.dump(simplified_report, f, indent=2)

logger.info(f"Simplified report saved to: {simplified_report_file}")
logger.info(f"To view the report: cat {simplified_report_file}")