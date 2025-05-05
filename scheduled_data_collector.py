#!/usr/bin/env python3

# Scheduled Data Collector for SolPool Insight
# This script runs continuously and collects data from various sources at regular intervals
# It ensures that the database is always up to date with the latest pool information

import os
import time
import logging
import random
import schedule
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import internal modules
from database.db_operations import store_pool_snapshot, get_db_manager
from data_services.data_service import get_data_service
from token_price_service import get_token_price_service
from performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('background_updater.log')
    ]
)

logger = logging.getLogger('scheduled_data_collector')

# Global performance monitor
perf_monitor = PerformanceMonitor('data_collection')

# Data collection frequency (in minutes)
DEFAULT_COLLECTION_FREQUENCY = 5  # Every 5 minutes by default
OFF_HOURS_COLLECTION_FREQUENCY = 15  # Less frequent during off-hours
TOKEN_PRICE_FREQUENCY = 10  # Token prices every 10 minutes
PREDICTION_FREQUENCY = 60  # Generate predictions every hour

# Time zone considerations
UTC_OFFSET = 0  # Adjust based on your preferred time zone

# Market hours (UTC) - Crypto markets are 24/7, but we can optimize collection frequency
MARKET_HOURS_START = 0  # 00:00 UTC
MARKET_HOURS_END = 24  # 24:00 UTC

# Global control flag
keep_running = True

# Cooldown system to avoid hammering APIs during errors
api_cooldown = {
    'data_service': {'last_error': None, 'cooldown_until': None},
    'price_service': {'last_error': None, 'cooldown_until': None}
}

# Lock for thread safety when updating the database
db_lock = threading.Lock()


def is_market_hours() -> bool:
    """Check if current time is within active market hours"""
    now = datetime.utcnow() + timedelta(hours=UTC_OFFSET)
    current_hour = now.hour
    return MARKET_HOURS_START <= current_hour < MARKET_HOURS_END


def get_collection_frequency() -> int:
    """Get the appropriate collection frequency based on time of day"""
    return DEFAULT_COLLECTION_FREQUENCY if is_market_hours() else OFF_HOURS_COLLECTION_FREQUENCY


def is_in_cooldown(service_name: str) -> bool:
    """Check if a service is in cooldown period after errors"""
    if service_name not in api_cooldown:
        return False

    service = api_cooldown[service_name]
    if service['cooldown_until'] is None:
        return False

    return datetime.now() < service['cooldown_until']


def set_cooldown(service_name: str, minutes: int = 5) -> None:
    """Set a cooldown period for a service after encountering errors"""
    if service_name not in api_cooldown:
        api_cooldown[service_name] = {'last_error': None, 'cooldown_until': None}

    service = api_cooldown[service_name]
    service['last_error'] = datetime.now()
    service['cooldown_until'] = datetime.now() + timedelta(minutes=minutes)
    logger.warning(f"{service_name} in cooldown until {service['cooldown_until'].strftime('%H:%M:%S')}")


def reset_cooldown(service_name: str) -> None:
    """Reset cooldown for a service after successful operations"""
    if service_name in api_cooldown:
        api_cooldown[service_name]['cooldown_until'] = None


def collect_and_store_pool_data() -> None:
    """Collect pool data from APIs and store it in the database"""
    logger.info("Starting scheduled pool data collection")
    perf_monitor.start_operation('collect_pool_data')

    try:
        # Skip if data service is in cooldown
        if is_in_cooldown('data_service'):
            logger.info("Skipping pool data collection due to API cooldown")
            return

        # Get the data service
        data_service = get_data_service()
        if not data_service:
            logger.error("Failed to initialize data service")
            set_cooldown('data_service', 10)  # Longer cooldown for initialization failure
            return

        # Get all pools
        pools = data_service.get_all_pools()
        if not pools:
            logger.warning("No pools returned from data service")
            set_cooldown('data_service', 5)
            return

        logger.info(f"Retrieved {len(pools)} pools from data service")

        # Get token price service for additional price data
        token_price_service = get_token_price_service()

        # Store each pool snapshot in database
        success_count = 0
        error_count = 0

        with db_lock:  # Ensure thread safety
            for pool in pools:
                try:
                    # Enhance pool data with token prices if available
                    if token_price_service:
                        token1_symbol = pool.get('token1_symbol', '')
                        token2_symbol = pool.get('token2_symbol', '')

                        if token1_symbol:
                            price = token_price_service.get_token_price(token1_symbol)
                            if price is not None and price > 0:
                                pool['token1_price'] = price

                        if token2_symbol:
                            price = token_price_service.get_token_price(token2_symbol)
                            if price is not None and price > 0:
                                pool['token2_price'] = price

                    # Add timestamp if not present
                    if 'timestamp' not in pool:
                        pool['timestamp'] = datetime.now()

                    # Store in database
                    if store_pool_snapshot(pool):
                        success_count += 1
                    else:
                        error_count += 1
                        logger.warning(f"Failed to store pool {pool.get('id', 'unknown')}")

                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing pool: {str(e)}")
                    logger.debug(traceback.format_exc())

        # Log results
        logger.info(f"Pool data collection completed: {success_count} successful, {error_count} failed")

        # Reset cooldown if successful
        if success_count > 0:
            reset_cooldown('data_service')

    except Exception as e:
        logger.error(f"Error in pool data collection: {str(e)}")
        logger.debug(traceback.format_exc())
        set_cooldown('data_service', 5)
    finally:
        perf_monitor.end_operation('collect_pool_data')


def update_token_prices() -> None:
    """Update token prices in the database"""
    logger.info("Starting scheduled token price update")
    perf_monitor.start_operation('update_token_prices')

    try:
        # Skip if price service is in cooldown
        if is_in_cooldown('price_service'):
            logger.info("Skipping token price update due to API cooldown")
            return

        # Get the token price service
        token_price_service = get_token_price_service()
        if not token_price_service:
            logger.error("Failed to initialize token price service")
            set_cooldown('price_service', 10)
            return

        # Get list of tracked tokens
        # First try from the database - pools table
        db = get_db_manager()
        tokens = set()

        if db:
            try:
                # This query would need to be implemented in the DBManager class
                # to get unique token symbols from all pools
                token_list = db.get_unique_tokens()
                tokens.update(token_list)
            except Exception as e:
                logger.error(f"Error getting token list from database: {str(e)}")

        # If we couldn't get tokens from the database, use a fallback approach with data service
        if not tokens:
            data_service = get_data_service()
            if data_service:
                pools = data_service.get_all_pools()
                for pool in pools:
                    if pool.get('token1_symbol'):
                        tokens.add(pool.get('token1_symbol'))
                    if pool.get('token2_symbol'):
                        tokens.add(pool.get('token2_symbol'))

        # Clean up token list - remove empty or None values
        tokens = {t for t in tokens if t and isinstance(t, str)}

        if not tokens:
            logger.warning("No tokens found to update prices")
            return

        logger.info(f"Updating prices for {len(tokens)} tokens")

        # Update prices in batches to avoid API rate limits
        batch_size = 10
        token_list = list(tokens)
        
        for i in range(0, len(token_list), batch_size):
            batch = token_list[i:i+batch_size]
            try:
                # Batch update request
                token_price_service.update_token_prices(batch)
                
                # Small delay between batches to be nice to the API
                if i + batch_size < len(token_list):
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error updating token batch {i//batch_size + 1}: {str(e)}")

        # Save the price cache
        token_price_service.save_price_cache()
        logger.info("Token price update completed")

        # Reset cooldown if successful
        reset_cooldown('price_service')

    except Exception as e:
        logger.error(f"Error in token price update: {str(e)}")
        logger.debug(traceback.format_exc())
        set_cooldown('price_service', 10)
    finally:
        perf_monitor.end_operation('update_token_prices')


def generate_predictions() -> None:
    """Generate predictions for all pools"""
    logger.info("Starting scheduled prediction generation")
    perf_monitor.start_operation('generate_predictions')

    try:
        # Get the database manager
        db = get_db_manager()
        if not db:
            logger.error("Failed to initialize database manager")
            return

        # Get data service for latest pool data
        data_service = get_data_service()
        if not data_service:
            logger.error("Failed to initialize data service for predictions")
            return

        # Get all pools
        pools = data_service.get_all_pools()
        if not pools:
            logger.warning("No pools returned from data service for predictions")
            return

        logger.info(f"Generating predictions for {len(pools)} pools")

        # Import prediction models dynamically to avoid circular imports
        from eda.ml_models import get_prediction_model
        prediction_model = get_prediction_model()

        if not prediction_model:
            logger.error("Failed to load prediction model")
            return

        # Process each pool
        success_count = 0
        error_count = 0

        for pool in pools:
            try:
                pool_id = pool.get('id')
                if not pool_id:
                    logger.warning("Pool without ID, skipping prediction")
                    continue

                # Get historical data for this pool
                historical_data = db.get_pool_metrics(pool_id, days=30)
                
                # Skip if we don't have enough historical data
                if historical_data.empty or len(historical_data) < 5:
                    logger.debug(f"Not enough historical data for pool {pool_id}, skipping prediction")
                    continue

                # Generate prediction using the model
                prediction = prediction_model.predict_pool_performance(pool_id, historical_data, pool)
                
                if prediction:
                    # Format the prediction results for storage
                    predicted_apr = prediction.get('predicted_apr', 0)
                    risk_score = prediction.get('risk_score', 0.5)
                    performance_class = prediction.get('performance_class', 'medium')
                    model_version = prediction.get('model_version', '1.0')
                    
                    # Save prediction to database
                    if db.save_prediction(pool_id, predicted_apr, performance_class, risk_score, model_version):
                        success_count += 1
                    else:
                        error_count += 1
                        logger.warning(f"Failed to save prediction for pool {pool_id}")
                else:
                    logger.warning(f"Model returned no prediction for pool {pool_id}")
                    error_count += 1

            except Exception as e:
                error_count += 1
                logger.error(f"Error generating prediction for pool {pool.get('id', 'unknown')}: {str(e)}")

        # Log results
        logger.info(f"Prediction generation completed: {success_count} successful, {error_count} failed")

    except Exception as e:
        logger.error(f"Error in prediction generation: {str(e)}")
        logger.debug(traceback.format_exc())
    finally:
        perf_monitor.end_operation('generate_predictions')


def run_maintenance_tasks() -> None:
    """Run database maintenance tasks"""
    logger.info("Starting scheduled database maintenance")
    perf_monitor.start_operation('maintenance_tasks')

    try:
        # Get the database manager
        db = get_db_manager()
        if not db:
            logger.error("Failed to initialize database manager for maintenance")
            return

        # Vacuum the database to reclaim space and optimize performance
        if hasattr(db, 'vacuum_database') and callable(db.vacuum_database):
            success = db.vacuum_database()
            if success:
                logger.info("Database vacuum completed successfully")
            else:
                logger.warning("Database vacuum failed")

        # Clean up old data to prevent database bloat
        if hasattr(db, 'clean_old_metrics') and callable(db.clean_old_metrics):
            # Keep last 90 days of metrics (3 months)
            days_to_keep = 90
            deleted_count = db.clean_old_metrics(days_to_keep)
            logger.info(f"Cleaned up {deleted_count} old metric records")

        # Log database statistics
        if hasattr(db, 'get_database_stats') and callable(db.get_database_stats):
            stats = db.get_database_stats()
            logger.info(f"Database stats: {stats}")

    except Exception as e:
        logger.error(f"Error in database maintenance: {str(e)}")
        logger.debug(traceback.format_exc())
    finally:
        perf_monitor.end_operation('maintenance_tasks')


def export_data_snapshot() -> None:
    """Export a snapshot of data for backup or analysis"""
    logger.info("Starting scheduled data export")
    perf_monitor.start_operation('export_data')

    try:
        # Get the database manager
        db = get_db_manager()
        if not db:
            logger.error("Failed to initialize database manager for export")
            return

        # Create export directory if it doesn't exist
        export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'exports')
        os.makedirs(export_dir, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M')
        filename = f"{timestamp}_export.csv"
        filepath = os.path.join(export_dir, filename)

        # Export pool data
        if hasattr(db, 'export_pool_data') and callable(db.export_pool_data):
            success = db.export_pool_data(filepath)
            if success:
                logger.info(f"Data export completed successfully: {filepath}")
            else:
                logger.warning("Data export failed")

    except Exception as e:
        logger.error(f"Error in data export: {str(e)}")
        logger.debug(traceback.format_exc())
    finally:
        perf_monitor.end_operation('export_data')


def initialize_schedule() -> None:
    """Initialize the scheduler with tasks at specified intervals"""
    # Collect pool data at the default frequency
    freq = get_collection_frequency()
    schedule.every(freq).minutes.do(collect_and_store_pool_data)
    logger.info(f"Scheduled pool data collection every {freq} minutes")

    # Update token prices less frequently
    schedule.every(TOKEN_PRICE_FREQUENCY).minutes.do(update_token_prices)
    logger.info(f"Scheduled token price updates every {TOKEN_PRICE_FREQUENCY} minutes")

    # Generate predictions hourly
    schedule.every(PREDICTION_FREQUENCY).minutes.do(generate_predictions)
    logger.info(f"Scheduled prediction generation every {PREDICTION_FREQUENCY} minutes")

    # Database maintenance daily at 3 AM
    schedule.every().day.at("03:00").do(run_maintenance_tasks)
    logger.info("Scheduled database maintenance daily at 03:00")

    # Data export weekly
    schedule.every().sunday.at("04:00").do(export_data_snapshot)
    logger.info("Scheduled data export weekly on Sunday at 04:00")


def update_schedule() -> None:
    """Update schedule based on time of day"""
    # Adjust collection frequency based on market hours
    freq = get_collection_frequency()
    
    # Clear existing pool data collection jobs
    schedule.clear('collect_pool_data')
    
    # Re-schedule with new frequency
    schedule.every(freq).minutes.do(collect_and_store_pool_data).tag('collect_pool_data')
    logger.info(f"Updated pool data collection frequency to every {freq} minutes")


def schedule_loop() -> None:
    """Main loop that runs the scheduler"""
    while keep_running:
        try:
            # Run pending jobs
            schedule.run_pending()
            
            # Check if we need to update the schedule based on time of day
            current_hour = datetime.now().hour
            if current_hour == MARKET_HOURS_START or current_hour == MARKET_HOURS_END:
                update_schedule()
            
            # Sleep a bit
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in scheduler loop: {str(e)}")
            logger.debug(traceback.format_exc())
            time.sleep(60)  # Longer sleep on error


def signal_handler(signum, frame) -> None:
    """Signal handler for graceful shutdown"""
    global keep_running
    logger.info(f"Received signal {signum}, shutting down...")
    keep_running = False


def main() -> None:
    """Main entry point"""
    try:
        logger.info("Starting scheduled data collector")
        
        # Initialize the scheduler
        initialize_schedule()
        
        # Run an initial data collection immediately
        collect_and_store_pool_data()
        
        # Start update token prices
        update_token_prices()
        
        # Run the scheduler loop
        schedule_loop()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info("Scheduled data collector stopped")


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
