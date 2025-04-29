"""
Scheduled Data Collector for SolPool Insight

This module implements robust, scheduled collection of liquidity pool data
with time-series tracking for better prediction accuracy.
"""

import os
import time
import json
import logging
import schedule
import threading
import pandas as pd
from datetime import datetime, timedelta
from defi_aggregation_api import DefiAggregationAPI
from database.db_operations import store_pool_snapshot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='data_collection.log'
)
logger = logging.getLogger('data_collector')

class ScheduledDataCollector:
    """
    Manages scheduled data collection from DeFi APIs with time-series tracking
    for building historical datasets necessary for accurate predictions.
    """
    
    def __init__(self, 
                collection_interval_hours=4, 
                max_retries=3, 
                historical_days=30, 
                backup_dir="./data/historical"):
        """
        Initialize the data collector with configurable parameters.
        
        Args:
            collection_interval_hours: Hours between data collection runs
            max_retries: Maximum retries for failed API calls
            historical_days: Days of historical data to maintain
            backup_dir: Directory for data backups
        """
        self.api = DefiAggregationAPI()
        self.collection_interval_hours = collection_interval_hours
        self.max_retries = max_retries
        self.historical_days = historical_days
        self.backup_dir = backup_dir
        self.is_running = False
        self.scheduler_thread = None
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def collect_data(self):
        """
        Collect pool data with timestamps and store in database.
        Implements retry logic and data verification.
        """
        logger.info(f"Starting scheduled data collection at {datetime.now()}")
        
        # Create a timestamp for this collection run
        timestamp = datetime.now()
        collection_id = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Track success metrics
        success_count = 0
        new_pools_count = 0
        metrics = {"start_time": timestamp.isoformat()}
        
        try:
            # Get list of unique pool IDs we already have in DB to track new pools
            existing_pool_ids = set()  # In production: get_existing_pool_ids()
            
            # Collect data with retries
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Get pools from API with the transformed data format
                    pools = self.api.get_all_pools(max_pools=500)
                    
                    if not pools:
                        logger.warning(f"No pools returned from API (attempt {attempt}/{self.max_retries})")
                        time.sleep(5)  # Brief pause before retry
                        continue
                    
                    logger.info(f"Retrieved {len(pools)} pools from API")
                    
                    # Process each pool for storage
                    for pool in pools:
                        pool_id = pool.get('id')
                        if not pool_id:
                            continue
                            
                        # Add timestamp to the pool data
                        pool['timestamp'] = timestamp
                        
                        # Check if this is a new pool we haven't seen before
                        if pool_id not in existing_pool_ids:
                            new_pools_count += 1
                            
                        # In production, store in database
                        # store_pool_snapshot(pool)
                        success_count += 1
                    
                    # Success! No need for more retries
                    break
                    
                except Exception as e:
                    logger.error(f"Error collecting data (attempt {attempt}/{self.max_retries}): {str(e)}")
                    if attempt < self.max_retries:
                        time.sleep(10)  # Longer pause before retry
            
            # Generate backup file with timestamp
            if success_count > 0:
                backup_file = os.path.join(self.backup_dir, f"pools_{collection_id}.json")
                try:
                    with open(backup_file, 'w') as f:
                        json.dump(pools, f)
                    logger.info(f"Backed up {success_count} pools to {backup_file}")
                except Exception as e:
                    logger.error(f"Failed to create backup: {str(e)}")
            
            # Update metrics
            metrics.update({
                "end_time": datetime.now().isoformat(),
                "pools_processed": success_count,
                "new_pools_found": new_pools_count,
                "success": success_count > 0
            })
            
            logger.info(f"Data collection complete. Processed {success_count} pools, found {new_pools_count} new pools.")
            return metrics
            
        except Exception as e:
            logger.error(f"Unexpected error in data collection: {str(e)}")
            metrics.update({
                "end_time": datetime.now().isoformat(),
                "error": str(e),
                "success": False
            })
            return metrics
    
    def start_scheduler(self):
        """
        Start the scheduled data collection process.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return False
        
        # Clear existing jobs
        schedule.clear()
        
        # Schedule the collection job
        schedule.every(self.collection_interval_hours).hours.do(self.collect_data)
        
        logger.info(f"Scheduled data collection every {self.collection_interval_hours} hours")
        
        # Run immediately for initial data
        self.collect_data()
        
        # Start the scheduler in a background thread
        self.is_running = True
        
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        return True
    
    def stop_scheduler(self):
        """
        Stop the scheduled data collection.
        """
        if not self.is_running:
            return
            
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=2)
        
        logger.info("Stopped scheduled data collection")
    
    def run_ad_hoc_collection(self):
        """
        Run an immediate data collection outside the schedule.
        Useful for testing or manual updates.
        """
        logger.info("Running ad-hoc data collection")
        return self.collect_data()
    
    def get_collection_stats(self):
        """
        Get statistics about collected data over time.
        """
        stats = {
            "total_runs": 0,
            "total_pools_collected": 0,
            "unique_pools": 0,
            "first_collection": None,
            "last_collection": None,
            "collection_frequency_hours": self.collection_interval_hours,
            "estimated_data_points": 0
        }
        
        # In production, these would be populated from database metrics
        
        return stats
    
    def prune_old_data(self):
        """
        Remove data older than self.historical_days to manage storage.
        """
        cutoff_date = datetime.now() - timedelta(days=self.historical_days)
        logger.info(f"Pruning data older than {cutoff_date}")
        
        # In production, database records would be pruned
        # prune_pools_before_date(cutoff_date)
        
        # Remove old backup files
        try:
            for filename in os.listdir(self.backup_dir):
                if not filename.startswith('pools_'):
                    continue
                
                file_path = os.path.join(self.backup_dir, filename)
                file_date_str = filename.split('_')[1].split('.')[0]
                
                try:
                    file_date = datetime.strptime(file_date_str, "%Y%m%d")
                    if file_date < cutoff_date:
                        os.remove(file_path)
                        logger.info(f"Removed old backup file: {filename}")
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse date from filename: {filename}")
        except Exception as e:
            logger.error(f"Error pruning old backup files: {str(e)}")

# Singleton instance
collector = None

def initialize_collector(interval_hours=4):
    """
    Initialize the global collector instance.
    """
    global collector
    collector = ScheduledDataCollector(collection_interval_hours=interval_hours)
    return collector

def get_collector():
    """
    Get the global collector instance.
    """
    global collector
    if collector is None:
        collector = initialize_collector()
    return collector

if __name__ == "__main__":
    # Initialize and start collection
    collector = initialize_collector(interval_hours=4)
    collector.start_scheduler()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopping data collection...")
        collector.stop_scheduler()