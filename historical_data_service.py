import logging
import os
import json
import time
from datetime import datetime, timedelta
import threading
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Singleton pattern for historical data service
_service_instance = None

class HistoricalDataService:
    """Service for storing and retrieving historical pool data"""
    
    def __init__(self):
        """Initialize the historical data service"""
        self.data_dir = "data/historical"
        self.pools_cache = {}
        self.metrics_cache = {}
        self.last_updated = {}
        self.initialized = False
        self.collection_start_time = datetime.now()
        
        # Create the data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load any existing data from disk
        self._load_data()
        
        # Start the background thread for data collection
        self.collection_thread = None
        self.stop_collection = False
        self.initialized = True
        logger.info("Historical data service initialized")
    
    def _load_data(self):
        """Load historical data from disk"""
        try:
            # Look for pool metrics files in the data directory
            metrics_files = [f for f in os.listdir(self.data_dir) if f.endswith('_metrics.json')]
            
            for file_name in metrics_files:
                pool_id = file_name.split('_metrics.json')[0]
                file_path = os.path.join(self.data_dir, file_name)
                
                try:
                    with open(file_path, 'r') as f:
                        metrics_data = json.load(f)
                    
                    # Convert to DataFrame for easier processing
                    if metrics_data and len(metrics_data) > 0:
                        # Convert string dates to datetime
                        for entry in metrics_data:
                            entry['timestamp'] = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                        
                        self.metrics_cache[pool_id] = pd.DataFrame(metrics_data)
                        logger.debug(f"Loaded historical metrics for pool {pool_id}")
                except Exception as e:
                    logger.error(f"Error loading metrics for pool {pool_id}: {str(e)}")
            
            logger.info(f"Loaded historical data for {len(self.metrics_cache)} pools")
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
    
    def save_pool_metrics(self, pool_id: str, metrics: Dict[str, Any]):
        """Save a single data point of pool metrics"""
        if not self.initialized:
            logger.warning("Historical data service not initialized yet")
            return
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now().isoformat()
            
            # Ensure timestamp is a string for consistent handling
            if isinstance(metrics['timestamp'], datetime):
                metrics['timestamp'] = metrics['timestamp'].isoformat()
            elif not isinstance(metrics['timestamp'], str):
                metrics['timestamp'] = str(metrics['timestamp'])
                
            # Convert to DataFrame format for consistency
            metrics_df = pd.DataFrame([metrics])
            
            # Handle timestamp format
            # We'll only convert to datetime when actually needed for filtering or display
            # Avoid conversion at this stage to prevent dt accessor issues
            
            # Update in-memory cache
            if pool_id in self.metrics_cache:
                self.metrics_cache[pool_id] = pd.concat([self.metrics_cache[pool_id], metrics_df])
            else:
                self.metrics_cache[pool_id] = metrics_df
            
            # Update last_updated timestamp
            self.last_updated[pool_id] = datetime.now()
            
            # Save to disk (could optimize to batch writes)
            self._save_pool_metrics(pool_id)
            
            logger.debug(f"Saved metrics for pool {pool_id}")
        except Exception as e:
            logger.error(f"Error saving metrics for pool {pool_id}: {str(e)}")
    
    def _save_pool_metrics(self, pool_id: str):
        """Save pool metrics to disk"""
        try:
            if pool_id in self.metrics_cache:
                # Convert to list of dictionaries for JSON serialization
                metrics_data = self.metrics_cache[pool_id].copy()
                
                # Convert timestamp to datetime if needed, then to string
                try:
                    # Ensure all timestamps are timezone-naive before string conversion
                    if pd.api.types.is_datetime64_dtype(metrics_data['timestamp']):
                        # Convert to timezone-naive by using .dt.tz_localize(None)
                        if hasattr(metrics_data['timestamp'].dt, 'tz_localize'):
                            metrics_data['timestamp'] = metrics_data['timestamp'].dt.tz_localize(None)
                        metrics_data['timestamp'] = metrics_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                    else:
                        # Convert string timestamps to datetime objects with consistent timezone handling
                        metrics_data['timestamp'] = pd.to_datetime(metrics_data['timestamp'], utc=True).dt.tz_localize(None)
                        metrics_data['timestamp'] = metrics_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                except Exception as e:
                    # If conversion fails, use a simpler approach
                    logger.warning(f"Timestamp conversion error: {e}, using direct string conversion instead")
                    # Create a timestamp_str column instead of modifying the original
                    metrics_data['timestamp_str'] = metrics_data['timestamp'].astype(str)
                    metrics_data = metrics_data.rename(columns={'timestamp_str': 'timestamp'})
                
                metrics_list = metrics_data.to_dict(orient='records')
                
                # Save to disk
                file_path = os.path.join(self.data_dir, f"{pool_id}_metrics.json")
                with open(file_path, 'w') as f:
                    json.dump(metrics_list, f)
        except Exception as e:
            logger.error(f"Error saving metrics to disk for pool {pool_id}: {str(e)}")
    
    def get_pool_metrics(self, pool_id: str, days: int = 7) -> pd.DataFrame:
        """Get historical metrics for a specific pool"""
        if not self.initialized:
            logger.warning("Historical data service not initialized yet")
            return pd.DataFrame()
        
        try:
            if pool_id in self.metrics_cache:
                # Filter for the requested time period
                cutoff_date = datetime.now() - timedelta(days=days)
                metrics_df = self.metrics_cache[pool_id]
                
                # Make sure timestamp is datetime format with consistent timezone handling
                if not pd.api.types.is_datetime64_dtype(metrics_df['timestamp']):
                    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'], utc=True)
                
                # Ensure all timestamps are timezone-naive for comparison
                if hasattr(metrics_df['timestamp'].dt, 'tz_localize') and metrics_df['timestamp'].dt.tz is not None:
                    metrics_df['timestamp'] = metrics_df['timestamp'].dt.tz_localize(None)
                    
                # Make cutoff_date timezone-naive for consistent comparison
                cutoff_date = pd.Timestamp(cutoff_date).tz_localize(None)
                
                # Filter by date
                recent_metrics = metrics_df[metrics_df['timestamp'] >= cutoff_date]
                
                # Sort by timestamp
                return recent_metrics.sort_values('timestamp')
            else:
                logger.debug(f"No metrics found for pool {pool_id}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting metrics for pool {pool_id}: {str(e)}")
            return pd.DataFrame()
    
    # Alias for get_pool_metrics to maintain compatibility with existing code
    def get_pool_history(self, pool_id: str, days: int = 7) -> list:
        """Alias for get_pool_metrics that returns a list of dictionaries"""
        metrics_df = self.get_pool_metrics(pool_id, days)
        if metrics_df.empty:
            return []
        return metrics_df.to_dict(orient='records')
    
    def collect_pool_data(self, pool_data: List[Dict[str, Any]]):
        """Store the current state of multiple pools for historical tracking"""
        if not pool_data or len(pool_data) == 0:
            logger.warning("No pool data provided for collection")
            return
        
        try:
            timestamp = datetime.now().isoformat()
            for pool in pool_data:
                # Skip pools without valid IDs
                pool_id = pool.get('id', '') or pool.get('poolId', '') or pool.get('pool_id', '')
                if not pool_id:
                    continue
                
                # Extract key metrics
                metrics = {
                    'timestamp': timestamp,
                    'liquidity': pool.get('liquidity', 0) or pool.get('tvl', 0) or pool.get('liquidityUsd', 0) or 0,
                    'volume': pool.get('volume_24h', 0) or pool.get('volume24h', 0) or pool.get('volume', 0) or 0,
                    'apr': pool.get('apr', 0) or pool.get('apr24h', 0) or pool.get('apy', 0) or 0
                }
                
                # Save the metrics for this pool
                self.save_pool_metrics(pool_id, metrics)
            
            logger.info(f"Collected historical data for {len(pool_data)} pools")
        except Exception as e:
            logger.error(f"Error collecting pool data: {str(e)}")
    
    def start_data_collection(self, data_service, interval_minutes: int = 60):
        """Start a background thread to collect data at regular intervals"""
        if self.collection_thread is not None and self.collection_thread.is_alive():
            logger.info("Data collection already running")
            return
        
        def collection_worker():
            logger.info(f"Starting historical data collection every {interval_minutes} minutes")
            
            while not self.stop_collection:
                try:
                    # Get all pools from the data service
                    pools = data_service.get_all_pools()
                    
                    if pools and len(pools) > 0:
                        # Collect data for all pools
                        self.collect_pool_data(pools)
                    else:
                        logger.warning("No pools returned from data service for historical collection")
                except Exception as e:
                    logger.error(f"Error in historical data collection: {str(e)}")
                
                # Sleep for the interval (but check stop flag more frequently)
                for _ in range(interval_minutes * 60 // 10):  # Check every 10 seconds
                    if self.stop_collection:
                        break
                    time.sleep(10)
            
            logger.info("Historical data collection stopped")
        
        # Start the collection thread
        self.stop_collection = False
        self.collection_thread = threading.Thread(target=collection_worker, daemon=True)
        self.collection_thread.start()
    
    def stop_data_collection(self):
        """Stop the background data collection thread"""
        if self.collection_thread is not None and self.collection_thread.is_alive():
            logger.info("Stopping historical data collection")
            self.stop_collection = True
            self.collection_thread.join(timeout=30)  # Wait up to 30 seconds
            logger.info("Historical data collection stopped")
        else:
            logger.info("No data collection running")
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get the status of historical data collection"""
        status = {
            "initialized": self.initialized,
            "collection_active": self.collection_thread is not None and self.collection_thread.is_alive(),
            "pools_tracked": len(self.metrics_cache),
            "collection_start_time": self.collection_start_time.isoformat(),
            "collection_duration_hours": (datetime.now() - self.collection_start_time).total_seconds() / 3600
        }
        return status

# Singleton accessor function
def get_historical_service() -> HistoricalDataService:
    global _service_instance
    
    if _service_instance is None:
        _service_instance = HistoricalDataService()
    
    return _service_instance

# Function to start collection from outside this module
def start_historical_collection(data_service, interval_minutes: int = 60):
    service = get_historical_service()
    service.start_data_collection(data_service, interval_minutes)
    return service
