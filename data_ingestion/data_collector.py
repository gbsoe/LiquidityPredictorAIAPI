import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Import our new Raydium API client
from .raydium_api_client import RaydiumAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_collector.log')
    ]
)

logger = logging.getLogger('data_collector')

# Database connection
from database.db_operations import DBManager

class DataCollector:
    """
    Main data collection service for liquidity pool data from Raydium API
    """
    
    def __init__(self):
        """Initialize the data collector"""
        # Initialize the Raydium API client
        self.api_client = RaydiumAPIClient()
        
        # Initialize connection to the database
        try:
            from database.db_operations import ensure_database_exists
            ensure_database_exists()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
    
    def get_all_pools(self):
        """Fetch all liquidity pools from Raydium"""
        try:
            logger.info("Fetching all liquidity pools")
            pools = self.api_client.get_all_pools()
            logger.info(f"Successfully fetched {len(pools)} pools")
            return pools
        except Exception as e:
            logger.error(f"Error fetching all pools: {e}")
            return []
    
    def get_pool_details(self, pool_id):
        """Fetch detailed information about a specific pool"""
        try:
            logger.info(f"Fetching details for pool {pool_id}")
            pool_data = self.api_client.get_pool_by_id(pool_id)
            return pool_data
        except Exception as e:
            logger.error(f"Error fetching details for pool {pool_id}: {e}")
            return None
    
    def get_pool_metrics(self, pool_id):
        """Fetch metrics for a specific pool"""
        try:
            logger.info(f"Fetching metrics for pool {pool_id}")
            metrics = self.api_client.get_pool_metrics(pool_id)
            return metrics
        except Exception as e:
            logger.error(f"Error fetching metrics for pool {pool_id}: {e}")
            return None
    
    def get_blockchain_stats(self):
        """Fetch Solana blockchain statistics"""
        try:
            logger.info("Fetching blockchain statistics")
            stats = self.api_client.get_blockchain_stats()
            return stats
        except Exception as e:
            logger.error(f"Error fetching blockchain stats: {e}")
            return None
    
    def collect_and_store_pool_data(self):
        """Collect data for all pools and store in database"""
        try:
            # Get all pools
            pools = self.get_all_pools()
            if not pools:
                logger.warning("No pools found or error fetching pools")
                return
            
            # Sample a subset of pools for testing/development
            # In production, process all pools or implement prioritization
            sample_size = min(100, len(pools))
            pools_to_process = pools[:sample_size]
            
            logger.info(f"Processing {len(pools_to_process)} pools")
            
            # Initialize DB Manager
            db = DBManager()
            
            # Process each pool
            processed_count = 0
            for pool in pools_to_process:
                try:
                    pool_id = pool.get('id') or pool.get('ammId')
                    if not pool_id:
                        logger.warning(f"Skipping pool with no ID: {pool}")
                        continue
                    
                    # Get detailed pool information if needed
                    # For efficiency, we can use the metrics from the pool if they exist
                    pool_metrics = None
                    if 'apr' in pool and 'liquidity' in pool and 'volume24h' in pool:
                        # Use metrics from the pool object if available
                        pool_metrics = pool
                    else:
                        # Otherwise fetch detailed metrics
                        pool_metrics = self.get_pool_metrics(pool_id)
                    
                    if not pool_metrics:
                        logger.warning(f"No metrics available for pool {pool_id}")
                        continue
                    
                    # Extract and transform data
                    pool_name = pool.get('name') or f"{pool.get('token1Symbol', 'Unknown')}/{pool.get('token2Symbol', 'Unknown')}"
                    
                    # Store basic pool data
                    db.save_pool_data(
                        pool_id=pool_id,
                        name=pool_name,
                        liquidity=pool_metrics.get('liquidity', 0),
                        volume_24h=pool_metrics.get('volume24h', 0),
                        apr=pool_metrics.get('apr', 0),
                        timestamp=datetime.now()
                    )
                    
                    # Store metrics with hourly granularity
                    db.save_pool_metrics(
                        pool_id=pool_id,
                        liquidity=pool_metrics.get('liquidity', 0),
                        volume=pool_metrics.get('volume24h', 0),
                        apr=pool_metrics.get('apr', 0),
                        timestamp=datetime.now()
                    )
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"Processed {processed_count}/{len(pools_to_process)} pools")
                    
                except Exception as e:
                    logger.error(f"Error processing pool {pool.get('id', 'unknown')}: {e}")
            
            logger.info(f"Successfully processed {processed_count} pools")
            
        except Exception as e:
            logger.error(f"Error in collect_and_store_pool_data: {e}")
    
    def run_collection_cycle(self):
        """Run a complete data collection cycle"""
        logger.info("Starting data collection cycle")
        start_time = time.time()
        
        try:
            # Collect blockchain stats
            blockchain_stats = self.get_blockchain_stats()
            if blockchain_stats:
                logger.info("Successfully collected blockchain stats")
                # Store blockchain stats in database
                db = DBManager()
                db.save_blockchain_stats(
                    slot=blockchain_stats.get('slot', 0),
                    block_height=blockchain_stats.get('blockHeight', 0),
                    avg_tps=blockchain_stats.get('tps', 0),
                    sol_price=blockchain_stats.get('solPrice', 0),
                    timestamp=datetime.now()
                )
            
            # Collect and store pool data
            self.collect_and_store_pool_data()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Data collection cycle completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during collection cycle: {e}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run_collection_cycle()
