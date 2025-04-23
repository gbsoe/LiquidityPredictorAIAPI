import os
import time
import json
import logging
import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

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

# Node.js backend endpoints
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')
DATABASE_PATH = os.getenv('DATABASE_PATH', '../database/liquidity_pools.db')

class DataCollector:
    """
    Main data collection service for liquidity pool data from Raydium via Node.js backend
    """
    
    def __init__(self):
        """Initialize the data collector"""
        self.backend_url = BACKEND_URL
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.session = requests.Session()
        
        # Initialize connection to the database
        try:
            from database.db_operations import ensure_database_exists
            ensure_database_exists()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
    
    def fetch_with_retry(self, url, params=None):
        """Fetch data with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to fetch data from {url} after {self.max_retries} attempts")
                    raise
    
    def get_all_pools(self):
        """Fetch all liquidity pools from Raydium"""
        url = f"{self.backend_url}/api/pools"
        try:
            logger.info("Fetching all liquidity pools")
            pools = self.fetch_with_retry(url)
            logger.info(f"Successfully fetched {len(pools)} pools")
            return pools
        except Exception as e:
            logger.error(f"Error fetching all pools: {e}")
            return []
    
    def get_pool_details(self, pool_id):
        """Fetch detailed information about a specific pool"""
        url = f"{self.backend_url}/api/pools/{pool_id}"
        try:
            logger.info(f"Fetching details for pool {pool_id}")
            pool_data = self.fetch_with_retry(url)
            return pool_data
        except Exception as e:
            logger.error(f"Error fetching details for pool {pool_id}: {e}")
            return None
    
    def get_pool_metrics(self, pool_id):
        """Fetch metrics for a specific pool"""
        url = f"{self.backend_url}/api/pools/{pool_id}/metrics"
        try:
            logger.info(f"Fetching metrics for pool {pool_id}")
            metrics = self.fetch_with_retry(url)
            return metrics
        except Exception as e:
            logger.error(f"Error fetching metrics for pool {pool_id}: {e}")
            return None
    
    def get_blockchain_stats(self):
        """Fetch Solana blockchain statistics"""
        url = f"{self.backend_url}/api/blockchain/stats"
        try:
            logger.info("Fetching blockchain statistics")
            stats = self.fetch_with_retry(url)
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
            
            # Connect to database
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # Process each pool
            processed_count = 0
            for pool in pools_to_process:
                try:
                    pool_id = pool.get('ammId') or pool.get('id')
                    if not pool_id:
                        logger.warning(f"Skipping pool with no ID: {pool}")
                        continue
                    
                    # Get detailed metrics
                    metrics = self.get_pool_metrics(pool_id)
                    if not metrics:
                        logger.warning(f"No metrics available for pool {pool_id}")
                        continue
                    
                    # Extract and transform data
                    pool_data = {
                        'pool_id': pool_id,
                        'name': metrics.get('name', 'Unknown'),
                        'liquidity': metrics.get('liquidity', 0),
                        'volume_24h': metrics.get('volume24h', 0),
                        'apr': metrics.get('apr', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Store basic pool data
                    cursor.execute('''
                        INSERT OR REPLACE INTO pool_data 
                        (pool_id, name, liquidity, volume_24h, apr, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        pool_data['pool_id'],
                        pool_data['name'],
                        pool_data['liquidity'],
                        pool_data['volume_24h'],
                        pool_data['apr'],
                        pool_data['timestamp']
                    ))
                    
                    # Store metrics with hourly granularity
                    cursor.execute('''
                        INSERT INTO pool_metrics
                        (pool_id, liquidity, volume, apr, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        pool_id,
                        metrics.get('liquidity', 0),
                        metrics.get('volume24h', 0),
                        metrics.get('apr', 0),
                        datetime.now().isoformat()
                    ))
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"Processed {processed_count}/{len(pools_to_process)} pools")
                        conn.commit()  # Intermediate commit
                    
                except Exception as e:
                    logger.error(f"Error processing pool {pool.get('ammId', 'unknown')}: {e}")
            
            # Final commit
            conn.commit()
            conn.close()
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
                conn = sqlite3.connect(DATABASE_PATH)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO blockchain_stats
                    (slot, block_height, avg_tps, sol_price, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    blockchain_stats.get('slot', 0),
                    blockchain_stats.get('blockHeight', 0),
                    blockchain_stats.get('averageTps', 0),
                    blockchain_stats.get('solPrice', 0),
                    blockchain_stats.get('currentTimestamp', datetime.now().isoformat())
                ))
                conn.commit()
                conn.close()
            
            # Collect and store pool data
            self.collect_and_store_pool_data()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Data collection cycle completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during collection cycle: {e}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run_collection_cycle()
