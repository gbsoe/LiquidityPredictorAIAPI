#!/usr/bin/env python3

"""
Initialize the Solana Liquidity Pool Analysis System
This script sets up the database and collects initial data
"""

import os
import sys
import time
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('system_init')

# Load environment variables from .env file
load_dotenv()

# Import project modules
from database.schema import initialize_database
from database.db_operations import DBManager
from data_ingestion.raydium_api_client import RaydiumAPIClient
from data_ingestion.data_collector import DataCollector

def setup_database():
    """Initialize the database with the required schema."""
    logger.info("Setting up database...")
    
    if not os.environ.get('DATABASE_URL'):
        logger.error("DATABASE_URL environment variable is not set")
        return False
    
    success = initialize_database()
    if success:
        logger.info("Database setup complete")
    else:
        logger.error("Database setup failed")
    
    return success

def collect_initial_data():
    """Collect initial data from Raydium API."""
    logger.info("Collecting initial data...")
    
    try:
        # Initialize data collector
        collector = DataCollector()
        
        # Run a collection cycle
        collector.run_collection_cycle()
        
        logger.info("Initial data collection complete")
        return True
    except Exception as e:
        logger.error(f"Error collecting initial data: {e}")
        return False

def check_api_connection():
    """Check that the Raydium API is accessible."""
    logger.info("Checking Raydium API connection...")
    
    try:
        api_client = RaydiumAPIClient()
        
        # Try to get blockchain stats as a simple test
        stats = api_client.get_blockchain_stats()
        
        if stats:
            logger.info("Raydium API connection successful")
            return True
        else:
            logger.warning("Raydium API returned no data, but connection was established")
            return True
    except Exception as e:
        logger.error(f"Failed to connect to Raydium API: {e}")
        return False

def main():
    """Run the initialization process."""
    logger.info("Starting system initialization")
    
    # Check API connection
    if not check_api_connection():
        logger.error("Cannot proceed without API connection")
        return False
    
    # Set up database
    if not setup_database():
        logger.error("Cannot proceed without database")
        return False
    
    # Collect initial data
    if not collect_initial_data():
        logger.warning("Initial data collection failed, but continuing")
    
    logger.info("System initialization complete")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)