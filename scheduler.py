#!/usr/bin/env python3
import os
import sys
import time
import logging
import schedule
import subprocess
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scheduler.log')
    ]
)

logger = logging.getLogger('scheduler')

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['DATABASE_PATH'] = os.path.join(BASE_DIR, 'database/liquidity_pools.db')

# Tasks
def run_data_collection():
    """Run data collection tasks."""
    logger.info("Running data collection...")
    try:
        # Run data collector
        subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, 'data_ingestion/data_collector.py')],
            check=True
        )
        logger.info("Data collector completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data collector failed with code {e.returncode}")
    except Exception as e:
        logger.error(f"Error running data collector: {e}")

def run_price_tracking():
    """Run price tracking tasks."""
    logger.info("Running price tracker...")
    try:
        # Run price tracker
        subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, 'data_ingestion/price_tracker.py')],
            check=True
        )
        logger.info("Price tracker completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Price tracker failed with code {e.returncode}")
    except Exception as e:
        logger.error(f"Error running price tracker: {e}")

def run_data_validation():
    """Run data validation tasks."""
    logger.info("Running data validation...")
    try:
        # Run data validator
        subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, 'data_ingestion/data_validator.py')],
            check=True
        )
        logger.info("Data validator completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data validator failed with code {e.returncode}")
    except Exception as e:
        logger.error(f"Error running data validator: {e}")

def run_ml_training():
    """Run ML model training."""
    logger.info("Running ML model training...")
    try:
        # Run ML models
        subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, 'eda/ml_models.py')],
            check=True
        )
        logger.info("ML model training completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"ML model training failed with code {e.returncode}")
    except Exception as e:
        logger.error(f"Error running ML model training: {e}")

# Run all tasks
def run_all_tasks():
    """Run all scheduled tasks in sequence."""
    logger.info("Running all scheduled tasks...")
    run_data_collection()
    run_price_tracking()
    run_data_validation()
    run_ml_training()
    logger.info("All scheduled tasks completed")

# Manual run
def manual_run():
    """Run tasks manually once."""
    logger.info("Starting manual run...")
    run_all_tasks()

# Schedule setup
def setup_schedule():
    """Set up the task schedule."""
    # Data collection every hour
    schedule.every(1).hours.do(run_data_collection)
    
    # Price tracking every 30 minutes
    schedule.every(30).minutes.do(run_price_tracking)
    
    # Data validation once a day
    schedule.every().day.at("00:30").do(run_data_validation)
    
    # ML training once a day
    schedule.every().day.at("01:00").do(run_ml_training)
    
    # Full refresh twice a day
    schedule.every().day.at("06:00").do(run_all_tasks)
    schedule.every().day.at("18:00").do(run_all_tasks)
    
    logger.info("Schedule set up successfully")

# Main function
def main():
    """Main scheduler function."""
    try:
        # Parse command-line arguments
        if len(sys.argv) > 1 and sys.argv[1] == "--manual":
            # Run tasks manually and exit
            manual_run()
            return
        
        # Set up schedule
        setup_schedule()
        
        # Run all tasks once at startup
        run_all_tasks()
        
        # Keep the scheduler running
        logger.info("Scheduler running. Press Ctrl+C to stop.")
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Error in scheduler: {e}")

if __name__ == "__main__":
    main()
