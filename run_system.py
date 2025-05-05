#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess
import threading
import signal
import atexit
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system.log')
    ]
)

logger = logging.getLogger('system')

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['DATABASE_PATH'] = os.path.join(BASE_DIR, 'database/liquidity_pools.db')

# Initialize database
def initialize_database():
    """Create the database and tables if they don't exist."""
    try:
        from database.schema import initialize_database
        success = initialize_database()
        if success:
            logger.info("Database initialized successfully")
        else:
            logger.error("Failed to initialize database")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        sys.exit(1)

# Start Node.js backend
def start_backend():
    """Start the Node.js backend server."""
    try:
        # Change to the backend directory
        os.chdir(os.path.join(BASE_DIR, 'node_backend'))
        
        # Start the Node.js backend with stdout and stderr redirected to a file
        backend_log = open(os.path.join(BASE_DIR, 'node_backend.log'), 'a')
        process = subprocess.Popen(
            ['node', 'server.js'],
            stdout=backend_log,
            stderr=backend_log,
            preexec_fn=os.setsid  # Use a process group for clean termination
        )
        
        logger.info(f"Started Node.js backend (PID: {process.pid})")
        return process
    except Exception as e:
        logger.error(f"Error starting Node.js backend: {e}")
        return None

# Start data collection threads
def start_data_collection():
    """Start the data collection services."""
    try:
        # Change back to the base directory
        os.chdir(BASE_DIR)
        
        # Start the data collector
        collector_log = open(os.path.join(BASE_DIR, 'data_collector.log'), 'a')
        collector_process = subprocess.Popen(
            [sys.executable, 'data_ingestion/data_collector.py'],
            stdout=collector_log,
            stderr=collector_log
        )
        
        logger.info(f"Started data collector (PID: {collector_process.pid})")
        
        # Start the price tracker
        price_log = open(os.path.join(BASE_DIR, 'price_tracker.log'), 'a')
        price_process = subprocess.Popen(
            [sys.executable, 'data_ingestion/price_tracker.py'],
            stdout=price_log,
            stderr=price_log
        )
        
        logger.info(f"Started price tracker (PID: {price_process.pid})")
        
        # Start the data validator (runs once at startup)
        validator_log = open(os.path.join(BASE_DIR, 'data_validator.log'), 'a')
        validator_process = subprocess.Popen(
            [sys.executable, 'data_ingestion/data_validator.py'],
            stdout=validator_log,
            stderr=validator_log
        )
        
        logger.info(f"Started data validator (PID: {validator_process.pid})")
        
        return [collector_process, price_process, validator_process]
    except Exception as e:
        logger.error(f"Error starting data collection: {e}")
        return []

# Start ML models
def start_ml_models():
    """Start the ML model training and prediction."""
    try:
        # Change to the base directory
        os.chdir(BASE_DIR)
        
        # Start the ML models
        ml_log = open(os.path.join(BASE_DIR, 'ml_models.log'), 'a')
        ml_process = subprocess.Popen(
            [sys.executable, 'eda/ml_models.py'],
            stdout=ml_log,
            stderr=ml_log
        )
        
        logger.info(f"Started ML models (PID: {ml_process.pid})")
        return ml_process
    except Exception as e:
        logger.error(f"Error starting ML models: {e}")
        return None

# Start API server
def start_api_server():
    """Start the API server."""
    try:
        # Change to the base directory
        os.chdir(BASE_DIR)
        
        # Start the API server
        api_log = open(os.path.join(BASE_DIR, 'api_server.log'), 'a')
        api_process = subprocess.Popen(
            [sys.executable, 'api_endpoints.py'],
            stdout=api_log,
            stderr=api_log
        )
        
        logger.info(f"Started API server (PID: {api_process.pid})")
        return api_process
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        return None

# Start Streamlit dashboard
def start_dashboard():
    """Start the Streamlit dashboard."""
    try:
        # Change to the base directory
        os.chdir(BASE_DIR)
        
        # Start the Streamlit app
        dashboard_log = open(os.path.join(BASE_DIR, 'dashboard.log'), 'a')
        dashboard_process = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', 'solpool_insight.py', '--server.port', '5000', '--server.address', '0.0.0.0'],
            stdout=dashboard_log,
            stderr=dashboard_log
        )
        
        logger.info(f"Started Streamlit dashboard (PID: {dashboard_process.pid})")
        return dashboard_process
    except Exception as e:
        logger.error(f"Error starting Streamlit dashboard: {e}")
        return None

# Graceful shutdown function
def shutdown(processes):
    """Shut down all running processes gracefully."""
    logger.info("Shutting down...")
    
    for process in processes:
        if process and process.poll() is None:  # If process is still running
            try:
                # Try graceful termination
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if not terminated in time
                try:
                    process.kill()
                except:
                    pass
    
    logger.info("Shutdown complete")

# Main function
def main():
    """Start all system components."""
    processes = []
    
    try:
        # Initialize database
        initialize_database()
        
        # Start backend
        backend_process = start_backend()
        if backend_process:
            processes.append(backend_process)
        
        # Give backend time to start
        time.sleep(2)
        
        # Start data collection
        data_processes = start_data_collection()
        processes.extend(data_processes)
        
        # Start ML models
        ml_process = start_ml_models()
        if ml_process:
            processes.append(ml_process)
        
        # Start API server
        api_process = start_api_server()
        if api_process:
            processes.append(api_process)
        
        # Start dashboard
        dashboard_process = start_dashboard()
        if dashboard_process:
            processes.append(dashboard_process)
        
        # Register cleanup function
        atexit.register(lambda: shutdown(processes))
        
        # Set up signal handlers
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            shutdown(processes)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Main loop - keep the script running
        logger.info("System started successfully. Press Ctrl+C to stop.")
        while True:
            # Check if any process has terminated unexpectedly
            for i, process in enumerate(processes):
                if process and process.poll() is not None:
                    logger.warning(f"Process {process.pid} has terminated with code {process.returncode}")
                    
                    # Restart if it was a critical process
                    if i == 0:  # Backend
                        logger.info("Restarting backend...")
                        new_process = start_backend()
                        if new_process:
                            processes[i] = new_process
                    elif i == len(processes) - 1:  # Dashboard
                        logger.info("Restarting dashboard...")
                        new_process = start_dashboard()
                        if new_process:
                            processes[i] = new_process
            
            time.sleep(10)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        shutdown(processes)

if __name__ == "__main__":
    main()
