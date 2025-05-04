"""
Check the prediction data in the database
"""

import os
import sys
import logging
import pandas as pd
import psycopg2

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.db_operations import DBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_connection_params():
    """Get database connection parameters from environment variables"""
    db_url = os.environ.get("DATABASE_URL")
    
    if db_url:
        # Parse DATABASE_URL
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        
        try:
            from urllib.parse import urlparse
            
            parsed = urlparse(db_url)
            
            return {
                "dbname": parsed.path[1:],
                "user": parsed.username,
                "password": parsed.password,
                "host": parsed.hostname,
                "port": parsed.port or 5432
            }
        except Exception as e:
            logger.error(f"Error parsing DATABASE_URL: {e}")
    
    # Fallback to individual environment variables
    return {
        "dbname": os.environ.get("PGDATABASE", "postgres"),
        "user": os.environ.get("PGUSER", "postgres"),
        "password": os.environ.get("PGPASSWORD", "postgres"),
        "host": os.environ.get("PGHOST", "localhost"),
        "port": int(os.environ.get("PGPORT", 5432))
    }

def check_tables():
    """Check database tables and their contents"""
    db_params = get_db_connection_params()
    
    try:
        # Connect to the database
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Check pools table
        cursor.execute("SELECT COUNT(*) FROM pools")
        pool_count = cursor.fetchone()[0]
        logger.info(f"Found {pool_count} pools in database")
        
        # Check predictions table
        cursor.execute("SELECT COUNT(*) FROM predictions")
        prediction_count = cursor.fetchone()[0]
        logger.info(f"Found {prediction_count} predictions in database")
        
        # If there are pools but no predictions, we need to generate predictions
        if pool_count > 0 and prediction_count == 0:
            logger.warning("No predictions found for existing pools. Need to generate predictions.")
        
        # Sample pools
        if pool_count > 0:
            cursor.execute("SELECT pool_id, name, token_a, token_b FROM pools LIMIT 5")
            pools = cursor.fetchall()
            logger.info("Sample pools:")
            for pool in pools:
                logger.info(f"  {pool[0]} - {pool[1]} ({pool[2]}/{pool[3]})")
        
        # Sample predictions
        if prediction_count > 0:
            cursor.execute("""
            SELECT p.pool_id, pl.name, p.predicted_apr, p.performance_class, p.risk_score, p.prediction_timestamp
            FROM predictions p
            JOIN pools pl ON p.pool_id = pl.pool_id
            ORDER BY p.prediction_timestamp DESC
            LIMIT 5
            """)
            predictions = cursor.fetchall()
            logger.info("Sample predictions:")
            for pred in predictions:
                # Map performance class
                perf_map = {1: 'low', 2: 'medium', 3: 'high'}
                perf_class = perf_map.get(pred[3], f"Unknown ({pred[3]})")
                
                logger.info(f"  {pred[1]} - APR: {pred[2]:.2f}%, Perf: {perf_class}, Risk: {pred[4]:.2f} ({pred[5]})")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error checking database tables: {e}")
        return False

def check_mock_db_predictions():
    """Check predictions from mock database"""
    try:
        # Create a mock DBManager
        mock_db = DBManager()
        mock_db.use_mock = True
        
        # Get top predictions
        top_predictions = mock_db.get_top_predictions("apr", 10)
        
        if not top_predictions.empty:
            logger.info("Mock database top 5 APR predictions:")
            for i, (_, pred) in enumerate(top_predictions.head(5).iterrows()):
                logger.info(f"  {i+1}. {pred['pool_name']} - APR: {pred['predicted_apr']:.2f}%, "
                          f"Perf: {pred['performance_class']}, Risk: {pred['risk_score']:.2f}")
        else:
            logger.warning("No predictions found in mock database")
    except Exception as e:
        logger.error(f"Error checking mock predictions: {e}")

def check_db_operations():
    """Check predictions using the DBManager"""
    try:
        # Create a real DBManager
        real_db = DBManager()
        real_db.use_mock = False
        
        # Get top predictions
        top_predictions = real_db.get_top_predictions("apr", 10)
        
        if not top_predictions.empty:
            logger.info("Real database top 5 APR predictions:")
            for i, (_, pred) in enumerate(top_predictions.head(5).iterrows()):
                logger.info(f"  {i+1}. {pred['pool_name']} - APR: {pred['predicted_apr']:.2f}%, "
                          f"Perf: {pred['performance_class']}, Risk: {pred['risk_score']:.2f}")
        else:
            logger.warning("No predictions found in database via DBManager")
    except Exception as e:
        logger.error(f"Error checking real predictions: {e}")

if __name__ == "__main__":
    logger.info("Checking database prediction data")
    
    # Check database tables
    check_tables()
    
    # Check mock DB predictions
    check_mock_db_predictions()
    
    # Check real DB predictions via DBManager
    check_db_operations()