"""
Initialize prediction tables in the PostgreSQL database
"""

import os
import logging
import psycopg2
from psycopg2 import sql
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

def create_prediction_tables():
    """Create tables for storing prediction data"""
    db_params = get_db_connection_params()
    
    try:
        # Connect to the database
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Enable psycopg2 to automatically convert timestamp with time zone
        cursor.execute("SET TIME ZONE 'UTC'")
        
        # Create pools table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pools (
            pool_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            token_a TEXT NOT NULL,
            token_b TEXT NOT NULL,
            dex TEXT,
            category TEXT,
            liquidity FLOAT DEFAULT 0,
            volume FLOAT DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """)
        
        # Create predictions table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            pool_id TEXT NOT NULL REFERENCES pools(pool_id),
            predicted_apr FLOAT NOT NULL,
            risk_score FLOAT NOT NULL,
            performance_class INTEGER NOT NULL,
            prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            model_version TEXT
        )
        """)
        
        # Create index on pool_id for predictions table
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_pool_id ON predictions(pool_id)
        """)
        
        # Create index on prediction_timestamp for time-based queries
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(prediction_timestamp)
        """)
        
        # Commit changes
        conn.commit()
        
        # Check table existence and report results
        tables = ["pools", "predictions"]
        for table in tables:
            cursor.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}')")
            exists = cursor.fetchone()[0]
            if exists:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"Table '{table}' exists with {count} records")
            else:
                logger.error(f"Table '{table}' was not created successfully")
        
        cursor.close()
        conn.close()
        
        logger.info("Database tables set up successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up database tables: {e}")
        return False

if __name__ == "__main__":
    logger.info("Initializing prediction tables in database")
    
    success = create_prediction_tables()
    
    if success:
        logger.info("Database initialization completed successfully")
    else:
        logger.error("Database initialization failed")