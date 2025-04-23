import os
import sys
import logging
import psycopg2
from sqlalchemy import create_engine, text
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('database_schema')

# Get database connection string from environment
DATABASE_URL = os.environ.get('DATABASE_URL')

# SQL statements to create tables - PostgreSQL syntax
CREATE_POOL_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS pool_data (
    pool_id TEXT PRIMARY KEY,
    name TEXT,
    liquidity FLOAT,
    volume_24h FLOAT,
    apr FLOAT,
    timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_POOL_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS pool_metrics (
    id SERIAL PRIMARY KEY,
    pool_id TEXT,
    liquidity FLOAT,
    volume FLOAT,
    apr FLOAT,
    timestamp TIMESTAMP,
    FOREIGN KEY (pool_id) REFERENCES pool_data (pool_id)
)
"""

CREATE_POOL_PRICE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS pool_price_history (
    id SERIAL PRIMARY KEY,
    pool_id TEXT,
    price_ratio FLOAT,
    timestamp TIMESTAMP,
    FOREIGN KEY (pool_id) REFERENCES pool_data (pool_id)
)
"""

CREATE_TOKEN_PRICES_TABLE = """
CREATE TABLE IF NOT EXISTS token_prices (
    id SERIAL PRIMARY KEY,
    token_symbol TEXT,
    price_usd FLOAT,
    timestamp TIMESTAMP
)
"""

CREATE_TOKEN_PRICE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS token_price_history (
    id SERIAL PRIMARY KEY,
    token_id TEXT,
    price_usd FLOAT,
    timestamp TIMESTAMP
)
"""

CREATE_BLOCKCHAIN_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS blockchain_stats (
    id SERIAL PRIMARY KEY,
    slot INTEGER,
    block_height INTEGER,
    avg_tps FLOAT,
    sol_price FLOAT,
    timestamp TIMESTAMP
)
"""

CREATE_MODEL_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    pool_id TEXT,
    predicted_apr FLOAT,
    performance_class TEXT,
    risk_score FLOAT,
    prediction_timestamp TIMESTAMP,
    model_version TEXT,
    FOREIGN KEY (pool_id) REFERENCES pool_data (pool_id)
)
"""

CREATE_POOL_METRICS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_pool_metrics_pool_id ON pool_metrics (pool_id)
"""

CREATE_POOL_METRICS_TIMESTAMP_INDEX = """
CREATE INDEX IF NOT EXISTS idx_pool_metrics_timestamp ON pool_metrics (timestamp)
"""

CREATE_TOKEN_PRICES_SYMBOL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_token_prices_symbol ON token_prices (token_symbol)
"""

CREATE_TOKEN_PRICES_TIMESTAMP_INDEX = """
CREATE INDEX IF NOT EXISTS idx_token_prices_timestamp ON token_prices (timestamp)
"""

def get_db_connection():
    """Get a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def initialize_database():
    """Initialize the database with the required schema."""
    try:
        if not DATABASE_URL:
            logger.error("DATABASE_URL environment variable is not set")
            return False

        # Connect to the database
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute(CREATE_POOL_DATA_TABLE)
        cursor.execute(CREATE_POOL_METRICS_TABLE)
        cursor.execute(CREATE_POOL_PRICE_HISTORY_TABLE)
        cursor.execute(CREATE_TOKEN_PRICES_TABLE)
        cursor.execute(CREATE_TOKEN_PRICE_HISTORY_TABLE)
        cursor.execute(CREATE_BLOCKCHAIN_STATS_TABLE)
        cursor.execute(CREATE_MODEL_PREDICTIONS_TABLE)
        
        # Create indexes
        cursor.execute(CREATE_POOL_METRICS_INDEX)
        cursor.execute(CREATE_POOL_METRICS_TIMESTAMP_INDEX)
        cursor.execute(CREATE_TOKEN_PRICES_SYMBOL_INDEX)
        cursor.execute(CREATE_TOKEN_PRICES_TIMESTAMP_INDEX)
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info("PostgreSQL database schema initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

def get_sqlalchemy_engine():
    """Get SQLAlchemy engine for the database."""
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        logger.error(f"Error creating SQLAlchemy engine: {e}")
        return None

if __name__ == "__main__":
    initialize_database()
