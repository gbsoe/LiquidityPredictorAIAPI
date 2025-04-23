import os
import sqlite3
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('database_schema')

# Database path
DATABASE_PATH = os.getenv('DATABASE_PATH', './liquidity_pools.db')

# SQL statements to create tables
CREATE_POOL_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS pool_data (
    pool_id TEXT PRIMARY KEY,
    name TEXT,
    liquidity REAL,
    volume_24h REAL,
    apr REAL,
    timestamp TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_POOL_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS pool_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pool_id TEXT,
    liquidity REAL,
    volume REAL,
    apr REAL,
    timestamp TEXT,
    FOREIGN KEY (pool_id) REFERENCES pool_data (pool_id)
)
"""

CREATE_POOL_PRICE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS pool_price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pool_id TEXT,
    price_ratio REAL,
    timestamp TEXT,
    FOREIGN KEY (pool_id) REFERENCES pool_data (pool_id)
)
"""

CREATE_TOKEN_PRICES_TABLE = """
CREATE TABLE IF NOT EXISTS token_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_symbol TEXT,
    price_usd REAL,
    timestamp TEXT
)
"""

CREATE_TOKEN_PRICE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS token_price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id TEXT,
    price_usd REAL,
    timestamp TEXT
)
"""

CREATE_BLOCKCHAIN_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS blockchain_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot INTEGER,
    block_height INTEGER,
    avg_tps REAL,
    sol_price REAL,
    timestamp TEXT
)
"""

CREATE_MODEL_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pool_id TEXT,
    predicted_apr REAL,
    performance_class TEXT,
    risk_score REAL,
    prediction_timestamp TEXT,
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

def initialize_database():
    """Initialize the database with the required schema."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(DATABASE_PATH)), exist_ok=True)
        
        # Connect to the database (creates it if it doesn't exist)
        conn = sqlite3.connect(DATABASE_PATH)
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
        
        logger.info(f"Database initialized successfully at {DATABASE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    initialize_database()
