import os
import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta
from .schema import initialize_database, DATABASE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('db_operations')

def ensure_database_exists():
    """Ensure the database exists and has the correct schema."""
    if not os.path.exists(DATABASE_PATH):
        logger.info(f"Database not found at {DATABASE_PATH}. Creating new database.")
        return initialize_database()
    return True

class DBManager:
    """Database operations manager for liquidity pools data."""
    
    def __init__(self):
        """Initialize the database manager."""
        self.db_path = DATABASE_PATH
        ensure_database_exists()
    
    def get_connection(self):
        """Get a connection to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def execute_query(self, query, params=None, fetch=True):
        """Execute a SQL query and optionally return results."""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                return cursor.fetchall()
            else:
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error executing query: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_many(self, query, params_list):
        """Execute a SQL query with multiple parameter sets."""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error executing batch query: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def query_to_dataframe(self, query, params=None):
        """Execute a SQL query and return results as a pandas DataFrame."""
        try:
            conn = self.get_connection()
            return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Error executing query to dataframe: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    # Pool data operations
    
    def get_all_pools(self):
        """Get all pools data."""
        query = "SELECT * FROM pool_data"
        return self.query_to_dataframe(query)
    
    def get_pool_by_id(self, pool_id):
        """Get pool data by ID."""
        query = "SELECT * FROM pool_data WHERE pool_id = ?"
        return self.query_to_dataframe(query, params=(pool_id,))
    
    def save_pool_data(self, pool_id, name, liquidity, volume_24h, apr, timestamp=None):
        """Save or update pool data."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        query = """
            INSERT OR REPLACE INTO pool_data 
            (pool_id, name, liquidity, volume_24h, apr, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        return self.execute_query(
            query, 
            params=(pool_id, name, liquidity, volume_24h, apr, timestamp),
            fetch=False
        )
    
    def save_pool_metrics(self, pool_id, liquidity, volume, apr, timestamp=None):
        """Save pool metrics data."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        query = """
            INSERT INTO pool_metrics
            (pool_id, liquidity, volume, apr, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """
        return self.execute_query(
            query, 
            params=(pool_id, liquidity, volume, apr, timestamp),
            fetch=False
        )
    
    def get_pool_metrics(self, pool_id, days=30):
        """Get historical metrics for a pool."""
        query = """
            SELECT * FROM pool_metrics 
            WHERE pool_id = ? AND timestamp >= ?
            ORDER BY timestamp
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        return self.query_to_dataframe(query, params=(pool_id, cutoff_date))
    
    # Token price operations
    
    def save_token_price(self, token_symbol, price_usd, timestamp=None):
        """Save current token price."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        query = """
            INSERT INTO token_prices
            (token_symbol, price_usd, timestamp)
            VALUES (?, ?, ?)
        """
        return self.execute_query(
            query, 
            params=(token_symbol, price_usd, timestamp),
            fetch=False
        )
    
    def get_token_price_history(self, token_symbol, days=30):
        """Get historical price data for a token."""
        query = """
            SELECT * FROM token_prices 
            WHERE token_symbol = ? AND timestamp >= ?
            ORDER BY timestamp
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        return self.query_to_dataframe(query, params=(token_symbol, cutoff_date))
    
    def get_latest_token_prices(self):
        """Get the latest price for each token."""
        query = """
            SELECT t.* FROM token_prices t
            INNER JOIN (
                SELECT token_symbol, MAX(timestamp) as max_timestamp
                FROM token_prices
                GROUP BY token_symbol
            ) m ON t.token_symbol = m.token_symbol AND t.timestamp = m.max_timestamp
        """
        return self.query_to_dataframe(query)
    
    # Blockchain stats operations
    
    def save_blockchain_stats(self, slot, block_height, avg_tps, sol_price, timestamp=None):
        """Save blockchain statistics."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        query = """
            INSERT INTO blockchain_stats
            (slot, block_height, avg_tps, sol_price, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """
        return self.execute_query(
            query, 
            params=(slot, block_height, avg_tps, sol_price, timestamp),
            fetch=False
        )
    
    def get_blockchain_stats(self, days=7):
        """Get historical blockchain statistics."""
        query = """
            SELECT * FROM blockchain_stats 
            WHERE timestamp >= ?
            ORDER BY timestamp
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        return self.query_to_dataframe(query, params=(cutoff_date,))
    
    # Model predictions operations
    
    def save_prediction(self, pool_id, predicted_apr, performance_class, risk_score, model_version, timestamp=None):
        """Save model prediction results."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        query = """
            INSERT INTO model_predictions
            (pool_id, predicted_apr, performance_class, risk_score, prediction_timestamp, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        return self.execute_query(
            query, 
            params=(pool_id, predicted_apr, performance_class, risk_score, timestamp, model_version),
            fetch=False
        )
    
    def get_latest_predictions(self, limit=100):
        """Get the latest predictions for each pool."""
        query = """
            SELECT p.* FROM model_predictions p
            INNER JOIN (
                SELECT pool_id, MAX(prediction_timestamp) as max_timestamp
                FROM model_predictions
                GROUP BY pool_id
            ) m ON p.pool_id = m.pool_id AND p.prediction_timestamp = m.max_timestamp
            LIMIT ?
        """
        return self.query_to_dataframe(query, params=(limit,))
    
    def get_pool_predictions(self, pool_id, days=30):
        """Get historical predictions for a specific pool."""
        query = """
            SELECT * FROM model_predictions 
            WHERE pool_id = ? AND prediction_timestamp >= ?
            ORDER BY prediction_timestamp
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        return self.query_to_dataframe(query, params=(pool_id, cutoff_date))
    
    # Data aggregation queries
    
    def get_top_pools_by_liquidity(self, limit=10):
        """Get top pools by liquidity."""
        query = """
            SELECT * FROM pool_data
            ORDER BY liquidity DESC
            LIMIT ?
        """
        return self.query_to_dataframe(query, params=(limit,))
    
    def get_top_pools_by_volume(self, limit=10):
        """Get top pools by 24h volume."""
        query = """
            SELECT * FROM pool_data
            ORDER BY volume_24h DESC
            LIMIT ?
        """
        return self.query_to_dataframe(query, params=(limit,))
    
    def get_top_pools_by_apr(self, limit=10):
        """Get top pools by APR."""
        query = """
            SELECT * FROM pool_data
            ORDER BY apr DESC
            LIMIT ?
        """
        return self.query_to_dataframe(query, params=(limit,))
    
    def get_avg_metrics_by_day(self, days=30):
        """Get average daily metrics across all pools."""
        query = """
            SELECT 
                DATE(timestamp) as date,
                AVG(liquidity) as avg_liquidity,
                AVG(volume) as avg_volume,
                AVG(apr) as avg_apr,
                COUNT(DISTINCT pool_id) as pool_count
            FROM pool_metrics
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        return self.query_to_dataframe(query, params=(cutoff_date,))
