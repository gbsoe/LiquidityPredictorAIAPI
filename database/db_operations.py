import os
import psycopg2
import pandas as pd
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from .schema import initialize_database, get_db_connection, get_sqlalchemy_engine, DATABASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('db_operations')

def ensure_database_exists():
    """Ensure the database exists and has the correct schema."""
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable is not set")
        return False
    
    logger.info("Initializing PostgreSQL database schema")
    return initialize_database()

class DBManager:
    """Database operations manager for liquidity pools data."""
    
    def __init__(self):
        """Initialize the database manager."""
        ensure_database_exists()
        # Initialize SQLAlchemy engine for pandas operations
        self.engine = get_sqlalchemy_engine()
    
    def get_connection(self):
        """Get a connection to the database."""
        return get_db_connection()
    
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
            if self.engine:
                # Use SQLAlchemy for pandas
                return pd.read_sql_query(query, self.engine, params=params)
            else:
                # Fallback to direct connection
                conn = self.get_connection()
                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Error executing query to dataframe: {e}")
            raise
        finally:
            if 'conn' in locals() and conn:
                conn.close()
    
    # Pool data operations
    
    def get_all_pools(self):
        """Get all pools data."""
        query = "SELECT * FROM pool_data"
        return self.query_to_dataframe(query)
    
    def get_pool_by_id(self, pool_id):
        """Get pool data by ID."""
        query = "SELECT * FROM pool_data WHERE pool_id = %s"
        return self.query_to_dataframe(query, params=(pool_id,))
    
    def save_pool_data(self, pool_id, name, liquidity, volume_24h, apr, timestamp=None):
        """Save or update pool data."""
        if timestamp is None:
            timestamp = datetime.now()
            
        query = """
            INSERT INTO pool_data 
            (pool_id, name, liquidity, volume_24h, apr, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (pool_id) DO UPDATE SET
                name = EXCLUDED.name,
                liquidity = EXCLUDED.liquidity,
                volume_24h = EXCLUDED.volume_24h,
                apr = EXCLUDED.apr,
                timestamp = EXCLUDED.timestamp
        """
        return self.execute_query(
            query, 
            params=(pool_id, name, liquidity, volume_24h, apr, timestamp),
            fetch=False
        )
    
    def save_pool_metrics(self, pool_id, liquidity, volume, apr, timestamp=None):
        """Save pool metrics data."""
        if timestamp is None:
            timestamp = datetime.now()
            
        query = """
            INSERT INTO pool_metrics
            (pool_id, liquidity, volume, apr, timestamp)
            VALUES (%s, %s, %s, %s, %s)
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
            WHERE pool_id = %s AND timestamp >= %s
            ORDER BY timestamp
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        return self.query_to_dataframe(query, params=(pool_id, cutoff_date))
    
    # Token price operations
    
    def save_token_price(self, token_symbol, price_usd, timestamp=None):
        """Save current token price."""
        if timestamp is None:
            timestamp = datetime.now()
            
        query = """
            INSERT INTO token_prices
            (token_symbol, price_usd, timestamp)
            VALUES (%s, %s, %s)
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
            WHERE token_symbol = %s AND timestamp >= %s
            ORDER BY timestamp
        """
        cutoff_date = datetime.now() - timedelta(days=days)
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
            timestamp = datetime.now()
            
        query = """
            INSERT INTO blockchain_stats
            (slot, block_height, avg_tps, sol_price, timestamp)
            VALUES (%s, %s, %s, %s, %s)
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
            WHERE timestamp >= %s
            ORDER BY timestamp
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        return self.query_to_dataframe(query, params=(cutoff_date,))
    
    # Model predictions operations
    
    def save_prediction(self, pool_id, predicted_apr, performance_class, risk_score, model_version, timestamp=None):
        """Save model prediction results."""
        if timestamp is None:
            timestamp = datetime.now()
            
        query = """
            INSERT INTO model_predictions
            (pool_id, predicted_apr, performance_class, risk_score, prediction_timestamp, model_version)
            VALUES (%s, %s, %s, %s, %s, %s)
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
            LIMIT %s
        """
        return self.query_to_dataframe(query, params=(limit,))
    
    def get_pool_predictions(self, pool_id, days=30):
        """Get historical predictions for a specific pool."""
        query = """
            SELECT * FROM model_predictions 
            WHERE pool_id = %s AND prediction_timestamp >= %s
            ORDER BY prediction_timestamp
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        return self.query_to_dataframe(query, params=(pool_id, cutoff_date))
    
    # Data aggregation queries
    
    def get_top_pools_by_liquidity(self, limit=10):
        """Get top pools by liquidity."""
        query = """
            SELECT * FROM pool_data
            ORDER BY liquidity DESC
            LIMIT %s
        """
        return self.query_to_dataframe(query, params=(limit,))
    
    def get_top_pools_by_volume(self, limit=10):
        """Get top pools by 24h volume."""
        query = """
            SELECT * FROM pool_data
            ORDER BY volume_24h DESC
            LIMIT %s
        """
        return self.query_to_dataframe(query, params=(limit,))
    
    def get_top_pools_by_apr(self, limit=10):
        """Get top pools by APR."""
        query = """
            SELECT * FROM pool_data
            ORDER BY apr DESC
            LIMIT %s
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
            WHERE timestamp >= %s
            GROUP BY DATE(timestamp)
            ORDER BY date
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        return self.query_to_dataframe(query, params=(cutoff_date,))
