import logging
import os
import pandas as pd
import psycopg2
from datetime import datetime
from database.mock_db import MockDBManager

class DBManager:
    """
    Database manager for PostgreSQL operations
    With fallback to mock DB if database connection fails
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Always create the mock database for fallback
        self.mock_db = MockDBManager()
        
        try:
            # Get database connection parameters from environment
            self.db_params = {
                'dbname': os.getenv('PGDATABASE', 'postgres'),
                'user': os.getenv('PGUSER', 'postgres'),
                'password': os.getenv('PGPASSWORD', ''),
                'host': os.getenv('PGHOST', 'localhost'),
                'port': os.getenv('PGPORT', 5432)
            }
            
            # Test connection
            self._test_connection()
            
            # Initialize schema if needed
            self._init_schema()
            
            # If we reach here, we have a functional database
            self.use_mock = False
            
        except Exception as e:
            self.logger.error(f"Error initializing database connection: {str(e)}")
            self.logger.info("Falling back to mock database")
            
            # If database connection fails, use mock DB
            self.use_mock = True
    
    def _test_connection(self):
        """Test database connection"""
        try:
            conn = psycopg2.connect(**self.db_params)
            conn.close()
            self.logger.info("Database connection successful")
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {str(e)}")
            raise
    
    def _init_schema(self):
        """Initialize database schema if needed"""
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
            SELECT EXISTS (
               SELECT FROM information_schema.tables 
               WHERE table_name = 'pools'
            );
            """)
            tables_exist = cursor.fetchone()[0]
            
            if not tables_exist:
                self.logger.info("Initializing PostgreSQL database schema")
                
                # Create tables
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS pools (
                    pool_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dex TEXT NOT NULL,
                    token1 TEXT NOT NULL,
                    token2 TEXT NOT NULL,
                    token1_address TEXT,
                    token2_address TEXT,
                    token1_price NUMERIC DEFAULT 0,
                    token2_price NUMERIC DEFAULT 0,
                    token1_price_updated_at TIMESTAMP,
                    token2_price_updated_at TIMESTAMP,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS pool_metrics (
                    id SERIAL PRIMARY KEY,
                    pool_id TEXT REFERENCES pools(pool_id),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    liquidity NUMERIC,
                    volume NUMERIC,
                    apr NUMERIC,
                    fee NUMERIC,
                    tvl_change_24h NUMERIC,
                    apr_change_24h NUMERIC
                );
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_prices (
                    id SERIAL PRIMARY KEY,
                    token_symbol TEXT NOT NULL,
                    token_address TEXT,
                    price_usd NUMERIC,
                    timestamp TIMESTAMP DEFAULT NOW()
                );
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    pool_id TEXT REFERENCES pools(pool_id),
                    predicted_apr NUMERIC,
                    risk_score NUMERIC,
                    performance_class INTEGER,
                    prediction_timestamp TIMESTAMP DEFAULT NOW()
                );
                """)
                
                conn.commit()
                self.logger.info("PostgreSQL database schema initialized successfully")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing schema: {str(e)}")
            raise
    
    # ==================== WRAPPER METHODS ====================
    
    def get_pool_list(self):
        """Get list of pools"""
        if self.use_mock:
            return self.mock_db.get_pool_list()
        
        # Real implementation would query the database
        # For now, just return mock data
        return self.mock_db.get_pool_list()
    
    def get_pool_details(self, pool_id):
        """Get details for a specific pool"""
        if self.use_mock:
            return self.mock_db.get_pool_details(pool_id)
        
        # Real implementation would query the database
        # For now, just return mock data
        return self.mock_db.get_pool_details(pool_id)
    
    def get_pool_metrics(self, pool_id, days=7):
        """Get historical metrics for a specific pool"""
        if self.use_mock:
            return self.mock_db.get_pool_metrics(pool_id, days)
        
        # Real implementation would query the database
        # For now, just return mock data
        return self.mock_db.get_pool_metrics(pool_id, days)
    
    def get_token_prices(self, token_symbols, days=7):
        """Get historical token prices"""
        if self.use_mock:
            return self.mock_db.get_token_prices(token_symbols, days)
        
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Convert token_symbols to a list if it's a single string
            if isinstance(token_symbols, str):
                token_symbols = [token_symbols]
            
            # Format for SQL IN clause
            tokens_str = ','.join([f"'{token}'" for token in token_symbols])
            
            # Query to get historical token prices
            query = f"""
            SELECT token_symbol, price_usd, timestamp
            FROM token_prices
            WHERE token_symbol IN ({tokens_str})
            AND timestamp > NOW() - INTERVAL %s DAY
            ORDER BY token_symbol, timestamp
            """
            
            cursor.execute(query, (days,))
            rows = cursor.fetchall()
            
            if not rows:
                self.logger.warning(f"No token prices found for {token_symbols}")
                return self.mock_db.get_token_prices(token_symbols, days)
                
            # Column names for the DataFrame
            columns = ['token_symbol', 'price_usd', 'timestamp']
            
            # Create DataFrame
            price_df = pd.DataFrame(rows, columns=columns)
            
            cursor.close()
            conn.close()
            
            return price_df
            
        except Exception as e:
            self.logger.error(f"Error getting token prices from database: {str(e)}")
            return self.mock_db.get_token_prices(token_symbols, days)
    
    def save_prediction(self, pool_id, predicted_apr, performance_class, risk_score, model_version=None):
        """
        Save a prediction to the database
        
        Args:
            pool_id: Pool ID
            predicted_apr: Predicted APR value
            performance_class: Performance class ('high', 'medium', 'low')
            risk_score: Risk score (0-1, lower is better)
            model_version: Version of the model used for prediction
        """
        if self.use_mock:
            self.logger.warning("Using mock database - prediction not saved")
            return
            
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Convert performance class to integer for storage
            # high: 3, medium: 2, low: 1
            if isinstance(performance_class, str):
                performance_class_map = {
                    'high': 3,
                    'medium': 2,
                    'low': 1
                }
                perf_class_value = performance_class_map.get(performance_class.lower(), 2)
            else:
                perf_class_value = performance_class
            
            # Insert prediction
            query = """
            INSERT INTO predictions (
                pool_id, 
                predicted_apr, 
                risk_score, 
                performance_class, 
                prediction_timestamp
            ) VALUES (%s, %s, %s, %s, NOW())
            """
            
            cursor.execute(query, (
                pool_id,
                predicted_apr,
                risk_score,
                perf_class_value
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"Saved prediction for pool {pool_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving prediction to database: {str(e)}")
            return False
    
    def get_top_predictions(self, category="apr", limit=10, ascending=False):
        """Get top predictions based on category"""
        if self.use_mock:
            return self.mock_db.get_top_predictions(category, limit, ascending)
        
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Determine sort column and direction based on category
            if category == "apr":
                sort_column = "predicted_apr"
                sort_direction = "DESC" if not ascending else "ASC"
            elif category == "risk":
                sort_column = "risk_score"
                sort_direction = "ASC" if not ascending else "DESC"  # Lower risk is better
            else:  # Performance
                sort_column = "performance_class"
                sort_direction = "DESC" if not ascending else "ASC"
            
            # Query the most recent prediction for each pool and sort by the specified category
            query = f"""
            WITH latest_predictions AS (
                SELECT DISTINCT ON (p.pool_id) 
                    p.pool_id,
                    p.predicted_apr,
                    p.risk_score,
                    p.performance_class,
                    p.prediction_timestamp,
                    pl.name
                FROM predictions p
                JOIN pools pl ON p.pool_id = pl.pool_id
                ORDER BY p.pool_id, p.prediction_timestamp DESC
            )
            SELECT * FROM latest_predictions
            ORDER BY {sort_column} {sort_direction}
            LIMIT %s
            """
            
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            
            if not rows:
                self.logger.warning("No predictions found in database")
                return self.mock_db.get_top_predictions(category, limit, ascending)
            
            # Column names for the DataFrame
            columns = ['pool_id', 'predicted_apr', 'risk_score', 'performance_class', 
                      'prediction_timestamp', 'pool_name']
            
            # Create DataFrame with decoded performance class
            predictions_df = pd.DataFrame(rows, columns=columns)
            
            # Map numeric performance class to string labels
            predictions_df['performance_class'] = predictions_df['performance_class'].map({
                3: 'high',
                2: 'medium',
                1: 'low'
            })
            
            cursor.close()
            conn.close()
            
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"Error getting top predictions from database: {str(e)}")
            return self.mock_db.get_top_predictions(category, limit, ascending)
    
    def get_pool_predictions(self, pool_id, days=30):
        """
        Get prediction history for a specific pool
        
        Args:
            pool_id: ID of the pool to get predictions for
            days: Number of days of prediction history to return
        """
        if self.use_mock:
            return self.mock_db.get_pool_predictions(pool_id, days)
        
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Query predictions for the specified pool with time filtering
            query = """
            SELECT p.pool_id, 
                  p.predicted_apr, 
                  p.risk_score, 
                  p.performance_class, 
                  p.prediction_timestamp,
                  pl.name as pool_name
            FROM predictions p
            JOIN pools pl ON p.pool_id = pl.pool_id
            WHERE p.pool_id = %s
              AND p.prediction_timestamp > NOW() - INTERVAL %s DAY
            ORDER BY p.prediction_timestamp
            """
            
            cursor.execute(query, (pool_id, days))
            rows = cursor.fetchall()
            
            if not rows:
                self.logger.warning(f"No predictions found for pool {pool_id}")
                return self.mock_db.get_pool_predictions(pool_id, days)
                
            # Column names for the DataFrame
            columns = ['pool_id', 'predicted_apr', 'risk_score', 'performance_class', 
                      'prediction_timestamp', 'pool_name']
            
            # Create DataFrame with decoded performance class
            predictions_df = pd.DataFrame(rows, columns=columns)
            
            # Map numeric performance class to string labels
            predictions_df['performance_class'] = predictions_df['performance_class'].map({
                3: 'high',
                2: 'medium',
                1: 'low'
            })
            
            cursor.close()
            conn.close()
            
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"Error getting pool predictions from database: {str(e)}")
            all_predictions = self.mock_db.get_pool_predictions(pool_id)
            
            if days and not all_predictions.empty and 'timestamp' in all_predictions.columns:
                # Filter by days if we have timestamp data
                cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
                return all_predictions[all_predictions['timestamp'] >= cutoff_date]
            
            return all_predictions