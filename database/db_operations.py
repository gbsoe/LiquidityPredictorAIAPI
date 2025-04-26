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
        
        # Real implementation would query the database
        # For now, just return mock data
        return self.mock_db.get_token_prices(token_symbols, days)
    
    def get_top_predictions(self, category="apr", limit=10, ascending=False):
        """Get top predictions based on category"""
        if self.use_mock:
            return self.mock_db.get_top_predictions(category, limit, ascending)
        
        # Real implementation would query the database
        # For now, just return mock data
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
        
        # Real implementation would query the database with time filtering
        # For example: SELECT * FROM predictions WHERE pool_id = ? AND timestamp > (NOW() - INTERVAL ? DAY)
        # For now, just get all mock data and filter by date
        all_predictions = self.mock_db.get_pool_predictions(pool_id)
        
        if days and not all_predictions.empty and 'timestamp' in all_predictions.columns:
            # Filter by days if we have timestamp data
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            return all_predictions[all_predictions['timestamp'] >= cutoff_date]
        
        return all_predictions