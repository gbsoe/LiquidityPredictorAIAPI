"""
PostgreSQL Database Initialization Script for SolPool Insight

This script ensures the PostgreSQL database is properly set up with the required schema.
It creates all necessary tables for the application to store liquidity pool data,
metrics, token prices, and predictions.
"""

import os
import psycopg2
import logging
from sqlalchemy import create_engine, text
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('db_init')

def get_database_url():
    """Get the database URL from environment variables"""
    db_url = os.getenv('DATABASE_URL')
    
    if db_url:
        logger.info("Found DATABASE_URL in environment variables")
        return db_url
        
    logger.error("DATABASE_URL not found in environment variables")
    return None

def test_connection(db_url):
    """Test the database connection with SQLAlchemy"""
    try:
        logger.info("Testing database connection with SQLAlchemy...")
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        logger.info("✓ Successfully connected to PostgreSQL database with SQLAlchemy")
        return True
    except Exception as e:
        logger.error(f"Error connecting to database with SQLAlchemy: {str(e)}")
        return False

def test_psycopg2_connection(db_url):
    """Test the database connection with psycopg2"""
    try:
        logger.info("Testing database connection with psycopg2...")
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        logger.info("✓ Successfully connected to PostgreSQL database with psycopg2")
        return True
    except Exception as e:
        logger.error(f"Error connecting to database with psycopg2: {str(e)}")
        return False

def initialize_schema(db_url):
    """Initialize database schema"""
    try:
        logger.info("Initializing database schema...")
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Create liquidity_pools table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS liquidity_pools (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dex TEXT NOT NULL,
            category TEXT NOT NULL,
            token1_symbol TEXT NOT NULL,
            token2_symbol TEXT NOT NULL,
            token1_address TEXT NOT NULL,
            token2_address TEXT NOT NULL,
            liquidity FLOAT NOT NULL,
            volume_24h FLOAT NOT NULL,
            apr FLOAT NOT NULL,
            fee FLOAT NOT NULL,
            version TEXT NOT NULL,
            apr_change_24h FLOAT NOT NULL,
            apr_change_7d FLOAT NOT NULL,
            tvl_change_24h FLOAT NOT NULL,
            tvl_change_7d FLOAT NOT NULL,
            prediction_score FLOAT NOT NULL,
            apr_change_30d FLOAT DEFAULT 0.0,
            tvl_change_30d FLOAT DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        # Create pools table from schema.py
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
        
        # Create pool_metrics table
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
        
        # Create token_prices table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS token_prices (
            id SERIAL PRIMARY KEY,
            token_symbol TEXT NOT NULL,
            token_address TEXT,
            price_usd NUMERIC,
            timestamp TIMESTAMP DEFAULT NOW()
        );
        """)
        
        # Create predictions table
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
        
        # Create pool_price_history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pool_price_history (
            id SERIAL PRIMARY KEY,
            pool_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW(),
            liquidity FLOAT NOT NULL DEFAULT 0,
            volume_24h FLOAT NOT NULL DEFAULT 0,
            apr_24h FLOAT NOT NULL DEFAULT 0,
            apr_7d FLOAT NOT NULL DEFAULT 0,
            apr_30d FLOAT NOT NULL DEFAULT 0,
            token1_price FLOAT NOT NULL DEFAULT 0,
            token2_price FLOAT NOT NULL DEFAULT 0
        );
        """)
        
        # Commit the changes
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✓ Successfully initialized database schema")
        return True
    except Exception as e:
        logger.error(f"Error initializing database schema: {str(e)}")
        return False

def main():
    """Main function"""
    logger.info("Starting database initialization...")
    
    # Get database URL
    db_url = get_database_url()
    if not db_url:
        logger.error("No database URL found. Aborting.")
        sys.exit(1)
    
    # Test connection with SQLAlchemy
    if not test_connection(db_url):
        logger.warning("SQLAlchemy connection test failed.")
    
    # Test connection with psycopg2
    if not test_psycopg2_connection(db_url):
        logger.error("psycopg2 connection test failed. Aborting.")
        sys.exit(1)
    
    # Initialize schema
    if not initialize_schema(db_url):
        logger.error("Schema initialization failed. Aborting.")
        sys.exit(1)
    
    logger.info("✓ Database initialization completed successfully")

if __name__ == "__main__":
    main()