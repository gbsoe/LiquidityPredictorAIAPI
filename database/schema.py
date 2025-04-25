import logging

logger = logging.getLogger(__name__)

def initialize_schema(conn):
    """
    Initialize database schema
    """
    try:
        cursor = conn.cursor()
        
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
        logger.info("PostgreSQL database schema initialized successfully")
        cursor.close()
        return True
        
    except Exception as e:
        logger.error(f"Error initializing schema: {str(e)}")
        return False