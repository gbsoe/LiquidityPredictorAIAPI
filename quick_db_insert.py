"""
Direct database insert for testing purposes
"""

import os
import logging
import psycopg2
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def connect_to_db():
    """Connect to the PostgreSQL database"""
    # Get database connection parameters from environment
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        return None
    
    # Replace postgres:// with postgresql:// if needed
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    try:
        conn = psycopg2.connect(db_url)
        logger.info("Connected to database")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def create_test_data():
    """Create simple test data for predictions"""
    conn = connect_to_db()
    if not conn:
        return False
    
    cursor = conn.cursor()
    
    try:
        # Clean existing data
        logger.info("Cleaning existing data")
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM pools")
        conn.commit()
        
        # Insert 5 test pools
        logger.info("Inserting test pools")
        pools = [
            ("pool1", "SOL/USDC", "Raydium", "SOL", "USDC"),
            ("pool2", "BTC/USDC", "Orca", "BTC", "USDC"),
            ("pool3", "ETH/SOL", "Raydium", "ETH", "SOL"),
            ("pool4", "BONK/USDC", "Meteor", "BONK", "USDC"),
            ("pool5", "RAY/USDC", "Raydium", "RAY", "USDC")
        ]
        
        for pool_id, name, dex, token1, token2 in pools:
            cursor.execute("""
            INSERT INTO pools (pool_id, name, dex, token1, token2)
            VALUES (%s, %s, %s, %s, %s)
            """, (pool_id, name, dex, token1, token2))
        
        conn.commit()
        
        # Check how many pools were inserted
        cursor.execute("SELECT COUNT(*) FROM pools")
        pool_count = cursor.fetchone()[0]
        logger.info(f"Inserted {pool_count} pools")
        
        # Insert current predictions for all pools
        logger.info("Inserting current predictions")
        for pool_id, name, _, _, _ in pools:
            # Generate realistic APR (between 3% and 35%)
            apr = random.uniform(3.0, 35.0)
            
            # Risk score (0-1, lower is better)
            risk = min(0.9, apr / 50.0 + random.uniform(-0.1, 0.1))
            risk = max(0.1, risk)
            
            # Performance class
            if apr > 20:
                perf_class = 3  # high
            elif apr > 10:
                perf_class = 2  # medium
            else:
                perf_class = 1  # low
                
            cursor.execute("""
            INSERT INTO predictions (
                pool_id, predicted_apr, risk_score, performance_class, prediction_timestamp
            ) VALUES (%s, %s, %s, %s, NOW())
            """, (pool_id, apr, risk, perf_class))
        
        conn.commit()
        
        # Generate historical data for first 3 pools
        logger.info("Inserting historical predictions")
        for pool_id, name, _, _, _ in pools[:3]:
            # Base APR
            base_apr = random.uniform(10.0, 25.0)
            
            # Historical data for last 14 days
            for day in range(14, 0, -1):
                # Vary APR over time with trend
                apr = base_apr + (day * 0.3 * random.choice([-1, 1])) + random.uniform(-2.0, 2.0)
                apr = max(3.0, min(35.0, apr))
                
                # Risk follows APR
                risk = min(0.9, apr / 50.0 + random.uniform(-0.1, 0.1))
                risk = max(0.1, risk)
                
                # Performance class
                if apr > 20:
                    perf_class = 3  # high
                elif apr > 10:
                    perf_class = 2  # medium
                else:
                    perf_class = 1  # low
                
                # Format the interval properly
                interval = f"{day} days"
                cursor.execute("""
                INSERT INTO predictions (
                    pool_id, predicted_apr, risk_score, performance_class, 
                    prediction_timestamp
                ) VALUES (%s, %s, %s, %s, NOW() - INTERVAL %s)
                """, (pool_id, apr, risk, perf_class, interval))
        
        conn.commit()
        
        # Check how many predictions were inserted
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        logger.info(f"Inserted {pred_count} predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating test data: {e}")
        conn.rollback()
        return False
    
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    if create_test_data():
        logger.info("Successfully created test data")
    else:
        logger.error("Failed to create test data")