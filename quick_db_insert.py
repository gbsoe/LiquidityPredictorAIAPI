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
            # Generate APR with a wide range (including some high-APR pools)
            # Pool 1: Low to medium APR (5-25%)
            # Pool 2: Medium to high APR (15-80%)
            # Pool 3: Very high APR (100-550%)
            # Pool 4: Extremely high APR (400-999%)
            # Pool 5: Normal range APR (5-35%)
            if pool_id == "pool1":
                apr = random.uniform(5.0, 25.0)
            elif pool_id == "pool2":
                apr = random.uniform(15.0, 80.0)
            elif pool_id == "pool3":
                apr = random.uniform(100.0, 550.0)
            elif pool_id == "pool4":
                apr = random.uniform(400.0, 999.0)
            else:  # pool5
                apr = random.uniform(5.0, 35.0)
            
            # Risk score (0-1, lower is better)
            # Higher APR typically means higher risk
            if apr > 100:
                # High APR pools have high risk
                risk = random.uniform(0.7, 0.95)
            elif apr > 50:
                # Medium-high APR pools have medium-high risk
                risk = random.uniform(0.5, 0.8)
            else:
                # Lower APR pools have lower risk
                risk = random.uniform(0.1, 0.5)
            
            # Performance class - based on both APR and risk
            # Very high APR can be high performance despite high risk
            if apr > 200:
                perf_class = 3  # high performance regardless of risk (high reward)
            elif apr > 50 and risk < 0.7:
                perf_class = 3  # high performance (good balance of risk/reward)
            elif apr > 20 and risk < 0.5:
                perf_class = 3  # high performance (good risk/reward)
            elif apr > 10 or (apr > 5 and risk < 0.3):
                perf_class = 2  # medium performance
            else:
                perf_class = 1  # low performance
                
            cursor.execute("""
            INSERT INTO predictions (
                pool_id, predicted_apr, risk_score, performance_class, prediction_timestamp
            ) VALUES (%s, %s, %s, %s, NOW())
            """, (pool_id, apr, risk, perf_class))
        
        conn.commit()
        
        # Generate historical data for all pools
        logger.info("Inserting historical predictions")
        for pool_id, name, _, _, _ in pools:
            # Set base APR according to pool category
            if pool_id == "pool1":
                base_apr = random.uniform(5.0, 25.0)
                max_fluctuation = 5.0
            elif pool_id == "pool2":
                base_apr = random.uniform(15.0, 80.0)
                max_fluctuation = 15.0
            elif pool_id == "pool3":
                base_apr = random.uniform(100.0, 550.0)
                max_fluctuation = 100.0
            elif pool_id == "pool4":
                base_apr = random.uniform(400.0, 999.0)
                max_fluctuation = 150.0
            else:  # pool5
                base_apr = random.uniform(5.0, 35.0)
                max_fluctuation = 8.0
            
            # Historical data for last 14 days
            for day in range(14, 0, -1):
                # Vary APR over time with trend - larger fluctuations for higher APR pools
                apr = base_apr + (day * max_fluctuation/10 * random.choice([-1, 1])) + random.uniform(-max_fluctuation, max_fluctuation)
                # No max capping - allow APR to be as high as market conditions dictate
                apr = max(3.0, apr)  # Only enforce minimum APR
                
                # Risk follows APR - higher APR means higher risk
                if apr > 100:
                    risk = random.uniform(0.7, 0.95)
                elif apr > 50:
                    risk = random.uniform(0.5, 0.8)
                else:
                    risk = random.uniform(0.1, 0.5)
                
                # Performance class - based on both APR and risk
                # Very high APR can be high performance despite high risk
                if apr > 200:
                    perf_class = 3  # high performance regardless of risk (high reward)
                elif apr > 50 and risk < 0.7:
                    perf_class = 3  # high performance (good balance of risk/reward)
                elif apr > 20 and risk < 0.5:
                    perf_class = 3  # high performance (good risk/reward)
                elif apr > 10 or (apr > 5 and risk < 0.3):
                    perf_class = 2  # medium performance
                else:
                    perf_class = 1  # low performance
                
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