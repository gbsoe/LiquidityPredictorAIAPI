"""
Generate test data for the database
"""

import os
import sys
import logging
import psycopg2
import json
import random

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.db_operations import DBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_test_pool_data():
    """Create test pool data in the database"""
    try:
        # Load test pools from JSON file
        with open('60sec_test_all_pools.json', 'r') as f:
            pools_data = json.load(f)
        
        # Initialize database connection
        db = DBManager()
        
        # If database is not available, exit
        if db.use_mock:
            logger.error("Cannot insert test data: using mock database")
            return False
            
        # Connect to the database
        conn = psycopg2.connect(**db.db_params)
        cursor = conn.cursor()
        
        # Insert pools
        pool_count = 0
        
        try:
            # First check if tables exist
            cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables WHERE table_name = 'pools'
            )
            """)
            
            tables_exist = cursor.fetchone()[0]
            
            if not tables_exist:
                logger.error("Tables do not exist. Run init_prediction_tables.py first.")
                return False
                
            # Clean existing data to ensure a fresh start
            try:
                logger.info("Cleaning existing test data...")
                cursor.execute("DELETE FROM predictions")
                cursor.execute("DELETE FROM pools")
                conn.commit()
                logger.info("Cleaned existing data from database")
            except Exception as e:
                logger.warning(f"Error cleaning existing data: {e}")
                conn.rollback()
                
            # Insert pools
            for pool in pools_data:
                # Skip pools with missing required fields
                if not all(k in pool for k in ['id', 'name', 'token_a', 'token_b']):
                    continue
                    
                # Map fields to database columns
                pool_id = pool.get('id', '')
                name = pool.get('name', f"{pool.get('token_a', 'Unknown')}/{pool.get('token_b', 'Unknown')}")
                dex = pool.get('dex', 'Unknown')
                token1 = pool.get('token_a', '')
                token2 = pool.get('token_b', '')
                token1_address = pool.get('token_a_address', '')
                token2_address = pool.get('token_b_address', '')
                category = pool.get('category', 'Unknown')
                
                # Insert pool
                try:
                    cursor.execute("""
                    INSERT INTO pools (
                        pool_id, name, dex, token1, token2, 
                        token1_address, token2_address, category
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (pool_id) DO NOTHING
                    """, (
                        pool_id, name, dex, token1, token2, 
                        token1_address, token2_address, category
                    ))
                    pool_count += 1
                except Exception as e:
                    logger.error(f"Error inserting pool {name}: {e}")
                    
            # Verify pool count    
            cursor.execute("SELECT COUNT(*) FROM pools")
            actual_pool_count = cursor.fetchone()[0]
            
            # If no pools were inserted, likely an issue with the query
            if actual_pool_count == 0:
                logger.error("No pools were inserted. Attempting alternative approach...")
                
                # Try a simpler version with hardcoded pools for testing
                for i, pool in enumerate(pools_data[:15]):
                    pool_id = pool.get('id', '')
                    name = pool.get('name', f"Test Pool {i}")
                    
                    try:
                        cursor.execute("""
                        INSERT INTO pools (pool_id, name, dex, token1, token2) 
                        VALUES (%s, %s, 'Test DEX', 'TokenA', 'TokenB')
                        ON CONFLICT (pool_id) DO NOTHING
                        """, (pool_id, name))
                    except Exception as e:
                        logger.error(f"Error in alternative insert for {name}: {e}")
                
                cursor.execute("SELECT COUNT(*) FROM pools")
                actual_pool_count = cursor.fetchone()[0]
            
            # Commit changes
            conn.commit()
            
            logger.info(f"Inserted {actual_pool_count} pools into database")
                
            # Generate realistic predictions for these pools
            for pool in pools_data[:15]:  # Just generate for the first 15 pools
                pool_id = pool.get('id', '')
                name = pool.get('name', 'Unknown Pool')
                
                # Generate APR (between 3% and 35%)
                predicted_apr = random.uniform(3.0, 35.0)
                
                # Determine performance class
                if predicted_apr > 20:
                    performance_class = 3  # high
                elif predicted_apr > 10:
                    performance_class = 2  # medium
                else:
                    performance_class = 1  # low
                
                # Risk score (0-1, lower is better)
                risk_score = min(0.9, predicted_apr / 50.0 + random.uniform(-0.1, 0.1))
                risk_score = max(0.1, risk_score)
                
                # Insert prediction
                try:
                    cursor.execute("""
                    INSERT INTO predictions (
                        pool_id, predicted_apr, risk_score, performance_class, prediction_timestamp
                    ) VALUES (%s, %s, %s, %s, NOW())
                    """, (
                        pool_id, predicted_apr, risk_score, performance_class
                    ))
                except Exception as e:
                    logger.error(f"Error inserting prediction for pool {name}: {e}")
            
            # Commit prediction changes
            conn.commit()
            logger.info(f"Generated predictions for the top 15 pools")
            
            # Generate historical data for the top 3 pools
            for pool in pools_data[:3]:
                pool_id = pool.get('id', '')
                name = pool.get('name', 'Unknown Pool')
                
                # Base APR (starting point)
                base_apr = random.uniform(10.0, 25.0)
                
                # Generate 14 days of data
                for day in range(14, 0, -1):
                    # Vary APR slightly day by day with a trend
                    historical_apr = base_apr + (day * 0.3 * random.choice([-1, 1])) + random.uniform(-2.0, 2.0)
                    historical_apr = max(3.0, min(35.0, historical_apr))
                    
                    # Performance class
                    if historical_apr > 20:
                        perf_class = 3  # high
                    elif historical_apr > 10:
                        perf_class = 2  # medium
                    else:
                        perf_class = 1  # low
                    
                    # Risk score
                    risk = min(0.9, historical_apr / 50.0 + random.uniform(-0.1, 0.1))
                    risk = max(0.1, risk)
                    
                    # Insert with timestamp offset
                    try:
                        cursor.execute("""
                        INSERT INTO predictions (
                            pool_id, predicted_apr, risk_score, performance_class, 
                            prediction_timestamp
                        ) VALUES (%s, %s, %s, %s, NOW() - INTERVAL %s DAY)
                        """, (
                            pool_id, historical_apr, risk, perf_class, day
                        ))
                    except Exception as e:
                        logger.error(f"Error inserting historical data for pool {name}: {e}")
            
            # Commit historical data changes
            conn.commit()
            logger.info(f"Generated historical prediction data for 3 pools")
            
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating test data: {e}")
            conn.rollback()
            return False
            
    except Exception as e:
        logger.error(f"Error in create_test_pool_data: {e}")
        return False

if __name__ == "__main__":
    logger.info("Generating test data for database")
    
    success = create_test_pool_data()
    
    if success:
        logger.info("Test data generated successfully")
    else:
        logger.error("Failed to generate test data")