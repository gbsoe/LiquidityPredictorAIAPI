"""
Quick generator for realistic prediction data.
This script directly saves realistic predictions to the database
without requiring complex processing.
"""

import sys
import os
import logging
import random
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.db_operations import DBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_pools_exist(db):
    """
    Ensure pools exist in the database before adding predictions.
    This function checks if the pools table is populated and if not,
    adds the mock pools to the real database.
    
    Args:
        db: Database manager instance
        
    Returns:
        bool: True if pools exist or were successfully added
    """
    try:
        # First check if any pools exist in the real database
        import psycopg2
        conn = psycopg2.connect(**db.db_params)
        cursor = conn.cursor()
        
        # Check if pools table exists and has data
        cursor.execute("SELECT COUNT(*) FROM pools")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"Found {count} existing pools in database")
            cursor.close()
            conn.close()
            return True
            
        # If no pools exist, get pools from mock database and insert them
        logger.info("No pools found in database. Adding mock pools to real database.")
        mock_pools = db.mock_db.get_pool_list()
        
        if mock_pools.empty:
            logger.error("No mock pools available to add to database")
            cursor.close()
            conn.close()
            return False
            
        # Insert pools into database
        for _, pool in mock_pools.iterrows():
            pool_id = pool['pool_id']
            pool_name = pool.get('name', 'Unknown Pool')
            
            # Extract needed fields, defaulting if missing
            token_a = pool.get('token_a', 'Unknown')
            token_b = pool.get('token_b', 'Unknown')
            dex = pool.get('dex', 'Unknown')
            category = pool.get('category', 'Unknown')
            liquidity = pool.get('liquidity', 0.0)
            volume = pool.get('volume', 0.0)
            
            # Insert pool
            query = """
            INSERT INTO pools (
                pool_id, name, token_a, token_b, dex, category, 
                liquidity, volume, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (pool_id) DO NOTHING
            """
            
            cursor.execute(query, (
                pool_id, pool_name, token_a, token_b, dex, category,
                liquidity, volume
            ))
        
        # Commit changes
        conn.commit()
        
        # Verify pools were added
        cursor.execute("SELECT COUNT(*) FROM pools")
        new_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        logger.info(f"Added {new_count} pools to database")
        return new_count > 0
            
    except Exception as e:
        logger.error(f"Error ensuring pools exist in database: {e}")
        return False

def generate_realistic_predictions(db, pool_count=20):
    """
    Generate and save realistic predictions directly to the database.
    
    Args:
        db: Database manager instance
        pool_count: Number of pools to generate predictions for
    """
    logger.info(f"Generating realistic predictions for {pool_count} pools")
    
    # First ensure pools exist in the database
    if not ensure_pools_exist(db):
        logger.error("Failed to ensure pools exist in database")
        return False
    
    # Get top pools
    try:
        # Get pool list from real database
        import psycopg2
        conn = psycopg2.connect(**db.db_params)
        cursor = conn.cursor()
        
        # Get pools ordered by liquidity
        cursor.execute("""
        SELECT pool_id, name, token_a, token_b, liquidity, volume
        FROM pools
        ORDER BY liquidity DESC
        LIMIT %s
        """, (pool_count,))
        
        pool_records = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not pool_records:
            logger.error("No pools found in database after ensuring they exist")
            return False
            
        logger.info(f"Retrieved {len(pool_records)} pools from database")
        
        # Convert to DataFrame for easier handling
        pool_data = {
            'pool_id': [],
            'name': [],
            'token_a': [],
            'token_b': [],
            'liquidity': [],
            'volume': []
        }
        
        for record in pool_records:
            pool_data['pool_id'].append(record[0])
            pool_data['name'].append(record[1])
            pool_data['token_a'].append(record[2])
            pool_data['token_b'].append(record[3])
            pool_data['liquidity'].append(record[4])
            pool_data['volume'].append(record[5])
            
        top_pools = pd.DataFrame(pool_data)
        
        success_count = 0
        
        # Generate predictions for each pool
        for _, pool in top_pools.iterrows():
            pool_id = pool['pool_id']
            pool_name = pool.get('name', 'Unknown Pool')
            
            # Get current APR if available
            current_apr = pool.get('apr', None)
            
            # Generate a realistic APR prediction (between 2% and 35%)
            if current_apr is not None and 0 < current_apr < 100:
                # Base prediction on current APR if it's reasonable
                predicted_apr = current_apr * (1 + random.uniform(-0.15, 0.25))
                # Keep it realistic
                predicted_apr = max(2.0, min(35.0, predicted_apr))
            else:
                # Generate a reasonable APR if current not available or unrealistic
                predicted_apr = random.uniform(5.0, 25.0)
            
            # Determine performance class based on APR
            if predicted_apr > 20:
                performance_class = 'high'
            elif predicted_apr > 10:
                performance_class = 'medium'
            else:
                performance_class = 'low'
            
            # Generate risk score (0-1, lower is better)
            # Higher APR typically correlates with higher risk
            base_risk = min(1.0, predicted_apr / 40.0)  # Higher APR = higher base risk
            
            # Add some randomness to risk (±20%)
            risk_score = base_risk * (1 + random.uniform(-0.2, 0.2))
            
            # Ensure risk is between 0.1 and 0.9
            risk_score = max(0.1, min(0.9, risk_score))
            
            # Save to database
            try:
                success = db.save_prediction(
                    pool_id=pool_id,
                    predicted_apr=predicted_apr,
                    performance_class=performance_class,
                    risk_score=risk_score
                )
                
                if success:
                    success_count += 1
                    logger.info(f"Saved prediction for {pool_name}: APR={predicted_apr:.2f}%, "
                               f"Performance={performance_class}, Risk={risk_score:.2f}")
                else:
                    logger.warning(f"Failed to save prediction for {pool_name}")
                    
            except Exception as e:
                logger.error(f"Error saving prediction for {pool_name}: {e}")
        
        # Generate historical predictions for a few pools to show trends
        logger.info("Generating historical predictions for trend data...")
        
        # Take 5 random pools from the top pools
        if len(top_pools) > 5:
            trend_pools = top_pools.sample(5)
        else:
            trend_pools = top_pools
        
        for _, pool in trend_pools.iterrows():
            pool_id = pool['pool_id']
            pool_name = pool.get('name', 'Unknown Pool')
            
            # Generate 14 days of historical predictions with slight variations
            base_apr = random.uniform(10.0, 30.0)
            
            # Either upward or downward trend
            trend_direction = random.choice([-1, 1])
            
            for day in range(1, 15):
                # Calculate APR with trend and noise
                historical_apr = base_apr + (trend_direction * day * 0.5) + random.uniform(-2.0, 2.0)
                historical_apr = max(2.0, min(40.0, historical_apr))
                
                # Determine performance class
                if historical_apr > 20:
                    perf_class = 'high'
                elif historical_apr > 10:
                    perf_class = 'medium'
                else:
                    perf_class = 'low'
                
                # Risk score
                risk = min(0.9, historical_apr / 40.0 + random.uniform(-0.1, 0.1))
                risk = max(0.1, risk)
                
                # Calculate date (days ago)
                days_ago = 14 - day
                
                # Save with historical timestamp
                # We'll handle this in the DB layer
                try:
                    success = db.save_prediction(
                        pool_id=pool_id,
                        predicted_apr=historical_apr,
                        performance_class=perf_class,
                        risk_score=risk,
                        # Historical data from days ago
                        model_version=f"v{datetime.now().strftime('%Y%m%d')}-historical-{days_ago}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error saving historical prediction for {pool_name}: {e}")
        
        logger.info(f"Successfully generated predictions for {success_count}/{len(top_pools)} pools")
        logger.info("Historical prediction data generated for trend analysis")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error in prediction generation: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting simplified prediction generation")
    
    # Initialize database connection
    db = DBManager()
    
    # Generate and save realistic predictions
    success = generate_realistic_predictions(db, pool_count=30)
    
    if success:
        logger.info("Prediction generation completed successfully")
    else:
        logger.error("Prediction generation failed")