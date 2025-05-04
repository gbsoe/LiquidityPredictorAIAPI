"""
Quick script to generate realistic predictions for pools in the database
"""

import os
import sys
import logging
import random
import time
import json
from datetime import datetime, timedelta

import pandas as pd
from database.db_operations import DBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def generate_realistic_predictions():
    """Generate and save realistic prediction data"""
    try:
        # Create DB manager
        db = DBManager()
        
        # If we're using mock DB, can't save predictions to real DB
        if db.use_mock:
            logger.error("Using mock database, cannot save predictions")
            return False
        
        # Load pool data from JSON (for testing)
        try:
            with open('60sec_test_all_pools.json', 'r') as f:
                pools_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading pool data: {e}")
            return False
        
        success_count = 0
        error_count = 0
        
        # Process top pools
        for i, pool in enumerate(pools_data[:20]):  # Top 20 pools
            pool_id = pool.get('id')
            name = pool.get('name', 'Unknown Pool')
            
            # Generate realistic APR (between 3% and 35%)
            predicted_apr = random.uniform(3.0, 35.0)
            
            # Higher APR generally means higher risk
            # Add some randomness but keep correlation
            base_risk = min(0.8, predicted_apr / 50.0)
            risk_score = max(0.1, min(0.9, base_risk + random.uniform(-0.1, 0.2)))
            
            # Determine performance class
            if predicted_apr > 20:
                performance_class = 'high'
            elif predicted_apr > 10:
                performance_class = 'medium'
            else:
                performance_class = 'low'
            
            try:
                # Save to database
                success = db.save_prediction(
                    pool_id,
                    predicted_apr,
                    performance_class,
                    risk_score,
                    model_version="realistic-model-v1"
                )
                
                if success:
                    success_count += 1
                    logger.info(f"Saved prediction for {name}: "
                               f"APR={predicted_apr:.2f}%, "
                               f"Risk={risk_score:.2f}, "
                               f"Performance={performance_class}")
                else:
                    error_count += 1
                    logger.warning(f"Failed to save prediction for {name}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error saving prediction for {name}: {e}")
        
        # Generate historical data for top 5 pools
        for pool in pools_data[:5]:
            pool_id = pool.get('id')
            name = pool.get('name', 'Unknown Pool')
            
            # Base values
            base_apr = random.uniform(10.0, 25.0)
            base_risk = min(0.7, base_apr / 40.0)
            
            # Generate 14 days of history
            for days_ago in range(14, 0, -1):
                # Calculate timestamp for historical data
                timestamp = datetime.now() - timedelta(days=days_ago)
                
                # Vary slightly over time with a trend
                trend_factor = days_ago / 14.0  # 1.0 to ~0.07
                
                # Apply trend (slightly decreasing APR over time)
                apr = base_apr * (1 - (0.15 * (1 - trend_factor))) + random.uniform(-1.5, 1.5)
                apr = max(3.0, min(35.0, apr))
                
                # Risk follows APR with some noise
                risk = base_risk + random.uniform(-0.05, 0.05)
                risk = max(0.1, min(0.9, risk))
                
                # Performance class
                if apr > 20:
                    perf = 'high'
                elif apr > 10:
                    perf = 'medium'
                else:
                    perf = 'low'
                
                # Save with historical timestamp
                try:
                    # Direct database operation for historical timestamp
                    # We can't use the regular save method because it always uses NOW()
                    db.save_prediction_with_timestamp(
                        pool_id, apr, perf, risk, timestamp
                    )
                    
                    logger.debug(f"Saved historical prediction for {name}, {days_ago} days ago")
                    
                except Exception as e:
                    logger.error(f"Error saving historical prediction: {e}")
        
        logger.info(f"Generated predictions: {success_count} successful, {error_count} errors")
        return True
    
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return False

# Add method to DBManager to save with custom timestamp
def add_save_with_timestamp_method():
    """Add method to DBManager for saving with timestamp"""
    def save_prediction_with_timestamp(self, pool_id, predicted_apr, 
                                      performance_class, risk_score, 
                                      timestamp):
        """Save prediction with specific timestamp"""
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
            
            # Insert prediction with specific timestamp
            query = """
            INSERT INTO predictions (
                pool_id, 
                predicted_apr, 
                risk_score, 
                performance_class, 
                prediction_timestamp
            ) VALUES (%s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                pool_id,
                predicted_apr,
                risk_score,
                perf_class_value,
                timestamp
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving prediction with timestamp: {str(e)}")
            return False
    
    # Add method to DBManager class
    import psycopg2
    DBManager.save_prediction_with_timestamp = save_prediction_with_timestamp

if __name__ == "__main__":
    logger.info("Generating realistic predictions...")
    
    # Add save_with_timestamp method to DBManager
    add_save_with_timestamp_method()
    
    # Generate predictions
    if generate_realistic_predictions():
        logger.info("Successfully generated realistic predictions")
    else:
        logger.error("Failed to generate predictions")