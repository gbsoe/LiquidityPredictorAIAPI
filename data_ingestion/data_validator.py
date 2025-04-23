import os
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_validator.log')
    ]
)

logger = logging.getLogger('data_validator')

# Constants
DATABASE_PATH = os.getenv('DATABASE_PATH', '../database/liquidity_pools.db')

class DataValidator:
    """
    Service for validating data quality and integrity
    """
    
    def __init__(self):
        """Initialize the data validator"""
        self.database_path = DATABASE_PATH
    
    def connect_to_db(self):
        """Establish a connection to the database"""
        try:
            conn = sqlite3.connect(self.database_path)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return None
    
    def validate_table_structure(self):
        """
        Validate that all required tables exist with the correct structure
        """
        required_tables = [
            'pool_data',
            'pool_metrics',
            'token_prices',
            'token_price_history',
            'blockchain_stats'
        ]
        
        conn = self.connect_to_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Get list of tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        # Check that all required tables exist
        missing_tables = set(required_tables) - set(existing_tables)
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            conn.close()
            return False
        
        logger.info("All required tables exist")
        conn.close()
        return True
    
    def validate_pool_data(self):
        """
        Validate pool data for consistency and completeness
        """
        conn = self.connect_to_db()
        if not conn:
            return False
        
        try:
            # Load data into pandas for analysis
            pool_data_df = pd.read_sql("SELECT * FROM pool_data", conn)
            
            # Check for null values in critical columns
            null_counts = pool_data_df[['pool_id', 'name', 'liquidity', 'volume_24h', 'apr']].isnull().sum()
            if null_counts.sum() > 0:
                logger.warning(f"Null values found in pool_data: {null_counts}")
            
            # Check for duplicate pool IDs
            duplicate_ids = pool_data_df['pool_id'].duplicated().sum()
            if duplicate_ids > 0:
                logger.warning(f"Found {duplicate_ids} duplicate pool IDs")
            
            # Check for unreasonable values
            if (pool_data_df['liquidity'] < 0).any():
                logger.warning("Negative liquidity values found")
            
            if (pool_data_df['volume_24h'] < 0).any():
                logger.warning("Negative volume values found")
            
            if (pool_data_df['apr'] < 0).any() or (pool_data_df['apr'] > 1000).any():
                logger.warning("Suspicious APR values found (negative or >1000%)")
            
            # Check data freshness
            now = datetime.now()
            pool_data_df['timestamp'] = pd.to_datetime(pool_data_df['timestamp'])
            stale_data = pool_data_df[pool_data_df['timestamp'] < (now - timedelta(days=3))]
            
            if len(stale_data) > 0:
                logger.warning(f"Found {len(stale_data)} pools with data older than 3 days")
            
            logger.info(f"Pool data validation complete: {len(pool_data_df)} records checked")
            return True
            
        except Exception as e:
            logger.error(f"Error validating pool data: {e}")
            return False
        finally:
            conn.close()
    
    def validate_metrics_data(self):
        """
        Validate metrics data for consistency and completeness
        """
        conn = self.connect_to_db()
        if not conn:
            return False
        
        try:
            # Load data into pandas for analysis
            metrics_df = pd.read_sql("SELECT * FROM pool_metrics ORDER BY timestamp DESC LIMIT 10000", conn)
            
            if len(metrics_df) == 0:
                logger.warning("No metrics data found")
                return False
            
            # Convert timestamp to datetime
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
            
            # Check for missing timestamps (gaps in data collection)
            # Group by pool_id and check time differences
            for pool_id, group in metrics_df.groupby('pool_id'):
                if len(group) < 2:
                    continue
                    
                sorted_group = group.sort_values('timestamp')
                time_diffs = sorted_group['timestamp'].diff()
                
                # Check for gaps larger than expected (e.g., 2 hours)
                large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]
                if len(large_gaps) > 0:
                    logger.warning(f"Pool {pool_id} has {len(large_gaps)} large gaps in data collection")
            
            # Check for outliers in metrics using IQR method
            for metric in ['liquidity', 'volume', 'apr']:
                q1 = metrics_df[metric].quantile(0.25)
                q3 = metrics_df[metric].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = metrics_df[(metrics_df[metric] < lower_bound) | (metrics_df[metric] > upper_bound)]
                if len(outliers) > 0:
                    logger.warning(f"Found {len(outliers)} outliers in {metric}")
            
            logger.info(f"Metrics data validation complete: {len(metrics_df)} records checked")
            return True
            
        except Exception as e:
            logger.error(f"Error validating metrics data: {e}")
            return False
        finally:
            conn.close()
    
    def validate_price_data(self):
        """
        Validate token price data
        """
        conn = self.connect_to_db()
        if not conn:
            return False
        
        try:
            # Load price data
            price_df = pd.read_sql("SELECT * FROM token_prices ORDER BY timestamp DESC", conn)
            
            if len(price_df) == 0:
                logger.warning("No price data found")
                return False
            
            # Convert timestamp to datetime
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            
            # Check for tokens with missing recent prices
            now = datetime.now()
            recent_cutoff = now - timedelta(days=1)
            
            tokens = price_df['token_symbol'].unique()
            tokens_with_recent_data = price_df[price_df['timestamp'] > recent_cutoff]['token_symbol'].unique()
            
            missing_recent_data = set(tokens) - set(tokens_with_recent_data)
            if missing_recent_data:
                logger.warning(f"{len(missing_recent_data)} tokens missing recent price data: {missing_recent_data}")
            
            # Check for extreme price changes (potential data errors)
            for token in tokens:
                token_prices = price_df[price_df['token_symbol'] == token].sort_values('timestamp')
                if len(token_prices) < 2:
                    continue
                
                price_changes = token_prices['price_usd'].pct_change()
                extreme_changes = price_changes[abs(price_changes) > 0.5]  # 50% change
                
                if len(extreme_changes) > 0:
                    logger.warning(f"Token {token} has {len(extreme_changes)} extreme price changes")
            
            logger.info(f"Price data validation complete: {len(price_df)} records checked")
            return True
            
        except Exception as e:
            logger.error(f"Error validating price data: {e}")
            return False
        finally:
            conn.close()
    
    def run_full_validation(self):
        """
        Run all validation checks
        """
        logger.info("Starting full data validation")
        
        # Check database structure
        structure_valid = self.validate_table_structure()
        if not structure_valid:
            logger.error("Database structure validation failed")
            return False
        
        # Validate pool data
        pool_data_valid = self.validate_pool_data()
        
        # Validate metrics data
        metrics_valid = self.validate_metrics_data()
        
        # Validate price data
        price_data_valid = self.validate_price_data()
        
        # Overall validation result
        validation_successful = all([
            structure_valid,
            pool_data_valid,
            metrics_valid,
            price_data_valid
        ])
        
        if validation_successful:
            logger.info("Full data validation successful")
        else:
            logger.warning("Data validation found issues that need attention")
        
        return validation_successful
    
    def clean_data(self):
        """
        Clean inconsistent or corrupted data
        """
        conn = self.connect_to_db()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Remove duplicate metrics entries
            cursor.execute("""
                DELETE FROM pool_metrics 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM pool_metrics 
                    GROUP BY pool_id, timestamp
                )
            """)
            duplicates_removed = cursor.rowcount
            
            # Remove metrics with negative values
            cursor.execute("DELETE FROM pool_metrics WHERE liquidity < 0 OR volume < 0 OR apr < 0")
            negative_removed = cursor.rowcount
            
            # Remove price entries with zero or negative prices
            cursor.execute("DELETE FROM token_prices WHERE price_usd <= 0")
            invalid_prices_removed = cursor.rowcount
            
            conn.commit()
            
            logger.info(f"Data cleaning complete: {duplicates_removed} duplicates, "
                       f"{negative_removed} negative metrics, {invalid_prices_removed} invalid prices removed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

if __name__ == "__main__":
    validator = DataValidator()
    validator.run_full_validation()
    validator.clean_data()
