import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sqlite3
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_operations import DBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('feature_engineering')

class FeatureEngineer:
    """
    Class for creating features from raw data for machine learning models
    """
    
    def __init__(self):
        """Initialize the feature engineer"""
        self.db = DBManager()
    
    def get_pool_with_metrics(self, pool_id, days=30):
        """
        Get pool data combined with its historical metrics
        """
        try:
            # Get basic pool data
            pool_df = self.db.get_pool_by_id(pool_id)
            if pool_df.empty:
                logger.warning(f"No data found for pool {pool_id}")
                return None
            
            # Get historical metrics
            metrics_df = self.db.get_pool_metrics(pool_id, days)
            if metrics_df.empty:
                logger.warning(f"No metrics found for pool {pool_id}")
                return None
            
            # Convert timestamps to datetime
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
            
            return metrics_df
            
        except Exception as e:
            logger.error(f"Error getting pool data: {e}")
            return None
    
    def create_time_features(self, df):
        """
        Create time-based features from timestamp
        """
        if df is None or df.empty:
            return None
        
        try:
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract time components
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
            return df
    
    def create_rolling_features(self, df, windows=[3, 6, 12, 24]):
        """
        Create rolling window features for time series data
        """
        if df is None or df.empty:
            return None
        
        try:
            # Sort by timestamp to ensure correct rolling calculations
            df = df.sort_values('timestamp')
            
            # Create rolling features for key metrics
            for window in windows:
                # Rolling mean
                df[f'liquidity_rolling_mean_{window}h'] = df['liquidity'].rolling(window=window).mean()
                df[f'volume_rolling_mean_{window}h'] = df['volume'].rolling(window=window).mean()
                df[f'apr_rolling_mean_{window}h'] = df['apr'].rolling(window=window).mean()
                
                # Rolling standard deviation (volatility)
                df[f'liquidity_rolling_std_{window}h'] = df['liquidity'].rolling(window=window).std()
                df[f'volume_rolling_std_{window}h'] = df['volume'].rolling(window=window).std()
                df[f'apr_rolling_std_{window}h'] = df['apr'].rolling(window=window).std()
                
                # Rolling min/max
                df[f'liquidity_rolling_min_{window}h'] = df['liquidity'].rolling(window=window).min()
                df[f'liquidity_rolling_max_{window}h'] = df['liquidity'].rolling(window=window).max()
                df[f'apr_rolling_min_{window}h'] = df['apr'].rolling(window=window).min()
                df[f'apr_rolling_max_{window}h'] = df['apr'].rolling(window=window).max()
            
            # Calculate percentage changes
            df['liquidity_pct_change'] = df['liquidity'].pct_change()
            df['volume_pct_change'] = df['volume'].pct_change()
            df['apr_pct_change'] = df['apr'].pct_change()
            
            # Fill NaN values resulting from rolling calculations
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {e}")
            return df
    
    def add_blockchain_features(self, df):
        """
        Add blockchain statistics as features
        """
        if df is None or df.empty:
            return None
        
        try:
            # Get blockchain stats
            blockchain_df = self.db.get_blockchain_stats()
            
            if blockchain_df.empty:
                logger.warning("No blockchain stats available")
                return df
            
            # Convert timestamps to datetime
            blockchain_df['timestamp'] = pd.to_datetime(blockchain_df['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Prepare blockchain features for merging
            blockchain_features = blockchain_df[['timestamp', 'avg_tps', 'sol_price']]
            
            # Merge blockchain features to pool data based on closest timestamp
            # This is a simplified approach; in production you might want more precise matching
            merged_df = pd.merge_asof(
                df.sort_values('timestamp'),
                blockchain_features.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error adding blockchain features: {e}")
            return df
    
    def add_token_price_features(self, df, pool_name):
        """
        Add token price features for the tokens in the pool
        """
        if df is None or df.empty or not pool_name:
            return None
        
        try:
            # Extract token symbols from pool name
            if '/' not in pool_name:
                logger.warning(f"Unexpected pool name format: {pool_name}")
                return df
                
            tokens = pool_name.split('/')
            token0 = tokens[0].strip()
            token1 = tokens[1].strip()
            
            # Get price history for both tokens
            token0_prices = self.db.get_token_price_history(token0)
            token1_prices = self.db.get_token_price_history(token1)
            
            if token0_prices.empty or token1_prices.empty:
                logger.warning(f"Missing price data for {token0} or {token1}")
                return df
            
            # Convert timestamps
            token0_prices['timestamp'] = pd.to_datetime(token0_prices['timestamp'])
            token1_prices['timestamp'] = pd.to_datetime(token1_prices['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Merge token0 prices
            df = pd.merge_asof(
                df.sort_values('timestamp'),
                token0_prices[['timestamp', 'price_usd']].sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                suffixes=('', f'_{token0}')
            )
            df = df.rename(columns={'price_usd': f'{token0}_price'})
            
            # Merge token1 prices
            df = pd.merge_asof(
                df.sort_values('timestamp'),
                token1_prices[['timestamp', 'price_usd']].sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                suffixes=('', f'_{token1}')
            )
            df = df.rename(columns={'price_usd': f'{token1}_price'})
            
            # Calculate price ratio and changes
            df['price_ratio'] = df[f'{token0}_price'] / df[f'{token1}_price']
            df['price_ratio_pct_change'] = df['price_ratio'].pct_change()
            
            # Calculate rolling features for price ratio
            df['price_ratio_rolling_std_24h'] = df['price_ratio'].rolling(window=24).std()
            df['price_ratio_rolling_mean_24h'] = df['price_ratio'].rolling(window=24).mean()
            
            # Fill NaN values
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding token price features: {e}")
            return df
    
    def prepare_pool_features(self, pool_id, days=30):
        """
        Prepare all features for a given pool
        """
        try:
            # Get pool basic info
            pool_info = self.db.get_pool_by_id(pool_id)
            if pool_info.empty:
                logger.warning(f"No data found for pool {pool_id}")
                return None
            
            # Get metrics data
            metrics_df = self.get_pool_with_metrics(pool_id, days)
            if metrics_df is None or metrics_df.empty:
                logger.warning(f"No metrics found for pool {pool_id}")
                return None
            
            # Apply feature engineering transformations
            df = self.create_time_features(metrics_df)
            df = self.create_rolling_features(df)
            df = self.add_blockchain_features(df)
            
            if not pool_info.empty and 'name' in pool_info.columns:
                pool_name = pool_info.iloc[0]['name']
                df = self.add_token_price_features(df, pool_name)
            
            # Drop rows with NaN values after feature engineering
            df = df.dropna()
            
            logger.info(f"Prepared features for pool {pool_id}: {df.shape[0]} records with {df.shape[1]} features")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features for pool {pool_id}: {e}")
            return None
    
    def prepare_features_for_all_pools(self, top_n=100, days=30):
        """
        Prepare features for multiple pools
        """
        try:
            # Get top pools by liquidity
            top_pools = self.db.get_top_pools_by_liquidity(limit=top_n)
            
            if top_pools.empty:
                logger.warning("No pools found")
                return None
            
            all_features = []
            
            # Process each pool
            for _, pool in top_pools.iterrows():
                pool_id = pool['pool_id']
                try:
                    pool_features = self.prepare_pool_features(pool_id, days)
                    
                    if pool_features is not None and not pool_features.empty:
                        # Add pool identifier
                        pool_features['pool_id'] = pool_id
                        pool_features['pool_name'] = pool['name']
                        
                        all_features.append(pool_features)
                except Exception as e:
                    logger.error(f"Error processing pool {pool_id}: {e}")
            
            if not all_features:
                logger.warning("No features generated for any pool")
                return None
            
            # Combine all pools' features
            combined_df = pd.concat(all_features, ignore_index=True)
            
            logger.info(f"Prepared features for {len(all_features)} pools with {combined_df.shape[1]} features")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error preparing features for all pools: {e}")
            return None
    
    def prepare_target_variables(self, df):
        """
        Prepare target variables for different ML models
        """
        if df is None or df.empty:
            return None
        
        try:
            # Sort by pool and time
            df = df.sort_values(['pool_id', 'timestamp'])
            
            # Create target for APR prediction (next day's APR)
            df['next_apr'] = df.groupby('pool_id')['apr'].shift(-24)  # Assuming hourly data
            
            # Create classification targets based on APR performance
            # Calculate APR changes over different time horizons
            df['apr_change_24h'] = df.groupby('pool_id')['apr'].pct_change(periods=24)
            
            # Create performance classification target
            # Assign categories: 'high', 'medium', 'low' based on percentiles
            df['performance_class'] = pd.qcut(
                df['apr_change_24h'], 
                q=[0, 0.33, 0.67, 1], 
                labels=['low', 'medium', 'high']
            )
            
            # Risk assessment target - simplified version based on price ratio volatility
            # In a real system, a more sophisticated risk model would be used
            if 'price_ratio_rolling_std_24h' in df.columns:
                df['risk_score'] = df['price_ratio_rolling_std_24h'] / df['price_ratio_rolling_mean_24h']
                df['risk_score'] = df['risk_score'].clip(0, 1)  # Normalize to 0-1
            
            # Drop rows where targets are NaN
            df = df.dropna(subset=['next_apr'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing target variables: {e}")
            return df

if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    
    # Prepare features for a specific pool
    pool_id = "example_pool_id"  # Replace with an actual pool ID
    features = engineer.prepare_pool_features(pool_id)
    
    if features is not None:
        print(f"Generated {features.shape[1]} features for pool {pool_id}")
        
    # Prepare features for top pools
    all_features = engineer.prepare_features_for_all_pools(top_n=10)
    
    if all_features is not None:
        print(f"Generated features for multiple pools: {all_features.shape}")
