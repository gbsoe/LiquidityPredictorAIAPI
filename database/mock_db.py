import os
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
import random

class MockDBManager:
    """
    A mock database manager that reads data from extracted_pools.json
    to enable the pages to work without a real database connection.
    """
    
    def __init__(self):
        """Initialize the mock DB manager and load data from JSON file"""
        self.logger = logging.getLogger("mock_db")
        self.data_file = "extracted_pools.json"
        self.pools = []
        self.load_data()
    
    def load_data(self):
        """Load data from the JSON file"""
        try:
            if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
                with open(self.data_file, 'r') as f:
                    self.pools = json.load(f)
                self.logger.info(f"Loaded {len(self.pools)} pools from {self.data_file}")
            else:
                self.logger.warning(f"Data file {self.data_file} not found or empty")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
    
    def get_pool_list(self):
        """Return a DataFrame with the list of available pools"""
        if not self.pools:
            return pd.DataFrame()
        
        pool_list = []
        for pool in self.pools:
            pool_list.append({
                'pool_id': pool.get('id', ''),
                'name': pool.get('name', ''),
                'dex': pool.get('dex', ''),
                'token1': pool.get('token1_symbol', ''),
                'token2': pool.get('token2_symbol', ''),
                'tvl': pool.get('liquidity', 0),
                'apr': pool.get('apr', 0)
            })
        
        return pd.DataFrame(pool_list)
    
    def get_pool_details(self, pool_id):
        """Return details for a specific pool"""
        for pool in self.pools:
            if pool.get('id') == pool_id:
                return {
                    'pool_id': pool.get('id', ''),
                    'name': pool.get('name', ''),
                    'dex': pool.get('dex', ''),
                    'token1': pool.get('token1_symbol', ''),
                    'token2': pool.get('token2_symbol', ''),
                    'liquidity': pool.get('liquidity', 0),
                    'volume_24h': pool.get('volume_24h', 0),
                    'apr': pool.get('apr', 0),
                    'fee': pool.get('fee', 0),
                    'category': pool.get('category', ''),
                    'timestamp': datetime.now().isoformat()
                }
        return None
    
    def get_pool_metrics(self, pool_id, days=7):
        """
        Generate simulated historical metrics for a pool
        since we don't have real historical data
        """
        for pool in self.pools:
            if pool.get('id') == pool_id:
                # Create a mock time series with the specified number of days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                dates = [start_date + timedelta(days=i) for i in range(days + 1)]
                
                # Base values from the pool
                base_liquidity = pool.get('liquidity', 1000000)
                base_volume = pool.get('volume_24h', 100000)
                base_apr = pool.get('apr', 10)
                
                # Generate time series with some random variations
                liquidity_series = [
                    base_liquidity * (1 + random.uniform(-0.05, 0.05)) 
                    for _ in range(len(dates))
                ]
                volume_series = [
                    base_volume * (1 + random.uniform(-0.1, 0.1)) 
                    for _ in range(len(dates))
                ]
                apr_series = [
                    base_apr * (1 + random.uniform(-0.07, 0.07)) 
                    for _ in range(len(dates))
                ]
                
                # Create DataFrame
                metrics_df = pd.DataFrame({
                    'timestamp': dates,
                    'liquidity': liquidity_series,
                    'volume': volume_series,
                    'apr': apr_series,
                    'pool_id': pool_id
                })
                
                return metrics_df
                
        return pd.DataFrame()
    
    def get_token_prices(self, token_symbols, days=7):
        """Generate simulated token price data"""
        if not token_symbols:
            return pd.DataFrame()
        
        # Create a mock time series with the specified number of days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = [start_date + timedelta(days=i) for i in range(days + 1)]
        
        # Price data for all tokens
        price_data = []
        
        for token in token_symbols:
            # Base price (random but consistent for each token)
            base_price = sum(ord(c) for c in token) % 100 + 1  # Simple hash for consistency
            
            # Generate price series with random walk
            price_series = [base_price]
            for i in range(1, len(dates)):
                # Random walk with 5% max change
                change = price_series[-1] * random.uniform(-0.05, 0.05)
                price_series.append(price_series[-1] + change)
            
            # Add to price data
            for i, date in enumerate(dates):
                price_data.append({
                    'timestamp': date,
                    'token_symbol': token,
                    'price_usd': price_series[i]
                })
        
        return pd.DataFrame(price_data)
    
    def get_top_predictions(self, category="apr", limit=10, ascending=False):
        """Return top predictions based on the specified category"""
        if not self.pools:
            return pd.DataFrame()
        
        predictions = []
        
        for pool in self.pools:
            # Use pool metrics as predictions
            if category == "apr":
                prediction_value = pool.get('apr', 0)
                # Add some variability for predicted values
                predicted_apr = pool.get('apr', 0) * (1 + random.uniform(-0.1, 0.2))
            elif category == "risk":
                # Generate a risk score (lower is better)
                # Base this on volatility metrics in the pool
                vol_factor = abs(pool.get('apr_change_7d', 0)) / 10 if 'apr_change_7d' in pool else random.uniform(0.1, 0.9)
                prediction_value = min(1.0, max(0.0, vol_factor))
                predicted_apr = pool.get('apr', 0)
            else:  # performance
                # Performance class: 1 (high), 2 (medium), 3 (low)
                # Base on APR (higher APR = better performance)
                apr = pool.get('apr', 0)
                if apr > 20:
                    prediction_value = 1
                elif apr > 10:
                    prediction_value = 2
                else:
                    prediction_value = 3
                predicted_apr = apr
            
            predictions.append({
                'pool_id': pool.get('id', ''),
                'pool_name': pool.get('name', ''),
                'dex': pool.get('dex', ''),
                'token1': pool.get('token1_symbol', ''),
                'token2': pool.get('token2_symbol', ''),
                'tvl': pool.get('liquidity', 0),
                'current_apr': pool.get('apr', 0),
                'predicted_apr': predicted_apr,
                'risk_score': random.uniform(0.1, 0.9) if category != "risk" else prediction_value,
                'performance_class': random.choice([1, 2, 3]) if category != "performance" else prediction_value,
                'prediction_timestamp': datetime.now().isoformat(),
                'prediction_value': prediction_value
            })
        
        # Convert to DataFrame and sort
        predictions_df = pd.DataFrame(predictions)
        
        if not predictions_df.empty:
            # Sort based on the prediction category
            if category == "apr":
                sort_column = "predicted_apr"
            elif category == "risk":
                sort_column = "risk_score"
            else:  # performance
                sort_column = "performance_class"
            
            predictions_df = predictions_df.sort_values(
                sort_column, ascending=ascending
            ).head(limit)
        
        return predictions_df
    
    def get_pool_predictions(self, pool_id):
        """Return prediction history for a specific pool"""
        for pool in self.pools:
            if pool.get('id') == pool_id:
                # Generate 10 days of prediction history
                end_date = datetime.now()
                start_date = end_date - timedelta(days=10)
                
                dates = [start_date + timedelta(days=i) for i in range(11)]
                
                # Base values from the pool
                base_apr = pool.get('apr', 10)
                
                # Generate time series with some random variations
                predicted_apr_series = [
                    base_apr * (1 + random.uniform(-0.1, 0.2)) 
                    for _ in range(len(dates))
                ]
                
                # Performance class (1=high, 2=medium, 3=low)
                # Base on predicted APR
                perf_class_series = []
                for apr in predicted_apr_series:
                    if apr > 20:
                        perf_class_series.append(1)
                    elif apr > 10:
                        perf_class_series.append(2)
                    else:
                        perf_class_series.append(3)
                
                # Risk score (0-1, lower is better)
                risk_score_series = [
                    random.uniform(0.1, 0.9) 
                    for _ in range(len(dates))
                ]
                
                # Create DataFrame
                predictions_df = pd.DataFrame({
                    'prediction_timestamp': dates,
                    'predicted_apr': predicted_apr_series,
                    'performance_class': perf_class_series,
                    'risk_score': risk_score_series,
                    'pool_id': pool_id
                })
                
                return predictions_df
                
        return pd.DataFrame()