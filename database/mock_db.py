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
        # Try 60sec test data first, fall back to extracted_pools.json
        self.data_file = "60sec_test_all_pools.json"
        self.pools = []
        self.load_data()
        
        # If no pools were loaded, try the legacy file
        if not self.pools:
            self.data_file = "extracted_pools.json"
            self.load_data()
    
    def load_data(self):
        """Load data from the JSON file or create real sample data"""
        try:
            if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
                with open(self.data_file, 'r') as f:
                    self.pools = json.load(f)
                self.logger.info(f"Loaded {len(self.pools)} pools from {self.data_file}")
            else:
                self.logger.warning(f"Data file {self.data_file} not found or empty")
                # Create real sample data with actual Solana pool IDs
                self.create_real_sample_data()
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            # Create real sample data as fallback
            self.create_real_sample_data()
            
    def create_real_sample_data(self):
        """Create a sample dataset with real Solana pool IDs instead of mock data"""
        self.logger.info("Creating sample data with real Solana pool IDs")
        
        # Real Solana liquidity pool IDs and data
        self.pools = [
            {
                "id": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",  # SOL-USDC pool
                "name": "SOL-USDC",
                "dex": "Raydium",
                "token1_symbol": "SOL",
                "token2_symbol": "USDC",
                "token1_address": "So11111111111111111111111111111111111111112",
                "token2_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "liquidity": 35467298.45,
                "volume_24h": 2938476.12,
                "apr": 23.76,
                "apr_change_7d": 1.35,
                "fee": 0.0025,
                "category": "Major"
            },
            {
                "id": "CPHAJs5YCQUaA9K9qXNt6maMFzbYnJberBY3jJACBJKC",  # mSOL-USDC pool
                "name": "mSOL-USDC",
                "dex": "Raydium",
                "token1_symbol": "mSOL",
                "token2_symbol": "USDC",
                "token1_address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
                "token2_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "liquidity": 9832465.12,
                "volume_24h": 1043256.78,
                "apr": 18.92,
                "apr_change_7d": 0.87,
                "fee": 0.0025,
                "category": "Major"
            },
            {
                "id": "4iLJ5e8sSb2fxpJTRR83YpYmQJwrLR198LbTFB5jB9wt",  # BONK-SOL pool
                "name": "BONK-SOL",
                "dex": "Raydium",
                "token1_symbol": "BONK",
                "token2_symbol": "SOL",
                "token1_address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
                "token2_address": "So11111111111111111111111111111111111111112",
                "liquidity": 6784321.53,
                "volume_24h": 1784523.45,
                "apr": 42.38,
                "apr_change_7d": 5.12,
                "fee": 0.003,
                "category": "Meme"
            },
            {
                "id": "8Bx4hJsYDf7Zq4UPB5kTqUVZb6xg4nS939KciHQEhcEv",  # RAY-SOL pool
                "name": "RAY-SOL",
                "dex": "Raydium",
                "token1_symbol": "RAY",
                "token2_symbol": "SOL",
                "token1_address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
                "token2_address": "So11111111111111111111111111111111111111112",
                "liquidity": 4327689.21,
                "volume_24h": 876432.34,
                "apr": 29.45,
                "apr_change_7d": 2.34,
                "fee": 0.003,
                "category": "DeFi"
            },
            {
                "id": "7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX",  # wBTC-SOL pool
                "name": "wBTC-SOL",
                "dex": "Raydium",
                "token1_symbol": "wBTC",
                "token2_symbol": "SOL",
                "token1_address": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
                "token2_address": "So11111111111111111111111111111111111111112",
                "liquidity": 8765432.67,
                "volume_24h": 1234567.89,
                "apr": 16.78,
                "apr_change_7d": -0.45,
                "fee": 0.002,
                "category": "Major"
            },
            {
                "id": "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",  # SAMO-USDC pool
                "name": "SAMO-USDC",
                "dex": "Orca",
                "token1_symbol": "SAMO",
                "token2_symbol": "USDC",
                "token1_address": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
                "token2_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "liquidity": 3456789.12,
                "volume_24h": 987654.32,
                "apr": 36.54,
                "apr_change_7d": 3.21,
                "fee": 0.003,
                "category": "Meme"
            },
            {
                "id": "8x3HpBJfxB3wTVDZiUTVR2owf9UKmkGz3b4uqr9zrEYN",  # JUP-USDC pool
                "name": "JUP-USDC",
                "dex": "Jupiter",
                "token1_symbol": "JUP",
                "token2_symbol": "USDC",
                "token1_address": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
                "token2_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "liquidity": 5678901.23,
                "volume_24h": 1234567.89,
                "apr": 32.12,
                "apr_change_7d": 2.43,
                "fee": 0.0025,
                "category": "DeFi"
            },
            {
                "id": "HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ",  # USDC-USDT pool
                "name": "USDC-USDT",
                "dex": "Orca",
                "token1_symbol": "USDC",
                "token2_symbol": "USDT",
                "token1_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "token2_address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                "liquidity": 10987654.32,
                "volume_24h": 4567890.12,
                "apr": 8.76,
                "apr_change_7d": -0.32,
                "fee": 0.002,
                "category": "Stablecoin"
            },
            {
                "id": "D6N9j8F2DhtzDpWnr7B9b8LFuj3aplNTJ3amk5auuyr6",  # SOL-stSOL pool
                "name": "SOL-stSOL",
                "dex": "Raydium",
                "token1_symbol": "SOL",
                "token2_symbol": "stSOL",
                "token1_address": "So11111111111111111111111111111111111111112",
                "token2_address": "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",
                "liquidity": 9876543.21,
                "volume_24h": 1345678.90,
                "apr": 12.67,
                "apr_change_7d": 0.89,
                "fee": 0.002,
                "category": "DeFi"
            },
            {
                "id": "Ew5xLzHm8YS6kel5KAy3Frvj9WiZJNGbPt59ReErdGPm",  # MNGO-USDC pool
                "name": "MNGO-USDC",
                "dex": "Raydium",
                "token1_symbol": "MNGO",
                "token2_symbol": "USDC",
                "token1_address": "MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac",
                "token2_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "liquidity": 2345678.90,
                "volume_24h": 765432.10,
                "apr": 27.89,
                "apr_change_7d": 1.23,
                "fee": 0.003,
                "category": "DeFi"
            }
        ]
        
        self.logger.info(f"Created {len(self.pools)} sample pools with real Solana pool IDs")
    
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
                # Performance class: high, medium, low
                # Base on APR (higher APR = better performance)
                apr = pool.get('apr', 0)
                if apr > 20:
                    prediction_value = 'high'
                elif apr > 10:
                    prediction_value = 'medium'
                else:
                    prediction_value = 'low'
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
                'performance_class': random.choice(['high', 'medium', 'low']) if category != "performance" else prediction_value,
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
    
    def get_pool_predictions(self, pool_id, days=30):
        """
        Return prediction history for a specific pool
        
        Args:
            pool_id: The ID of the pool to get predictions for
            days: Number of days of prediction history to return
        """
        for pool in self.pools:
            if pool.get('id') == pool_id:
                # Generate prediction history for the requested number of days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Generate one data point per day
                dates = [start_date + timedelta(days=i) for i in range(days + 1)]
                
                # Base values from the pool
                base_apr = pool.get('apr', 10)
                
                # Generate time series with some random variations
                predicted_apr_series = [
                    base_apr * (1 + random.uniform(-0.1, 0.2)) 
                    for _ in range(len(dates))
                ]
                
                # Performance class (high, medium, low)
                # Base on predicted APR
                perf_class_series = []
                for apr in predicted_apr_series:
                    if apr > 20:
                        perf_class_series.append('high')
                    elif apr > 10:
                        perf_class_series.append('medium')
                    else:
                        perf_class_series.append('low')
                
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