import os
import time
import json
import logging
import sqlite3
import requests
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('price_tracker.log')
    ]
)

logger = logging.getLogger('price_tracker')

# Constants
DATABASE_PATH = os.getenv('DATABASE_PATH', '../database/liquidity_pools.db')
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '')  # Optional API key
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

class PriceTracker:
    """
    Service for tracking token prices from CoinGecko
    """
    
    def __init__(self):
        """Initialize the price tracker"""
        self.api_url = COINGECKO_API_URL
        self.api_key = COINGECKO_API_KEY
        self.session = requests.Session()
        self.headers = {}
        
        if self.api_key:
            self.headers['x-cg-pro-api-key'] = self.api_key
            logger.info("Using CoinGecko API key")
        else:
            logger.warning("No CoinGecko API key provided, rate limits may apply")
    
    def fetch_with_retry(self, url, params=None):
        """Fetch data with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(
                    url, 
                    params=params, 
                    headers=self.headers,
                    timeout=REQUEST_TIMEOUT
                )
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', RETRY_DELAY))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Failed to fetch data from {url} after {MAX_RETRIES} attempts")
                    raise
    
    def get_token_ids(self, token_symbols):
        """
        Get CoinGecko token IDs for a list of token symbols
        """
        try:
            # Fetch the list of all coins from CoinGecko
            url = f"{self.api_url}/coins/list"
            all_coins = self.fetch_with_retry(url)
            
            # Create a mapping of symbol to coin id
            symbol_to_id = {}
            for coin in all_coins:
                symbol = coin.get('symbol', '').lower()
                coin_id = coin.get('id')
                if symbol and coin_id:
                    # If multiple coins have the same symbol, prefer the one that appears to be on Solana
                    # This is a heuristic and might need refinement
                    if symbol not in symbol_to_id or 'solana' in coin_id.lower():
                        symbol_to_id[symbol] = coin_id
            
            # Look up IDs for the requested symbols
            result = {}
            for symbol in token_symbols:
                symbol_lower = symbol.lower()
                if symbol_lower in symbol_to_id:
                    result[symbol] = symbol_to_id[symbol_lower]
                else:
                    logger.warning(f"Could not find CoinGecko ID for token symbol: {symbol}")
            
            logger.info(f"Found CoinGecko IDs for {len(result)}/{len(token_symbols)} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching token IDs: {e}")
            return {}
    
    def get_token_prices(self, token_ids, vs_currency='usd'):
        """
        Get current prices for a list of token IDs
        """
        if not token_ids:
            logger.warning("No token IDs provided for price fetching")
            return {}
            
        try:
            # Format the list of IDs for the API request
            ids_param = ','.join(token_ids)
            
            # Fetch prices
            url = f"{self.api_url}/simple/price"
            params = {
                'ids': ids_param,
                'vs_currencies': vs_currency
            }
            
            price_data = self.fetch_with_retry(url, params)
            
            # Transform response to a simpler format
            prices = {}
            for token_id, price_info in price_data.items():
                prices[token_id] = price_info.get(vs_currency)
            
            logger.info(f"Successfully fetched prices for {len(prices)} tokens")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching token prices: {e}")
            return {}
    
    def get_historical_prices(self, token_id, days=30, vs_currency='usd'):
        """
        Get historical price data for a token
        """
        try:
            url = f"{self.api_url}/coins/{token_id}/market_chart"
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': 'daily'
            }
            
            data = self.fetch_with_retry(url, params)
            
            # Extract price data from response
            prices = []
            for timestamp_ms, price in data.get('prices', []):
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                prices.append({
                    'token_id': token_id,
                    'timestamp': timestamp.isoformat(),
                    'price': price
                })
            
            logger.info(f"Successfully fetched {len(prices)} historical price points for {token_id}")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching historical prices for {token_id}: {e}")
            return []
    
    def get_tokens_from_database(self):
        """
        Get list of token symbols from the pools in the database
        """
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # Get unique token symbols from pool names (assuming format like "TOKEN1/TOKEN2")
            cursor.execute("SELECT DISTINCT name FROM pool_data")
            pool_names = cursor.fetchall()
            
            tokens = set()
            for name in pool_names:
                if name[0] and '/' in name[0]:
                    parts = name[0].split('/')
                    tokens.add(parts[0].strip())
                    tokens.add(parts[1].strip())
            
            conn.close()
            return list(tokens)
            
        except Exception as e:
            logger.error(f"Error getting tokens from database: {e}")
            return []
    
    def update_token_prices(self):
        """
        Update prices for all tokens in the database
        """
        try:
            # Get list of tokens from the database
            tokens = self.get_tokens_from_database()
            logger.info(f"Found {len(tokens)} unique tokens in the database")
            
            if not tokens:
                logger.warning("No tokens found in database")
                return
            
            # Get CoinGecko IDs for these tokens
            token_id_map = self.get_token_ids(tokens)
            if not token_id_map:
                logger.warning("Could not find CoinGecko IDs for any tokens")
                return
            
            # Get current prices
            token_ids = list(token_id_map.values())
            prices = self.get_token_prices(token_ids)
            
            if not prices:
                logger.warning("Could not fetch any token prices")
                return
            
            # Prepare data for database
            current_time = datetime.now().isoformat()
            price_records = []
            
            for symbol, token_id in token_id_map.items():
                if token_id in prices and prices[token_id] is not None:
                    price_records.append({
                        'token_symbol': symbol,
                        'price_usd': prices[token_id],
                        'timestamp': current_time
                    })
            
            # Store in database
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            for record in price_records:
                cursor.execute('''
                    INSERT INTO token_prices (token_symbol, price_usd, timestamp)
                    VALUES (?, ?, ?)
                ''', (
                    record['token_symbol'],
                    record['price_usd'],
                    record['timestamp']
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully updated prices for {len(price_records)} tokens")
            
            # Update historical data for a subset of tokens
            # To avoid rate limiting, we'll only update a few tokens each time
            # In production, this would be more sophisticated
            tokens_for_history = list(token_id_map.values())[:5]  # First 5 tokens
            self.update_historical_prices(tokens_for_history)
            
        except Exception as e:
            logger.error(f"Error updating token prices: {e}")
    
    def update_historical_prices(self, token_ids, days=30):
        """
        Update historical price data for selected tokens
        """
        if not token_ids:
            return
            
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            for token_id in token_ids:
                # Get historical data
                price_data = self.get_historical_prices(token_id, days)
                
                if not price_data:
                    continue
                
                # Store in database
                for record in price_data:
                    cursor.execute('''
                        INSERT OR IGNORE INTO token_price_history 
                        (token_id, price_usd, timestamp)
                        VALUES (?, ?, ?)
                    ''', (
                        record['token_id'],
                        record['price'],
                        record['timestamp']
                    ))
                
                logger.info(f"Added {len(price_data)} historical price points for {token_id}")
                
                # Avoid rate limiting
                time.sleep(1)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating historical prices: {e}")

if __name__ == "__main__":
    tracker = PriceTracker()
    tracker.update_token_prices()
