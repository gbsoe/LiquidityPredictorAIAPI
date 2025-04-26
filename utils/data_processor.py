import pandas as pd
import logging
from database.mock_db import MockDBManager

logger = logging.getLogger(__name__)

def get_pool_list(db=None):
    """
    Get list of pools from the database.
    Falls back to mock DB if real DB fails.
    """
    try:
        if db is None:
            db = MockDBManager()
        return db.get_pool_list()
    except Exception as e:
        logger.error(f"Error getting pool list: {str(e)}")
        # Fallback to mock if real DB fails
        mock_db = MockDBManager()
        return mock_db.get_pool_list()

def get_pool_details(db, pool_id):
    """
    Get details for a specific pool.
    Falls back to mock DB if real DB fails.
    """
    try:
        if db is None:
            db = MockDBManager()
        pool_details = db.get_pool_details(pool_id)
        if pool_details is None:
            # Fallback to mock if real DB returns None
            mock_db = MockDBManager()
            return mock_db.get_pool_details(pool_id)
        return pool_details
    except Exception as e:
        logger.error(f"Error getting pool details: {str(e)}")
        # Fallback to mock if real DB fails
        mock_db = MockDBManager()
        return mock_db.get_pool_details(pool_id)

def get_pool_metrics(db, pool_id, days=7):
    """
    Get historical metrics for a specific pool.
    Falls back to mock DB if real DB fails.
    """
    try:
        if db is None:
            db = MockDBManager()
        metrics = db.get_pool_metrics(pool_id, days)
        if metrics.empty:
            # Fallback to mock if real DB returns empty
            mock_db = MockDBManager()
            return mock_db.get_pool_metrics(pool_id, days)
        return metrics
    except Exception as e:
        logger.error(f"Error getting pool metrics: {str(e)}")
        # Fallback to mock if real DB fails
        mock_db = MockDBManager()
        return mock_db.get_pool_metrics(pool_id, days)

def get_token_prices(db, token_symbols, days=7):
    """
    Get historical token prices.
    First tries CoinGecko for real data, then falls back to database,
    and finally to mock DB if needed.
    """
    # First try to get real-time prices from CoinGecko
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from token_price_service import get_multiple_prices
        
        # Get prices from CoinGecko
        cg_prices = get_multiple_prices(token_symbols)
        
        if cg_prices and any(cg_prices.values()):
            # Convert to DataFrame format expected by the app
            price_data = []
            for token, price in cg_prices.items():
                if price > 0:
                    price_data.append({
                        'token_symbol': token,
                        'price_usd': price,
                        'timestamp': pd.Timestamp.now()
                    })
            
            if price_data:
                return pd.DataFrame(price_data)
    except Exception as e:
        logger.warning(f"Could not get prices from CoinGecko: {str(e)}")
    
    # If CoinGecko fails, try the database
    try:
        if db is None:
            db = MockDBManager()
        prices = db.get_token_prices(token_symbols, days)
        if not prices.empty:
            return prices
    except Exception as e:
        logger.error(f"Error getting token prices from database: {str(e)}")
    
    # If all else fails, use mock data
    try:
        mock_db = MockDBManager()
        return mock_db.get_token_prices(token_symbols, days)
    except Exception as e:
        logger.error(f"Error getting mock token prices: {str(e)}")
        # Return empty DataFrame as last resort
        return pd.DataFrame(columns=['token_symbol', 'price_usd', 'timestamp'])

def get_top_predictions(db, category="apr", limit=10, ascending=False):
    """
    Get top predictions based on category.
    Falls back to mock DB if real DB fails.
    """
    try:
        if db is None:
            db = MockDBManager()
        predictions = db.get_top_predictions(category, limit, ascending)
        if predictions.empty:
            # Fallback to mock if real DB returns empty
            mock_db = MockDBManager()
            return mock_db.get_top_predictions(category, limit, ascending)
        return predictions
    except Exception as e:
        logger.error(f"Error getting top predictions: {str(e)}")
        # Fallback to mock if real DB fails
        mock_db = MockDBManager()
        return mock_db.get_top_predictions(category, limit, ascending)

def get_pool_predictions(db, pool_id, days=30):
    """
    Get prediction history for a specific pool.
    Falls back to mock DB if real DB fails.
    
    Args:
        db: Database manager instance
        pool_id: The ID of the pool to get predictions for
        days: Number of days of prediction history to return (default: 30)
    """
    try:
        if db is None:
            db = MockDBManager()
        
        # Try to call with days parameter if it's supported
        try:
            predictions = db.get_pool_predictions(pool_id, days)
        except TypeError:
            # If the days parameter isn't supported, fall back to basic version
            predictions = db.get_pool_predictions(pool_id)
            
        if predictions.empty:
            # Fallback to mock if real DB returns empty
            mock_db = MockDBManager()
            try:
                return mock_db.get_pool_predictions(pool_id, days)
            except TypeError:
                return mock_db.get_pool_predictions(pool_id)
                
        return predictions
    except Exception as e:
        logger.error(f"Error getting pool predictions: {str(e)}")
        # Fallback to mock if real DB fails
        mock_db = MockDBManager()
        try:
            return mock_db.get_pool_predictions(pool_id, days)
        except TypeError:
            return mock_db.get_pool_predictions(pool_id)