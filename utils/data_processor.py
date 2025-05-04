import pandas as pd
import logging
from database.mock_db import MockDBManager

logger = logging.getLogger(__name__)

def get_pool_list(db=None):
    """
    Get list of pools from the database.
    Returns empty DataFrame if fails.
    """
    try:
        if db is None:
            logger.error("No database connection provided")
            return pd.DataFrame()  # Return empty DataFrame instead of using mock data
        
        pool_list = db.get_pool_list()
        
        if pool_list.empty:
            logger.warning("No pool data found in the database")
            return pd.DataFrame()  # Return empty DataFrame
            
        return pool_list
    except Exception as e:
        logger.error(f"Error getting pool list: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame instead of using mock data

def get_pool_details(db, pool_id):
    """
    Get details for a specific pool.
    Returns None if fails.
    """
    try:
        if db is None:
            logger.error("No database connection provided")
            return None  # Return None instead of using mock data
        
        pool_details = db.get_pool_details(pool_id)
        if pool_details is None:
            logger.warning(f"No details found for pool ID: {pool_id}")
            return None  # Return None instead of using mock data
            
        return pool_details
    except Exception as e:
        logger.error(f"Error getting pool details: {str(e)}")
        return None  # Return None instead of using mock data

def get_pool_metrics(db, pool_id, days=7):
    """
    Get historical metrics for a specific pool.
    Returns empty DataFrame if fails.
    """
    try:
        if db is None:
            logger.error("No database connection provided")
            return pd.DataFrame()  # Return empty DataFrame instead of using mock data
        
        metrics = db.get_pool_metrics(pool_id, days)
        if metrics.empty:
            logger.warning(f"No metrics found for pool ID: {pool_id}")
            return pd.DataFrame()  # Return empty DataFrame instead of using mock data
            
        return metrics
    except Exception as e:
        logger.error(f"Error getting pool metrics: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame instead of using mock data

def get_token_prices(db, token_symbols, days=7):
    """
    Get historical token prices.
    First tries CoinGecko for real data, then falls back to database.
    Returns empty DataFrame if both fail.
    """
    # First try to get real-time prices from CoinGecko
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from token_price_service import get_token_price
        
        # Get prices with source information from token price service
        price_data = []
        for token in token_symbols:
            price_result = get_token_price(token, return_source=True)
            
            if isinstance(price_result, tuple) and len(price_result) == 2:
                price, source = price_result
                if price and price > 0:
                    price_data.append({
                        'token_symbol': token,
                        'price_usd': float(price),
                        'price_source': source,
                        'timestamp': pd.Timestamp.now()
                    })
            elif price_result and float(price_result) > 0:
                # Handle case when only price is returned
                price_data.append({
                    'token_symbol': token,
                    'price_usd': float(price_result),
                    'price_source': 'unknown',
                    'timestamp': pd.Timestamp.now()
                })
        
        if price_data:
            return pd.DataFrame(price_data)
    except Exception as e:
        logger.warning(f"Could not get prices from CoinGecko: {str(e)}")
    
    # If CoinGecko fails, try the database
    try:
        if db is None:
            logger.error("No database connection provided")
            return pd.DataFrame(columns=['token_symbol', 'price_usd', 'timestamp'])
            
        prices = db.get_token_prices(token_symbols, days)
        if not prices.empty:
            return prices
    except Exception as e:
        logger.error(f"Error getting token prices from database: {str(e)}")
    
    # Return empty DataFrame if all sources fail
    logger.error("Failed to get token prices from all sources")
    return pd.DataFrame(columns=['token_symbol', 'price_usd', 'timestamp'])

def get_top_predictions(db, category="apr", limit=10, ascending=False):
    """
    Get top predictions based on category.
    Uses only authentic data from the database.
    """
    try:
        if db is None:
            logger.error("No database connection provided")
            return pd.DataFrame()  # Return empty DataFrame instead of using mock data
        
        predictions = db.get_top_predictions(category, limit, ascending)
        
        if predictions.empty:
            logger.warning(f"No prediction data found for category: {category}")
            return pd.DataFrame()  # Return empty DataFrame instead of using mock data
            
        return predictions
    except Exception as e:
        logger.error(f"Error getting top predictions: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame instead of using mock data

def get_pool_predictions(db, pool_id, days=30):
    """
    Get prediction history for a specific pool.
    Returns empty DataFrame if fails.
    
    Args:
        db: Database manager instance
        pool_id: The ID of the pool to get predictions for
        days: Number of days of prediction history to return (default: 30)
    """
    try:
        if db is None:
            logger.error("No database connection provided")
            return pd.DataFrame()  # Return empty DataFrame instead of using mock data
        
        # Try to call with days parameter if it's supported
        try:
            predictions = db.get_pool_predictions(pool_id, days)
        except TypeError:
            # If the days parameter isn't supported, fall back to basic version
            predictions = db.get_pool_predictions(pool_id)
            
        if predictions.empty:
            logger.warning(f"No prediction data found for pool ID: {pool_id}")
            return pd.DataFrame()  # Return empty DataFrame instead of using mock data
                
        return predictions
    except Exception as e:
        logger.error(f"Error getting pool predictions: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame instead of using mock data