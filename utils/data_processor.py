import pandas as pd
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the data service directly for consistent data sources
from data_services.data_service import get_data_service

logger = logging.getLogger(__name__)

def get_pool_list(db=None):
    """
    Get list of pools from the data service directly.
    This ensures the same data source is used across the application.
    
    Args:
        db: Optional database connection (kept for backward compatibility)
        
    Returns:
        DataFrame with pool list or empty DataFrame if fails
    """
    try:
        # Get data directly from the data service for consistency
        data_service = get_data_service()
        if data_service:
            # Get the raw pool data from the service
            pool_data = data_service.get_all_pools()
            
            if not pool_data or len(pool_data) == 0:
                # If data service failed, try the DB as fallback
                if db is not None:
                    pool_list = db.get_pool_list()
                    if not pool_list.empty:
                        return pool_list
                
                logger.warning("No pool data found in the data service")
                return pd.DataFrame()
            
            # Convert list of dictionaries to DataFrame for consistency
            import pandas as pd
            if isinstance(pool_data, list):
                return pd.DataFrame(pool_data)
            else:
                # If it's already a DataFrame, return as is
                return pool_data
        elif db is not None:
            # Fallback to database if data service isn't available
            pool_list = db.get_pool_list()
            if not pool_list.empty:
                return pool_list
        
        logger.error("No data service or database connection available")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting pool list: {str(e)}")
        return pd.DataFrame()

def get_pool_details(db, pool_id):
    """
    Get details for a specific pool from the data service.
    
    Args:
        db: Optional database connection (kept for backward compatibility)
        pool_id: The ID of the pool to get details for
        
    Returns:
        Dictionary with pool details or None if fails
    """
    try:
        # Try to get data from the data service first for consistency
        data_service = get_data_service()
        if data_service:
            # Get all pool data
            all_pools = data_service.get_all_pools()
            if all_pools and len(all_pools) > 0:
                # Convert list of dictionaries to DataFrame for filtering
                import pandas as pd
                pools_df = pd.DataFrame(all_pools)
                
                # Filter for the specific pool ID if we have a DataFrame
                if not pools_df.empty and 'pool_id' in pools_df.columns:
                    pool_data = pools_df[pools_df['pool_id'] == pool_id]
                    if not pool_data.empty:
                        # Return the first row as a dictionary
                        return pool_data.iloc[0].to_dict()
        
        # If data service fails or pool not found, try the database
        if db is not None:
            pool_details = db.get_pool_details(pool_id)
            if pool_details is not None:
                return pool_details
            
        logger.warning(f"No details found for pool ID: {pool_id}")
        return None
    except Exception as e:
        logger.error(f"Error getting pool details: {str(e)}")
        return None

def get_pool_metrics(db, pool_id, days=7):
    """
    Get historical metrics for a specific pool from the data service.
    
    Args:
        db: Optional database connection (kept for backward compatibility)
        pool_id: The ID of the pool to get metrics for
        days: Number of days of metrics to return
        
    Returns:
        DataFrame with pool metrics or empty DataFrame if fails
    """
    try:
        # Try to get data from the historical service
        from historical_data_service import get_historical_service
        
        historical_service = get_historical_service()
        if historical_service:
            # Get metrics from the historical service
            metrics = historical_service.get_pool_metrics(pool_id, days)
            if not metrics.empty:
                return metrics
        
        # If historical service fails, try the database
        if db is not None:
            metrics = db.get_pool_metrics(pool_id, days)
            if not metrics.empty:
                return metrics
            
        logger.warning(f"No metrics found for pool ID: {pool_id}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting pool metrics: {str(e)}")
        return pd.DataFrame()

def get_token_prices(db, token_symbols, days=7):
    """
    Get token prices using the token_data_service first and then CoinGecko.
    This ensures the same data sources are used across the application.
    
    Args:
        db: Optional database connection (kept for backward compatibility)
        token_symbols: List of token symbols to get prices for
        days: Number of days of price history to return
        
    Returns:
        DataFrame with token prices or empty DataFrame if all sources fail
    """
    # First try the token_data_service which is used by the main page
    try:
        from token_data_service import get_token_data_service
        
        token_service = get_token_data_service()
        if token_service:
            price_data = []
            for token in token_symbols:
                # Try to get price from token service
                price = token_service.get_token_price(token)
                if price and float(price) > 0:
                    price_data.append({
                        'token_symbol': token,
                        'price_usd': float(price),
                        'price_source': 'token_service',
                        'timestamp': pd.Timestamp.now()
                    })
            
            if price_data:
                return pd.DataFrame(price_data)
    except Exception as e:
        logger.warning(f"Could not get prices from token_data_service: {str(e)}")
    
    # Then try using CoinGecko via the token_price_service
    try:
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
                    'price_source': 'coingecko',
                    'timestamp': pd.Timestamp.now()
                })
        
        if price_data:
            return pd.DataFrame(price_data)
    except Exception as e:
        logger.warning(f"Could not get prices from CoinGecko: {str(e)}")
    
    # If both primary sources fail, try the database
    try:
        if db is not None:
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