import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

def get_top_pools(db, metric, limit=10):
    """
    Get top pools by a specific metric
    
    Args:
        db: Database manager instance
        metric: Metric to sort by ('liquidity', 'volume', or 'apr')
        limit: Number of pools to return
    
    Returns:
        DataFrame with top pools
    """
    try:
        if metric == 'liquidity':
            return db.get_top_pools_by_liquidity(limit)
        elif metric == 'volume':
            return db.get_top_pools_by_volume(limit)
        elif metric == 'apr':
            return db.get_top_pools_by_apr(limit)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error in get_top_pools: {e}")
        return pd.DataFrame()

def get_blockchain_stats(db):
    """
    Get the latest blockchain stats
    
    Args:
        db: Database manager instance
    
    Returns:
        Dictionary with latest blockchain stats
    """
    try:
        stats_df = db.get_blockchain_stats(days=1)
        
        if stats_df.empty:
            return None
        
        # Get the latest stats
        latest_stats = stats_df.iloc[-1].to_dict()
        return latest_stats
    except Exception as e:
        print(f"Error in get_blockchain_stats: {e}")
        return None

def get_pool_list(db):
    """
    Get list of all pools
    
    Args:
        db: Database manager instance
    
    Returns:
        DataFrame with pool information
    """
    try:
        return db.get_all_pools()
    except Exception as e:
        print(f"Error in get_pool_list: {e}")
        return pd.DataFrame()

def get_pool_details(db, pool_id):
    """
    Get details for a specific pool
    
    Args:
        db: Database manager instance
        pool_id: ID of the pool
    
    Returns:
        Dictionary with pool details
    """
    try:
        pool_df = db.get_pool_by_id(pool_id)
        
        if pool_df.empty:
            return None
        
        return pool_df.iloc[0].to_dict()
    except Exception as e:
        print(f"Error in get_pool_details: {e}")
        return None

def get_pool_metrics(db, pool_id, days=30):
    """
    Get historical metrics for a pool
    
    Args:
        db: Database manager instance
        pool_id: ID of the pool
        days: Number of days of data to retrieve
    
    Returns:
        DataFrame with pool metrics
    """
    try:
        return db.get_pool_metrics(pool_id, days)
    except Exception as e:
        print(f"Error in get_pool_metrics: {e}")
        return pd.DataFrame()

def get_token_prices(db, token_symbols, days=30):
    """
    Get price history for tokens
    
    Args:
        db: Database manager instance
        token_symbols: List of token symbols
        days: Number of days of data to retrieve
    
    Returns:
        DataFrame with token price history
    """
    try:
        all_prices = []
        
        for symbol in token_symbols:
            prices = db.get_token_price_history(symbol, days)
            if not prices.empty:
                all_prices.append(prices)
        
        if all_prices:
            return pd.concat(all_prices, ignore_index=True)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error in get_token_prices: {e}")
        return pd.DataFrame()

def get_prediction_metrics(db, limit=10):
    """
    Get the latest prediction metrics for top pools
    
    Args:
        db: Database manager instance
        limit: Number of predictions to return
    
    Returns:
        DataFrame with prediction metrics
    """
    try:
        # Get latest predictions
        predictions = db.get_latest_predictions(limit)
        
        if predictions.empty:
            return None
        
        # Join with pool data to get names
        pool_data = db.get_all_pools()
        
        if not pool_data.empty:
            merged = pd.merge(
                predictions,
                pool_data[['pool_id', 'name']],
                on='pool_id',
                how='left'
            )
            
            # Rename for clarity
            merged = merged.rename(columns={'name': 'pool_name'})
            return merged
        else:
            return predictions
    except Exception as e:
        print(f"Error in get_prediction_metrics: {e}")
        return None

def get_pool_predictions(db, pool_id, days=30):
    """
    Get historical prediction data for a pool
    
    Args:
        db: Database manager instance
        pool_id: ID of the pool
        days: Number of days of predictions to retrieve
    
    Returns:
        DataFrame with pool prediction history
    """
    try:
        return db.get_pool_predictions(pool_id, days)
    except Exception as e:
        print(f"Error in get_pool_predictions: {e}")
        return pd.DataFrame()

def get_top_predictions(db, category, limit=10, ascending=False):
    """
    Get top predictions based on category
    
    Args:
        db: Database manager instance
        category: Category to sort by ('apr', 'performance', or 'risk')
        limit: Number of predictions to return
        ascending: Whether to sort in ascending order
    
    Returns:
        DataFrame with top predictions
    """
    try:
        # Get latest predictions
        predictions = db.get_latest_predictions(100)  # Get a large sample to filter from
        
        if predictions.empty:
            return pd.DataFrame()
        
        # Join with pool data to get names
        pool_data = db.get_all_pools()
        
        if not pool_data.empty:
            merged = pd.merge(
                predictions,
                pool_data[['pool_id', 'name']],
                on='pool_id',
                how='left'
            )
            
            # Rename for clarity
            merged = merged.rename(columns={'name': 'pool_name'})
            
            # Sort based on category
            if category == 'apr':
                sorted_df = merged.sort_values('predicted_apr', ascending=ascending)
            elif category == 'performance':
                # Map performance classes to numeric values for sorting
                perf_map = {'high': 0, 'medium': 1, 'low': 2}
                merged['perf_value'] = merged['performance_class'].map(perf_map)
                sorted_df = merged.sort_values('perf_value', ascending=ascending)
            else:  # risk
                sorted_df = merged.sort_values('risk_score', ascending=ascending)
            
            # Return top N
            return sorted_df.head(limit)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error in get_top_predictions: {e}")
        return pd.DataFrame()
