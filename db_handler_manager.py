"""
Database Handler Manager for SolPool Insight application.
Provides a centralized access point for database operations.
"""

import sys
import os
import logging
import traceback
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_handler_manager')

# Global DB handler instance
_db_handler_instance = None

def get_db_handler():
    """
    Get a database handler instance with proper error checking
    
    Returns:
        A valid database handler instance or None if not available
    """
    global _db_handler_instance
    
    if _db_handler_instance is not None:
        # Check if the instance is still valid
        try:
            # If it has an engine attribute, try a simple test query
            if hasattr(_db_handler_instance, 'engine') and _db_handler_instance.engine is not None:
                with _db_handler_instance.engine.connect() as conn:
                    conn.execute(_db_handler_instance.sa.text("SELECT 1"))
                # If we get here, the connection is still valid
                return _db_handler_instance
        except Exception:
            # Connection is no longer valid, clear the instance
            logger.warning("Existing database connection is no longer valid - will attempt to reconnect")
            _db_handler_instance = None
    
    # Try multiple approaches to get a valid handler
    for attempt in range(3):
        try:
            # Import the real db_handler module
            import db_handler
            
            # Get the handler using the module's function
            handler = db_handler.get_db_handler()
            
            # Test that the handler is working
            if hasattr(handler, 'engine') and handler.engine is not None:
                with handler.engine.connect() as conn:
                    conn.execute(handler.sa.text("SELECT 1"))
                
                # Store it for future use
                _db_handler_instance = handler
                
                logger.info("Successfully initialized database handler")
                return handler
            else:
                logger.warning("Database handler has no valid engine")
                time.sleep(0.5)  # Brief pause before retry
        except ImportError as e:
            logger.error(f"Could not import db_handler module: {str(e)}")
            logger.error("This is a critical error - database functionality will not be available")
            return None
        except Exception as e:
            logger.error(f"Error initializing database handler (attempt {attempt+1}/3): {str(e)}")
            logger.debug(traceback.format_exc())
            time.sleep(1)  # Pause before retry
    
    logger.error("All attempts to initialize database handler failed")
    return None

def is_db_connected():
    """
    Check if the database is connected
    
    Returns:
        bool: True if connected, False otherwise
    """
    handler = get_db_handler()
    
    if handler is None:
        return False
        
    try:
        # Check if engine exists
        if hasattr(handler, 'engine') and handler.engine is not None:
            # Try to perform a simple test query
            try:
                with handler.engine.connect() as conn:
                    conn.execute(handler.sa.text("SELECT 1"))
                return True
            except Exception:
                return False
        return False
    except Exception:
        return False

def safe_db_call(method_name, *args, **kwargs):
    """
    Safely call a database method with error handling
    
    Args:
        method_name: Name of the database method to call
        *args: Arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
        
    Returns:
        Result of the method call or None if it fails
    """
    handler = get_db_handler()
    
    if handler is None:
        logger.warning(f"Cannot call {method_name} - database handler not available")
        return None
        
    try:
        # Check if the method exists
        if hasattr(handler, method_name):
            method = getattr(handler, method_name)
            
            # Call the method with the provided arguments
            return method(*args, **kwargs)
        else:
            logger.warning(f"Method {method_name} not found in database handler")
            return None
    except Exception as e:
        logger.error(f"Error calling database method {method_name}: {str(e)}")
        return None


# Add special handlers for common operations that might fail
def get_watchlists():
    """
    Get watchlists with enhanced error handling for missing tables
    
    Returns:
        List of watchlists or empty list if not available
    """
    try:
        result = safe_db_call('get_watchlists')
        if result:
            return result
    except Exception as e:
        logger.error(f"Error retrieving watchlists: {str(e)}")
    
    # Return empty list as fallback
    return []

def get_pools_in_watchlist(watchlist_id):
    """
    Get pools in a watchlist with enhanced error handling
    
    Args:
        watchlist_id: ID of the watchlist
        
    Returns:
        List of pool IDs or empty list if not available
    """
    try:
        result = safe_db_call('get_pools_in_watchlist', watchlist_id)
        if result:
            return result
    except Exception as e:
        logger.error(f"Error retrieving pools in watchlist: {str(e)}")
    
    # Return empty list as fallback
    return []

def get_pools(limit=None):
    """
    Get pools from database with enhanced error handling
    
    Args:
        limit: Maximum number of pools to retrieve (None for all)
        
    Returns:
        List of pool data dictionaries or empty list if not available
    """
    try:
        # Try to get pools through the safe_db_call
        result = safe_db_call('get_pools', limit)
        if result and len(result) > 0:
            logger.info(f"Successfully retrieved {len(result)} pools from database")
            return result
            
        # If no pools were found or the call failed, check for JSON backup
        logger.warning("No pools found in database, checking JSON backup...")
        json_result = safe_db_call('load_from_json')
        if json_result and len(json_result) > 0:
            logger.info(f"Retrieved {len(json_result)} pools from JSON backup")
            return json_result
    except Exception as e:
        logger.error(f"Error retrieving pools: {str(e)}")
    
    # Return empty list as fallback
    logger.error("Could not retrieve pools from any source")
    return []

def store_pools(pool_data, replace=True):
    """
    Store pools in database with enhanced error handling
    
    Args:
        pool_data: List of dictionaries containing pool data
        replace: If True, replace existing entries; if False, skip duplicates
        
    Returns:
        Number of pools stored
    """
    if not pool_data or len(pool_data) == 0:
        logger.warning("No pool data provided to store_pools")
        return 0
        
    try:
        # Try to store pools through the safe_db_call
        result = safe_db_call('store_pools', pool_data, replace)
        if result and result > 0:
            logger.info(f"Successfully stored {result} pools in database")
            return result
            
        # If the database store failed, try to backup to JSON
        logger.warning("Database store failed, attempting JSON backup...")
        json_result = safe_db_call('backup_to_json', pool_data)
        if json_result is not None:
            logger.info("Successfully backed up pools to JSON file")
            return len(pool_data)
    except Exception as e:
        logger.error(f"Error storing pools: {str(e)}")
        
        # Try JSON backup as last resort
        try:
            safe_db_call('backup_to_json', pool_data)
            logger.info("Emergency JSON backup successful after database error")
            return 0
        except Exception as json_e:
            logger.error(f"Emergency JSON backup also failed: {str(json_e)}")
    
    # Return 0 as fallback
    return 0
