"""
Database Handler Manager for SolPool Insight application.
Provides a centralized access point for database operations.
"""

import sys
import os
import logging
import traceback

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
        return _db_handler_instance
        
    try:
        # Import the real db_handler module
        import db_handler
        
        # Get the handler using the module's function
        handler = db_handler.get_db_handler()
        
        # Store it for future use
        _db_handler_instance = handler
        
        logger.info("Successfully initialized database handler")
        return handler
    except Exception as e:
        logger.error(f"Error initializing database handler: {str(e)}")
        logger.error(traceback.format_exc())
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
