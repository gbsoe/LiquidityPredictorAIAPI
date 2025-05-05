"""Helper module for consistent API authentication"""
import os
import logging

# Store the API key for consistent access
_api_key = None

def set_api_key(key):
    """
    Set the API key to use for all API calls.
    This should be called at application startup.
    
    Args:
        key: API key to use
    """
    global _api_key
    
    if not key:
        logging.error("No API key provided - API calls will fail")
        return
        
    # Set global variable
    _api_key = key
    
    # Also set environment variable for other components
    os.environ["DEFI_API_KEY"] = key
    
    logging.info(f"API key set successfully: {key[:5]}...")

def get_api_key():
    """
    Get the current API key.
    
    Returns:
        API key
    """
    global _api_key
    
    # Try global variable first
    if _api_key:
        return _api_key
        
    # Then try environment variable
    env_key = os.getenv("DEFI_API_KEY")
    if env_key:
        _api_key = env_key  # Cache it
        return env_key
        
    # Hard-coded fallback for development only
    fallback_key = "9feae0d0af47e4948e061f2d7820461e374e040c21cf65c087166d7ed18f5ed6"
    _api_key = fallback_key
    os.environ["DEFI_API_KEY"] = fallback_key
    
    logging.warning("Using fallback API key - should be configured properly in production")
    return fallback_key

def get_api_headers():
    """
    Get the appropriate API headers based on the determined best format.
    This ensures consistent authentication across all API calls.
    """
    # Use configured API key
    api_key = get_api_key()
    
    # Based on our testing, "x-api-key" lowercase format works most consistently
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    return headers