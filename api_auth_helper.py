"""
Helper module for consistent API authentication
"""
import os

def get_api_headers():
    """
    Get the appropriate API headers based on the determined best format.
    This ensures consistent authentication across all API calls.
    """
    # Use environment variable for API key
    import os
    api_key = os.getenv("DEFI_API_KEY")
    
    if not api_key:
        import logging
        logging.error("DEFI_API_KEY environment variable not set - API calls will fail")
    
    # Based on our testing, "x-api-key" lowercase format works most consistently
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    return headers