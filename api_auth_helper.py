"""
Helper module for consistent API authentication
"""
import os

def get_api_headers():
    """
    Get the appropriate API headers based on the determined best format.
    This ensures consistent authentication across all API calls.
    """
    # Hard-code the API key for now, to ensure it works correctly
    api_key = "9feae0d0af47e4948e061f2d7820461e374e040c21cf65c087166d7ed18f5ed6"
    
    # Based on our testing, "x-api-key" lowercase format works most consistently
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    return headers