"""
API Key Manager for SolPool Insight

This module provides functions for managing API keys for external services,
including the DeFi Aggregation API.
"""

import os
import logging
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

def get_defi_api_key() -> str:
    """
    Get the DeFi API key from session state or environment variables.
    
    Returns:
        The API key as a string, or an empty string if not found
    """
    # Try to get from session state first (UI configured)
    try:
        if "defi_api_key" in st.session_state and st.session_state["defi_api_key"]:
            return st.session_state["defi_api_key"]
    except Exception:
        # Streamlit may not be in context
        pass
    
    # Then try environment variable
    api_key = os.environ.get("DEFI_API_KEY")
    if api_key:
        # Also store in session state for consistency
        try:
            st.session_state["defi_api_key"] = api_key
        except Exception:
            # Streamlit may not be in context
            pass
        return api_key
    
    # Not found - return empty string to avoid None issues
    return ""

def set_defi_api_key(api_key: str) -> bool:
    """
    Set the DeFi API key in session state and environment variables.
    
    Args:
        api_key: The API key to set
        
    Returns:
        True if set successfully, False otherwise
    """
    if not api_key:
        logger.warning("Attempted to set empty API key")
        return False
    
    try:
        # Set in session state
        st.session_state["defi_api_key"] = api_key
        
        # Set in environment for other components
        os.environ["DEFI_API_KEY"] = api_key
        
        logger.info("DeFi API key configured successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting API key: {str(e)}")
        return False

def render_api_key_form():
    """Render a form for configuring the DeFi API key"""
    with st.expander("🔑 API Key Configuration", expanded=not get_defi_api_key()):
        st.write("Configure your DeFi API key to access authentic liquidity pool data.")
        
        # Show current status
        current_key = get_defi_api_key()
        if current_key:
            st.success("✅ DeFi API key is configured")
            
            # Show masked key
            masked_key = current_key[:4] + "*" * (len(current_key) - 8) + current_key[-4:]
            st.code(masked_key, language=None)
            
            # Option to clear
            if st.button("Clear API Key"):
                st.session_state["defi_api_key"] = None
                if "DEFI_API_KEY" in os.environ:
                    del os.environ["DEFI_API_KEY"]
                st.success("API key cleared. Refresh to apply changes.")
                st.rerun()
        else:
            st.warning("⚠️ DeFi API key not configured")
            st.info("Configure your API key to access authentic pool data.")
        
        # API key input form
        with st.form("api_key_form"):
            new_api_key = st.text_input(
                "Enter DeFi API Key",
                type="password",
                help="Your DeFi API key for accessing authenticated data"
            )
            
            submitted = st.form_submit_button("Save API Key")
            
            if submitted and new_api_key:
                if set_defi_api_key(new_api_key):
                    st.success("✅ API key saved successfully")
                    st.info("Reloading application to apply changes...")
                    # Force reload
                    st.rerun()
                else:
                    st.error("Failed to save API key")