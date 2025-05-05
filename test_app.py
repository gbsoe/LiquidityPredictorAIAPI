
import streamlit as st
import pandas as pd
import os
import sys
import time
from datetime import datetime, timedelta
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('solpool_insight')

# Set page configuration
st.set_page_config(
    page_title="SolPool Insight - Solana Liquidity Pool Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database handler
try:
    from database.db_operations import DBManager
    db_handler = DBManager()
    DB_CONNECTED = True
except Exception as e:
    logger.error(f"Error initializing database handler: {str(e)}")
    DB_CONNECTED = False
    db_handler = None

def load_data(limit=50):
    """Load pool data from database or fallback to JSON"""
    try:
        if DB_CONNECTED and db_handler:
            pools = db_handler.get_pools(limit=limit)
            return pools
        else:
            # Fallback to JSON
            st.warning("Database connection not available - loading sample data")
            try:
                with open('extracted_pools.json', 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                return []
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return []

def main():
    """Main application function"""
    # Configure sidebar
    with st.sidebar:
        st.title("SolPool Insight")
        st.session_state['pool_count'] = 15
    
    # Main content
    st.title("SolPool Insight - Solana Liquidity Pool Analytics")
    
    # About section
    with st.expander("About SolPool Insight"):
        st.markdown("""
        SolPool Insight provides real-time analytics for Solana liquidity pools across various DEXes
        including Raydium, Orca, Jupiter, Meteora, Saber, and more.
        """)
    
    # Database status
    if DB_CONNECTED and db_handler:
        st.success("✓ Connected to PostgreSQL database")
    else:
        st.warning("⚠ Database connection not available - using file-based storage")
    
    # Create basic tabs
    tab_explore, tab_tokens = st.tabs(["Data Explorer", "Token Explorer"])
    
    # Load data
    pool_data = load_data()
    
    if not pool_data or len(pool_data) == 0:
        st.error("No pool data available. Please check the database connection.")
        return
    
    # Convert to DataFrame for easier manipulation
    try:
        df = pd.DataFrame(pool_data)
        st.write(f"Loaded {len(df)} pools successfully")
        
        # Display data in the explore tab
        with tab_explore:
            st.subheader("Liquidity Pool Data")
            st.dataframe(df)
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main()
