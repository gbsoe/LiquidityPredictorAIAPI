"""Simplified SolPool Insight - For testing and debugging"""

import streamlit as st
import os
import pandas as pd
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simplified_solpool')

# Set page configuration
st.set_page_config(
    page_title="Simplified SolPool Insight",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function with robust error handling."""
    # Configure the sidebar
    with st.sidebar:
        st.sidebar.title("SolPool Insight")
        st.sidebar.markdown("### 📊 Data Sources")
        
        if st.sidebar.button("💹 Fetch DeFi API Data"):
            st.session_state['api_clicked'] = True
            st.info("Button clicked: Fetch DeFi API Data")
    
    # Main content area
    st.title("SolPool Insight - Simplified Version")
    st.markdown("### Welcome to the Simplified SolPool Analytics Platform")
    
    # Check if we have clicked the button
    if 'api_clicked' in st.session_state and st.session_state['api_clicked']:
        st.success("API data button was clicked!")
        
    # Create a simple table with sample data
    st.subheader("Sample Pool Data")
    
    # Create sample data
    sample_data = [
        {"Pool": "SOL/USDC", "APR": "5.2%", "Liquidity": "$2.5M", "Volume": "$500K"},
        {"Pool": "ETH/USDC", "APR": "4.8%", "Liquidity": "$1.8M", "Volume": "$320K"},
        {"Pool": "BTC/USDC", "APR": "3.9%", "Liquidity": "$3.2M", "Volume": "$780K"}
    ]
    
    # Display as a dataframe
    df = pd.DataFrame(sample_data)
    st.dataframe(df)
    
    # Add some basic controls
    st.header("Filter Options")
    
    col1, col2 = st.columns(2)
    with col1:
        min_apr = st.slider("Minimum APR", 0.0, 10.0, 3.0, 0.1)
        st.write(f"Selected minimum APR: {min_apr}%")
    
    with col2:
        min_liquidity = st.slider("Minimum Liquidity", 100000, 5000000, 1000000, 100000, format="$%d")
        st.write(f"Selected minimum liquidity: ${min_liquidity:,}")

if __name__ == "__main__":
    # Start the main application
    main()
