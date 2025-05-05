import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Add project directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import create_metrics_chart, create_liquidity_volume_chart, create_token_price_chart, create_pool_comparison_chart
from utils.data_processor import get_pool_list, get_pool_metrics, get_pool_details, get_token_prices
from database.db_operations import DBManager
from api_key_manager import get_defi_api_key, render_api_key_form

# Page configuration
st.set_page_config(
    page_title="Data Explorer - Solana Liquidity Pool Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database connection
@st.cache_resource
def get_db_connection():
    return DBManager()

db = get_db_connection()

# Header
st.title("Data Explorer")
st.markdown("Explore detailed metrics and historical data for Solana liquidity pools.")

st.image("https://images.unsplash.com/photo-1542744173-05336fcc7ad4", 
         caption="Financial Data Visualization")

# Sidebar for pool selection
st.sidebar.header("Pool Selection")

# Add API key form
render_api_key_form()

# Show a warning if API key is missing
if not get_defi_api_key():
    st.sidebar.warning("⚠️ API key missing. Set your API key to access pool data.")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_pool_list():
    # Check for API key
    api_key = get_defi_api_key()
    if not api_key:
        st.sidebar.warning("⚠️ API key not configured. Using cached data from previous API tests.")
    
    # Try to get pool list
    return get_pool_list(db)

try:
    pool_list = load_pool_list()
    
    if not pool_list.empty:
        pool_options = [f"{row['name']} ({row['pool_id']})" for _, row in pool_list.iterrows()]
        
        selected_pool_option = st.sidebar.selectbox(
            "Select Pool:",
            options=pool_options
        )
        
        # Extract pool_id from selection
        try:
            selected_pool_id = selected_pool_option.split("(")[-1].split(")")[0]
            st.session_state['selected_pool_id'] = selected_pool_id  # Cache the ID in session state
        except Exception as e:
            # If we can't parse the ID, show an error but don't return since we're not in a function
            st.error("Could not extract pool ID from selection")
            selected_pool_id = ""  # Use empty ID instead of returning
        
        # Time period selection
        time_period = st.sidebar.selectbox(
            "Time Period:",
            options=["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
            index=1  # Default to 7 days
        )
        
        # Convert time period to days
        if time_period == "Last 24 Hours":
            days = 1
        elif time_period == "Last 7 Days":
            days = 7
        else:  # Last 30 Days
            days = 30
        
        # Load pool details
        pool_details = get_pool_details(db, selected_pool_id)
        
        if pool_details is not None:
            # Pool details section
            pool_name = pool_details.get('name', 'Unknown')
            # If name is not there, try to construct it
            if pool_name == 'Unknown':
                token1 = pool_details.get('token1_symbol', '')
                token2 = pool_details.get('token2_symbol', '')
                if token1 and token2:
                    pool_name = f"{token1}/{token2}"
                else:
                    # Use a truncated ID
                    pool_id_short = selected_pool_id[:8] + '...' if len(selected_pool_id) > 8 else selected_pool_id
                    pool_name = f"Pool {pool_id_short}"
            
            st.header(f"Pool Details: {pool_name}")
            
            # Pool summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Use get() method with defaults to handle missing fields
            with col1:
                liquidity = pool_details.get('liquidity', 0) or pool_details.get('tvl', 0) or pool_details.get('liquidityUsd', 0) or 0
                st.metric("Liquidity", f"${float(liquidity):,.2f}")
                
            with col2:
                volume = pool_details.get('volume_24h', 0) or pool_details.get('volume24h', 0) or pool_details.get('volume', 0) or 0
                st.metric("24h Volume", f"${float(volume):,.2f}")
                
            with col3:
                apr = pool_details.get('apr', 0) or pool_details.get('apr24h', 0) or pool_details.get('apy', 0) or 0
                st.metric("APR", f"{float(apr):.2f}%")
            
            with col4:
                # Handle missing timestamp
                timestamp = pool_details.get('timestamp', None)
                if timestamp:
                    # Calculate time since last update
                    try:
                        last_updated = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_diff = datetime.now() - last_updated
                        
                        if time_diff < timedelta(hours=1):
                            time_str = f"{int(time_diff.total_seconds() / 60)} minutes ago"
                        elif time_diff < timedelta(days=1):
                            time_str = f"{int(time_diff.total_seconds() / 3600)} hours ago"
                        else:
                            time_str = f"{int(time_diff.days)} days ago"
                    except Exception as e:
                        time_str = "Unknown"
                else:
                    time_str = "Unknown"
                
                st.metric("Last Updated", time_str)
            
            # Load pool metrics
            pool_metrics = get_pool_metrics(db, selected_pool_id, days)
            
            if not pool_metrics.empty:
                st.subheader("Historical Metrics")
                
                # Create tabs for different visualizations
                metrics_tabs = st.tabs(["APR", "Liquidity & Volume", "Token Prices"])
                
                # APR Tab
                with metrics_tabs[0]:
                    st.plotly_chart(
                        create_metrics_chart(pool_metrics, 'apr', 'APR (%)', 'APR Over Time'),
                        use_container_width=True
                    )
                
                # Liquidity & Volume Tab
                with metrics_tabs[1]:
                    st.plotly_chart(
                        create_liquidity_volume_chart(pool_metrics),
                        use_container_width=True
                    )
                
                # Token Prices Tab
                with metrics_tabs[2]:
                    # Extract token symbols from pool name or directly from token fields
                    pool_name = pool_details.get('name', '')
                    
                    # Try to get token symbols from various fields
                    token_symbol1 = pool_details.get('token1_symbol', '') or pool_details.get('baseSymbol', '') or ''
                    token_symbol2 = pool_details.get('token2_symbol', '') or pool_details.get('quoteSymbol', '') or ''
                    
                    # If we don't have token symbols directly, try to extract from name
                    if (not token_symbol1 or not token_symbol2) and '/' in pool_name:
                        token_parts = pool_name.split('/')
                        if len(token_parts) >= 2:
                            token_symbol1 = token_symbol1 or token_parts[0].strip()
                            token_symbol2 = token_symbol2 or token_parts[1].strip()
                    
                    # Only proceed if we have both token symbols
                    if token_symbol1 and token_symbol2:
                        # Get token prices
                        token_prices = get_token_prices(db, [token_symbol1, token_symbol2], days)
                        
                        if not token_prices.empty:
                            st.plotly_chart(
                                create_token_price_chart(token_prices, token_symbol1, token_symbol2),
                                use_container_width=True
                            )
                        else:
                            st.info(f"No price data available for {token_symbol1} and {token_symbol2}.")
                    else:
                        st.info("Token price data not available for this pool.")
                
                # Raw data option
                with st.expander("View Raw Data"):
                    st.dataframe(
                        pool_metrics[['timestamp', 'liquidity', 'volume', 'apr']].rename(
                            columns={
                                'timestamp': 'Timestamp', 
                                'liquidity': 'Liquidity (USD)', 
                                'volume': 'Volume (USD)', 
                                'apr': 'APR (%)'
                            }
                        ),
                        use_container_width=True
                    )
            else:
                st.info(f"No historical metrics available for this pool over the selected time period ({time_period}).")
        else:
            st.error(f"Error loading details for pool: {selected_pool_id}")
    else:
        # Check if API key is set
        defi_api_key = get_defi_api_key()
        if not defi_api_key:
            st.error("No pools available - API key required.")
            st.info("Please configure your DeFi API key using the form in the sidebar to access authentic pool data.")
            
            # Add a button to quickly scroll to the API key form
            if st.button("Configure API Key"):
                st.info("Please look at the API Key Configuration section in the sidebar.")
        else:
            st.warning("No pools available. The system may be collecting initial data.")
except Exception as e:
    st.error(f"Error loading pool data: {str(e)}")

# Pool comparison section
st.header("Pool Comparison")

# Select pools to compare
try:
    if not pool_list.empty:
        pool_options = [f"{row['name']} ({row['pool_id']})" for _, row in pool_list.iterrows()]
        
        selected_pools = st.multiselect(
            "Select pools to compare (max 5):",
            options=pool_options,
            max_selections=5
        )
        
        if selected_pools:
            # Extract pool IDs with error handling
            pool_ids = []
            try:
                pool_ids = [pool.split("(")[-1].split(")")[0] for pool in selected_pools]
            except Exception as e:
                st.error(f"Could not extract pool IDs from selection: {str(e)}")
            
            # Metric to compare
            compare_metric = st.selectbox(
                "Metric to compare:",
                options=["APR", "Liquidity", "Volume"],
                index=0
            )
            
            # Convert to DB column
            metric_map = {
                "APR": "apr",
                "Liquidity": "liquidity",
                "Volume": "volume"
            }
            db_metric = metric_map[compare_metric]
            
            # Get data for each pool
            comparison_data = []
            
            for pool_id in pool_ids:
                pool_details = get_pool_details(db, pool_id)
                pool_metrics = get_pool_metrics(db, pool_id, days)
                
                if pool_details is not None and not pool_metrics.empty:
                    pool_data = pool_metrics[['timestamp', db_metric]].copy()
                    # Get pool name with fallback
                    pool_name = pool_details.get('name', 'Unknown')
                    if pool_name == 'Unknown':
                        # Try to construct from token symbols
                        token1 = pool_details.get('token1_symbol', '')
                        token2 = pool_details.get('token2_symbol', '')
                        if token1 and token2:
                            pool_name = f"{token1}/{token2}"
                        else:
                            # Use a truncated ID
                            pool_id_short = pool_id[:8] + '...' if len(pool_id) > 8 else pool_id
                            pool_name = f"Pool {pool_id_short}"
                    pool_data['pool_name'] = pool_name
                    comparison_data.append(pool_data)
            
            if comparison_data:
                # Combine data
                combined_data = pd.concat(comparison_data)
                
                # Create comparison chart
                st.plotly_chart(
                    create_pool_comparison_chart(combined_data, db_metric, compare_metric),
                    use_container_width=True
                )
            else:
                st.info("No data available for the selected pools and time period.")
        else:
            st.info("Select pools to compare.")
    else:
        # Check if API key is set
        defi_api_key = get_defi_api_key()
        if not defi_api_key:
            st.warning("No pools available for comparison - API key needed.")
        else:
            st.warning("No pools available for comparison.")
except Exception as e:
    st.error(f"Error in pool comparison: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Solana Liquidity Pool Analysis System • Data Explorer")
