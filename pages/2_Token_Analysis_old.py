
"""
Token Analysis - Comprehensive Token Explorer for Solpool Insight

This page provides detailed analysis for tokens across Solana DEXes,
with comprehensive metadata and visualization of token relationships.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
import logging
import traceback
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("token_analysis_page")

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our services
from token_data_service import get_token_service
from token_price_service import get_token_price, get_multiple_prices
from data_services.data_service import DataService

# Initialize the data service for direct access to pool data
data_service = DataService()

# Page configuration
st.set_page_config(
    page_title="Token Analysis - Solpool Insight",
    page_icon="💰",
    layout="wide"
)

# Initialize the token service
token_service = get_token_service()

# Function to load token data from actual pools
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_token_data():
    try:
        # Create a placeholder for pool data in case we can't get it
        token_df = pd.DataFrame()
        
        # First, try to get pools from data service
        try:
            # Get all pools to extract actual tokens used in the system
            all_pools = data_service.get_all_pools()
            if not all_pools or len(all_pools) == 0:
                logger.warning("No pools returned from data service")
                st.warning("No pool data is available. Using static token list instead.")
                # Use the common token list as fallback
                common_tokens = ["SOL", "USDC", "USDT", "ETH", "BTC", "RAY", "ORCA"]
                token_symbols = set(common_tokens)
            else:
                # Extract unique tokens from all pools
                token_symbols = set()
                for pool in all_pools:
                    # Extract token symbols
                    token1 = pool.get('token1', {})
                    token2 = pool.get('token2', {})
                    
                    token1_symbol = token1.get('symbol', '') if isinstance(token1, dict) else str(token1)
                    token2_symbol = token2.get('symbol', '') if isinstance(token2, dict) else str(token2)
                    
                    if token1_symbol and len(token1_symbol) > 0:
                        token_symbols.add(token1_symbol)
                    if token2_symbol and len(token2_symbol) > 0:
                        token_symbols.add(token2_symbol)
                        
                logger.info(f"Found {len(token_symbols)} unique tokens in pool data")
                
                # Create a mapping of tokens to pools for quick lookup
                token_pools_map = {}
                for pool in all_pools:
                    # Extract token symbols
                    token1 = pool.get('token1', {})
                    token2 = pool.get('token2', {})
                    
                    token1_symbol = token1.get('symbol', '') if isinstance(token1, dict) else str(token1)
                    token2_symbol = token2.get('symbol', '') if isinstance(token2, dict) else str(token2)
                    
                    if token1_symbol and len(token1_symbol) > 0:
                        if token1_symbol not in token_pools_map:
                            token_pools_map[token1_symbol] = []
                        token_pools_map[token1_symbol].append(pool)
                        
                    if token2_symbol and len(token2_symbol) > 0:
                        if token2_symbol not in token_pools_map:
                            token_pools_map[token2_symbol] = []
                        token_pools_map[token2_symbol].append(pool)
                        
                # Store in session state for quick access
                st.session_state.token_pools_map = token_pools_map
        except Exception as e:
            logger.error(f"Error getting pools: {e}")
            st.warning("Could not fetch pool data. Using static token list instead.")
            # Use the common token list as fallback
            common_tokens = ["SOL", "USDC", "USDT", "ETH", "BTC", "RAY", "ORCA"]
            token_symbols = set(common_tokens)
        
        # Create a list to hold detailed token data
        token_data_list = []
        
        # Get detailed information for each token
        for symbol in token_symbols:
            try:
                token_data = token_service.get_token_data(symbol)
                if token_data and isinstance(token_data, dict):
                    # Ensure the symbol is in the dictionary
                    if 'symbol' not in token_data:
                        token_data['symbol'] = symbol
                    token_data_list.append(token_data)
            except Exception as e:
                logger.warning(f"Could not get data for token {symbol}: {e}")
                # Add a minimal entry so the token still appears
                token_data_list.append({
                    'symbol': symbol,
                    'name': symbol,
                    'address': '',
                    'price_source': 'unknown',
                    'address_source': 'unknown'
                })
        
        # Convert to DataFrame
        if token_data_list:
            token_df = pd.DataFrame(token_data_list)
        else:
            # Create empty DataFrame with expected columns if no tokens found
            token_df = pd.DataFrame(columns=['symbol', 'name', 'address', 'price_source', 'address_source'])
        
        # Add price data
        try:
            price_data = get_multiple_prices(list(token_symbols))
            
            # Ensure symbol column exists
            if 'symbol' not in token_df.columns or token_df.empty:
                token_df['symbol'] = list(token_symbols)
                
            # Add name column if it doesn't exist
            if 'name' not in token_df.columns:
                token_df['name'] = token_df['symbol']
            
            # Add price to dataframe with source information
            token_df['current_price'] = token_df['symbol'].apply(
                lambda x: price_data.get(x, 0) if x in price_data else 0
            )
        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            # Ensure we have a current_price column even if price fetching fails
            if 'current_price' not in token_df.columns:
                token_df['current_price'] = 0
        
        # Add price_source and address_source columns if they don't exist
        if 'price_source' not in token_df.columns:
            token_df['price_source'] = 'coingecko'  # Default source
        
        if 'address_source' not in token_df.columns:
            token_df['address_source'] = 'coingecko'  # Default source
        
        # Log the outcome
        if not token_df.empty:
            logger.info(f"Successfully loaded data for {len(token_df)} tokens")
        else:
            logger.warning("No token data was loaded")
            
        return token_df
    except Exception as e:
        logger.error(f"Error loading token data: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        
        # Create a minimal dataframe with common tokens to avoid complete failure
        common_tokens = ["SOL", "USDC", "USDT", "ETH", "BTC"]
        backup_data = [{
            'symbol': symbol,
            'name': symbol,
            'current_price': 0,
            'price_source': 'unknown',
            'address_source': 'unknown'
        } for symbol in common_tokens]
        
        return pd.DataFrame(backup_data)

# Function to load liquidity pools for a specific token
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_token_pools(token_symbol: str):
    try:
        pools = []
        
        # First check session state for cached pool data
        if 'token_pools_map' in st.session_state and token_symbol in st.session_state.token_pools_map:
            logger.info(f"Using cached pool data for {token_symbol}")
            pools = st.session_state.token_pools_map[token_symbol]
            
        if not pools or len(pools) == 0:
            # Try data service next
            try:
                pools = data_service.get_pools_by_token(token_symbol)
                logger.info(f"Got {len(pools) if pools else 0} pools for {token_symbol} from data service")
            except Exception as e:
                logger.error(f"Error getting pools from data service: {e}")
        
        if not pools or len(pools) == 0:
            # Fall back to token service if needed
            try:
                pools = token_service.get_pools_for_token(token_symbol)
                logger.info(f"Got {len(pools) if pools else 0} pools for {token_symbol} from token service")
            except Exception as e:
                logger.error(f"Error getting pools from token service: {e}")
                
        if not pools or len(pools) == 0:
            logger.warning(f"No pools found for token {token_symbol}")
            return pd.DataFrame()
        
        # Process pools to ensure consistent format
        processed_pools = []
        for pool in pools:
            # Convert token objects to symbols if needed
            token1 = pool.get('token1', {})
            token2 = pool.get('token2', {})
            
            token1_symbol = token1.get('symbol', '') if isinstance(token1, dict) else str(token1)
            token2_symbol = token2.get('symbol', '') if isinstance(token2, dict) else str(token2)
            
            # Create clean pool object
            clean_pool = {
                'id': pool.get('pool_id', pool.get('id', '')),
                'dex': pool.get('dex', 'Unknown'),
                'token1_symbol': token1_symbol,
                'token2_symbol': token2_symbol
            }
            
            # Extract and normalize liquidity value
            try:
                liquidity = pool.get('liquidity', 0)
                if isinstance(liquidity, dict):
                    clean_pool['liquidity'] = float(liquidity.get('usd', 0))
                else:
                    clean_pool['liquidity'] = float(liquidity) if liquidity else 0
            except (TypeError, ValueError):
                clean_pool['liquidity'] = 0
            
            # Extract and normalize volume data
            try:
                volume = pool.get('volume', {})
                if isinstance(volume, dict):
                    clean_pool['volume_24h'] = float(volume.get('h24', volume.get('usd', 0)))
                else:
                    clean_pool['volume_24h'] = float(volume) if volume else 0
            except (TypeError, ValueError):
                clean_pool['volume_24h'] = 0
            
            # Add APR if available
            try:
                apr = pool.get('apr', {})
                if isinstance(apr, dict):
                    clean_pool['apr'] = float(apr.get('total', apr.get('day', 0))) * 100  # Convert to percentage
                else:
                    clean_pool['apr'] = float(apr) * 100 if apr else 0
            except (TypeError, ValueError):
                clean_pool['apr'] = 0
            
            processed_pools.append(clean_pool)
        
        logger.info(f"Loaded {len(processed_pools)} pools for token {token_symbol}")
        return pd.DataFrame(processed_pools)
    except Exception as e:
        logger.error(f"Error loading pools for {token_symbol}: {e}")
        return pd.DataFrame()

# Function to generate token network visualization
def generate_token_network(token_symbol: str, pool_df: pd.DataFrame):
    if pool_df.empty:
        return None
        
    # Create nodes for each unique token
    nodes = set([token_symbol])
    for _, row in pool_df.iterrows():
        if 'token1_symbol' in row and row['token1_symbol']:
            nodes.add(row['token1_symbol'])
        if 'token2_symbol' in row and row['token2_symbol']:
            nodes.add(row['token2_symbol'])
    
    # Create node dataframe
    node_df = pd.DataFrame({
        'id': list(nodes),
        'name': list(nodes),
        'size': [20 if node == token_symbol else 15 for node in nodes]
    })
    
    # Create edges
    edges = []
    for _, row in pool_df.iterrows():
        if 'token1_symbol' in row and 'token2_symbol' in row:
            token1 = row['token1_symbol']
            token2 = row['token2_symbol']
            if token1 and token2:
                # Add bidirectional edges
                edges.append((token1, token2, row.get('liquidity', 0)))
                edges.append((token2, token1, row.get('liquidity', 0)))
    
    edge_df = pd.DataFrame(edges, columns=['source', 'target', 'weight'])
    
    # Filter out invalid edges
    edge_df = edge_df[edge_df['source'].isin(nodes) & edge_df['target'].isin(nodes)]
    
    if edge_df.empty:
        return None
    
    # Normalize edge weights for visualization
    max_weight = edge_df['weight'].max()
    if max_weight > 0:
        edge_df['width'] = 1 + 5 * edge_df['weight'] / max_weight
    else:
        edge_df['width'] = 1
    
    # Create the network visualization
    fig = go.Figure()
    
    # Add edges
    for _, edge in edge_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[edge['source'], edge['target']], 
                y=[0, 0],
                mode='lines',
                line=dict(width=edge['width'], color='rgba(100,100,100,0.7)'),
                hoverinfo='none',
                showlegend=False
            )
        )
    
    # Add nodes
    for _, node in node_df.iterrows():
        is_main = node['id'] == token_symbol
        fig.add_trace(
            go.Scatter(
                x=[node['id']], 
                y=[0],
                mode='markers+text',
                marker=dict(
                    size=node['size'],
                    color='#2E86C1' if is_main else '#85C1E9',
                    line=dict(width=2, color='white')
                ),
                text=[node['name']],
                textposition='top center',
                name=node['name'],
                hoverinfo='text',
                hovertext=f"Token: {node['name']}"
            )
        )
    
    # Update the layout
    fig.update_layout(
        title=f"Token Relationship Network for {token_symbol}",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        template="plotly_white"
    )
    
    return fig

# Function to create price history chart
def create_price_chart(token_symbol: str):
    try:
        # Get current price as a reference
        price_result = get_token_price(token_symbol, return_source=True)
        
        # Handle different return types
        if isinstance(price_result, tuple):
            current_price, source = price_result
        else:
            current_price = price_result
            source = "Unknown"
            
        # Show a nice price chart with the current price
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        # Create a simple chart with just the current price
        # This shows a flat line at the current price
        if current_price and current_price > 0:
            prices = [current_price] * len(dates)
            source_text = f"Current price from {source if source else 'Unknown'}"
        else:
            # No price available
            prices = [0] * len(dates)
            current_price = 0
            source_text = "No price data available"
        
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices
        })
        
        # Create a simple chart
        fig = px.line(
            df, 
            x='Date', 
            y='Price',
            title=f"{token_symbol} - ${current_price:.6f} USD"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=300,
            margin=dict(l=0, r=10, t=30, b=0)
        )
        
        return fig, source_text
        
    except Exception as e:
        logger.error(f"Error creating price chart for {token_symbol}: {e}")
        
        # Create an empty chart as fallback
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=f"Price data unavailable for {token_symbol}",
            height=300
        )
        
        return empty_fig, "No price data available"

# This helper function has been integrated into create_price_chart

# Main app layout
st.title("🪙 Token Analysis")
st.markdown("Explore detailed information about tokens trading on Solana DEXes")

# Status indicator in sidebar
with st.sidebar:
    st.subheader("Status")
    status_container = st.empty()

# Load token data with simpler, more reliable approach
try:
    with st.spinner("Loading token data..."):
        try:
            # First try to get data from the full loader
            token_df = load_token_data()
            if not token_df.empty:
                status_container.success(f"Loaded data for {len(token_df)} tokens")
            else:
                raise ValueError("No token data returned from loader")
        except Exception as e:
            logger.error(f"Error in primary token loading: {e}")
            
            # Create a guaranteed fallback with common tokens
            common_tokens = ["SOL", "USDC", "USDT", "ETH", "BTC", "MSOL", "BONK", "RAY"]
            
            # Directly create tokens with known names
            token_names = {
                "SOL": "Solana", 
                "USDC": "USD Coin", 
                "USDT": "Tether", 
                "ETH": "Ethereum", 
                "BTC": "Bitcoin",
                "MSOL": "Marinade Staked SOL",
                "BONK": "Bonk",
                "RAY": "Raydium"
            }
            
            # Create fallback DataFrame
            token_df = pd.DataFrame({
                'symbol': common_tokens,
                'name': [token_names.get(s, s) for s in common_tokens],
                'current_price': [0] * len(common_tokens),
                'price_source': ['unknown'] * len(common_tokens),
                'address_source': ['unknown'] * len(common_tokens)
            })
            
            # Try to get real prices in a very reliable way
            try:
                # Try one token at a time to avoid rate limiting
                for i, symbol in enumerate(common_tokens):
                    try:
                        price_result = get_token_price(symbol, return_source=True)
                        # Handle different return types
                        if isinstance(price_result, tuple):
                            price, source = price_result
                        else:
                            price = price_result
                            source = "Unknown"
                            
                        if price and price > 0:
                            token_df.at[i, 'current_price'] = price
                            token_df.at[i, 'price_source'] = source
                    except Exception:
                        # Just continue with the next token if one fails
                        pass
                        
                # Check if we got any prices
                if any(token_df['current_price'] > 0):
                    status_container.warning("Using limited token data with available prices")
                else:
                    status_container.error("No token data available. Using static token list.")
            except Exception as price_err:
                logger.error(f"Error getting fallback prices: {price_err}")
                status_container.error("Unable to load price data. Using static token list.")
    
    # Always ensure we have a token_df with at least these tokens to display
    if token_df.empty:
        token_df = pd.DataFrame({
            'symbol': ["SOL", "USDC", "USDT", "ETH", "BTC"],
            'name': ["Solana", "USD Coin", "Tether", "Ethereum", "Bitcoin"],
            'current_price': [0, 0, 0, 0, 0],
            'price_source': ['fallback'] * 5,
            'address_source': ['fallback'] * 5
        })
        
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Token Explorer", "DEX Categorization", "Data Sources"])    
except Exception as e:
    st.error(f"An error occurred while loading token data: {str(e)}")
    logger.error(f"Critical error in Token Analysis page: {traceback.format_exc()}")
    # Minimal UI even in case of catastrophic failure
    tab1, tab2, tab3 = st.tabs(["Token Explorer", "DEX Categorization", "Data Sources"])  
    token_df = pd.DataFrame()  # Empty DataFrame
    
    with tab1:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Token selection
            st.subheader("Select Token")
            
            # Filter options
            price_filter = st.checkbox("Show only tokens with price data")
            
            if price_filter:
                filtered_df = token_df[token_df['current_price'] > 0]
            else:
                filtered_df = token_df
            
            # Sort options
            sort_options = ["Symbol (A-Z)", "Name (A-Z)", "Price (High to Low)"]
            sort_by = st.selectbox("Sort by", sort_options)
            
            if sort_by == "Symbol (A-Z)":
                filtered_df = filtered_df.sort_values(by="symbol")
            elif sort_by == "Name (A-Z)":
                filtered_df = filtered_df.sort_values(by="name")
            elif sort_by == "Price (High to Low)":
                filtered_df = filtered_df.sort_values(by="current_price", ascending=False)
            
            # Search box
            search_query = st.text_input("Search tokens", "")
            if search_query:
                filtered_df = filtered_df[
                    filtered_df['symbol'].str.contains(search_query, case=False, na=False) | 
                    filtered_df['name'].str.contains(search_query, case=False, na=False)
                ]
            
            # Token list with prices
            if not filtered_df.empty:
                # Create a selection dataframe with symbol and price
                display_df = filtered_df[['symbol', 'name', 'current_price']].copy()
                display_df['display_name'] = display_df.apply(
                    lambda x: f"{x['symbol']} - {x['current_price']:.6f} USD" if x['current_price'] > 0 else x['symbol'], 
                    axis=1
                )
                
                # Create a selection box with token symbols
                selected_token_display = st.selectbox(
                    "Choose a token",
                    display_df['display_name'].tolist()
                )
                
                # Extract the symbol from the selection
                if selected_token_display:
                    selected_token = selected_token_display.split(' - ')[0]
                else:
                    selected_token = None
            else:
                st.warning("No tokens match your filters")
                selected_token = None
        
        with col2:
            if selected_token:
                # Get token details
                token_data = token_service.get_token_data(selected_token)
                if token_data:
                    # Get price with source
                    price_result = get_token_price(selected_token, return_source=True)
                    
                    # Handle different return types
                    if isinstance(price_result, tuple):
                        price, price_source = price_result
                    else:
                        price = price_result
                        price_source = "Unknown"
                    
                    st.subheader(f"{token_data.get('name', selected_token)} ({selected_token})")
                    
                    # Load pools for this token to calculate metrics
                    with st.spinner(f"Loading metrics for {selected_token}..."):
                        pool_df = load_token_pools(selected_token)
                    
                    # Calculate token metrics
                    total_tvl = 0
                    total_volume = 0
                    dex_list = set()
                    pool_count = 0
                    
                    if not pool_df.empty:
                        total_tvl = pool_df['liquidity'].sum() if 'liquidity' in pool_df.columns else 0
                        total_volume = pool_df['volume_24h'].sum() if 'volume_24h' in pool_df.columns else 0
                        dex_list = set(pool_df['dex'].unique()) if 'dex' in pool_df.columns else set()
                        pool_count = len(pool_df)
                    
                    # Token details in columns
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown(f"**Price:** ${price:.6f} USD")
                        st.markdown(f"**Price Source:** {price_source.capitalize() if price_source else 'Unknown'}")
                        
                        # Show address with copy functionality
                        address = token_data.get('address', 'Unknown')
                        if address and len(address) > 10:
                            short_address = address[:6] + '...' + address[-4:]
                            st.markdown(f"**Address:** {short_address}")
                        else:
                            st.markdown(f"**Address:** {address}")
                        
                        st.markdown(f"**Address Source:** {token_data.get('address_source', 'Unknown')}")
                    
                    with detail_col2:
                        # Show liquidity pools info
                        st.markdown(f"**Total TVL:** ${total_tvl:,.2f} USD")
                        st.markdown(f"**24h Volume:** ${total_volume:,.2f} USD")
                        st.markdown(f"**Pool Count:** {pool_count}")
                        
                        if dex_list:
                            st.markdown(f"**Found on:** {', '.join(sorted(dex_list))}")
                        else:
                            st.markdown("**Found on:** No DEX information available")
                    
                    # Price history chart
                    st.subheader("Price History")
                    price_chart, source_text = create_price_chart(selected_token)
                    st.plotly_chart(price_chart, use_container_width=True)
                    st.caption(source_text)
                    
                    # Load pools for this token
                    with st.spinner(f"Loading liquidity pools for {selected_token}..."):
                        pool_df = load_token_pools(selected_token)
                    
                    if not pool_df.empty:
                        st.subheader("Liquidity Pools")
                        
                        # Display pool data with APR information
                        if 'apr' in pool_df.columns:
                            display_pools = pool_df[['dex', 'token1_symbol', 'token2_symbol', 'liquidity', 'volume_24h', 'apr']].copy()
                        else:
                            display_pools = pool_df[['dex', 'token1_symbol', 'token2_symbol', 'liquidity', 'volume_24h']].copy()
                            
                        # Rename columns for display
                        column_map = {
                            'dex': 'DEX',
                            'token1_symbol': 'Token 1',
                            'token2_symbol': 'Token 2',
                            'liquidity': 'Liquidity (USD)',
                            'volume_24h': '24h Volume (USD)',
                            'apr': 'APR (%)'
                        }
                        display_pools = display_pools.rename(columns=column_map)
                        
                        # Format numeric columns
                        for col in ['Liquidity (USD)', '24h Volume (USD)']:
                            if col in display_pools.columns:
                                display_pools[col] = display_pools[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) and x > 0 else "-")
                                
                        # Format APR column if it exists
                        if 'APR (%)' in display_pools.columns:
                            display_pools['APR (%)'] = display_pools['APR (%)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) and x > 0 else "-")
                                
                        st.dataframe(display_pools, use_container_width=True)
                        
                        # Network visualization
                        st.subheader("Token Relationships")
                        network_fig = generate_token_network(selected_token, pool_df)
                        if network_fig:
                            st.plotly_chart(network_fig, use_container_width=True)
                        else:
                            st.info("Not enough data to create a token relationship network")
                    else:
                        st.info(f"No liquidity pools found for {selected_token}")
                else:
                    st.error(f"Unable to load details for {selected_token}")
            else:
                st.info("Please select a token to view its details")
    
    with tab2:
        st.subheader("Tokens by DEX")
        
        # Get all pools to build DEX categorization
        with st.spinner("Loading DEX data..."):
            try:
                # Get all pools directly from the data service to guarantee we use only real data
                all_pools = data_service.get_all_pools()
                
                if not all_pools or len(all_pools) == 0:
                    st.warning("No pool data available for DEX categorization.")
                    # Create some minimal example data for display purposes
                    all_pools = [
                        {"dex": "Raydium", "token1": {"symbol": "SOL"}, "token2": {"symbol": "USDC"}},
                        {"dex": "Orca", "token1": {"symbol": "SOL"}, "token2": {"symbol": "USDT"}},
                        {"dex": "Jupiter", "token1": {"symbol": "SOL"}, "token2": {"symbol": "ETH"}}
                    ]
                
                # Count pools by DEX
                pool_counts = {}
                
                # Group tokens by DEX
                dex_tokens = {}
                dex_pool_data = {}
                
                # Process each pool
                for pool in all_pools:
                    dex = pool.get('dex', 'Unknown')
                    pool_id = pool.get('pool_id', pool.get('id', 'unknown'))
                    
                    # Count pools by DEX
                    if dex not in pool_counts:
                        pool_counts[dex] = 0
                    pool_counts[dex] += 1
                    
                    # Store pool data by DEX for more detailed analysis
                    if dex not in dex_pool_data:
                        dex_pool_data[dex] = []
                    dex_pool_data[dex].append(pool)
                    
                    # Extract token symbols
                    token1 = pool.get('token1', {})
                    token2 = pool.get('token2', {})
                    
                    token1_symbol = token1.get('symbol', '') if isinstance(token1, dict) else str(token1)
                    token2_symbol = token2.get('symbol', '') if isinstance(token2, dict) else str(token2)
                    
                    # Add to dex tokens mapping
                    if token1_symbol and len(token1_symbol) > 0:
                        if dex not in dex_tokens:
                            dex_tokens[dex] = set()
                        dex_tokens[dex].add(token1_symbol)
                        
                    if token2_symbol and len(token2_symbol) > 0:
                        if dex not in dex_tokens:
                            dex_tokens[dex] = set()
                        dex_tokens[dex].add(token2_symbol)
                
                # Convert sets to lists
                dex_tokens = {dex: list(tokens) for dex, tokens in dex_tokens.items()}
                
                logger.info(f"Loaded token categorization for {len(dex_tokens)} DEXes")
                
                # Calculate total TVL by DEX
                dex_tvl = {}
                for dex, pools in dex_pool_data.items():
                    total_liquidity = 0
                    for pool in pools:
                        liquidity = pool.get('liquidity', 0)
                        if isinstance(liquidity, dict):
                            total_liquidity += float(liquidity.get('usd', 0))
                        else:
                            total_liquidity += float(liquidity) if liquidity else 0
                    dex_tvl[dex] = total_liquidity
                    
            except Exception as e:
                logger.error(f"Error loading DEX data: {e}")
                dex_tokens = {}
                pool_counts = {}
                dex_tvl = {}
        
        # Display tokens by DEX
        if dex_tokens:
            col1, col2 = st.columns(2)
            
            with col1:
                # DEX selection
                available_dexes = sorted(list(dex_tokens.keys()))
                selected_dex = st.selectbox("Select DEX", available_dexes)
                
                if selected_dex and selected_dex in dex_tokens:
                    # Display token count
                    st.markdown(f"**{len(dex_tokens[selected_dex])} tokens** found on {selected_dex}")
                    
                    # Create a table of tokens with their prices
                    dex_token_df = token_df[token_df['symbol'].isin(dex_tokens[selected_dex])].copy()
                    dex_token_df = dex_token_df[['symbol', 'name', 'current_price', 'address_source']].sort_values(by='symbol')
                    dex_token_df.columns = ['Symbol', 'Name', 'Price (USD)', 'Address Source']
                    
                    # Format price column
                    dex_token_df['Price (USD)'] = dex_token_df['Price (USD)'].apply(lambda x: f"${x:.6f}" if pd.notnull(x) and x > 0 else "-")
                    
                    st.dataframe(dex_token_df, use_container_width=True)
            
            with col2:
                # DEX comparison
                st.subheader("DEX Comparison")
                
                # Prepare comparison data
                comparison_data = []
                for dex in dex_tokens.keys():
                    entry = {
                        'DEX': dex,
                        'Token Count': len(dex_tokens.get(dex, [])),
                        'Pool Count': pool_counts.get(dex, 0),
                        'TVL (USD)': dex_tvl.get(dex, 0)
                    }
                    comparison_data.append(entry)
                
                # Create dataframe
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values(by='TVL (USD)', ascending=False)
                
                # Format the TVL column for display
                display_df = comparison_df.copy()
                display_df['TVL (USD)'] = display_df['TVL (USD)'].apply(
                    lambda x: f"${x:,.2f}" if pd.notnull(x) and x > 0 else "-"
                )
                
                # Show the table
                st.dataframe(display_df, use_container_width=True)
                
                # Create a bar chart for token count
                fig = px.bar(
                    comparison_df, 
                    x='DEX', 
                    y=['Token Count', 'Pool Count'],
                    title='Tokens and Pools by DEX',
                    barmode='group',
                    color_discrete_sequence=['#2E86C1', '#85C1E9']
                )
                
                fig.update_layout(
                    xaxis_title="DEX",
                    yaxis_title="Count",
                    height=400,
                    legend_title="Metric"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a pie chart for TVL
                if any(comparison_df['TVL (USD)'] > 0):
                    fig2 = px.pie(
                        comparison_df, 
                        values='TVL (USD)', 
                        names='DEX',
                        title='Total Value Locked (TVL) by DEX',
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    
                    fig2.update_traces(textposition='inside', textinfo='percent+label')
                    fig2.update_layout(height=400)
                    
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No DEX categorization data available")
    
    with tab3:
        st.subheader("Data Sources")
        st.markdown("This tool integrates data from multiple sources to provide comprehensive token information:")
        
        st.markdown("### Price Data Sources")
        st.markdown("- **CoinGecko API**: Primary source for token prices, with proper API key integration for reliable data")
        st.markdown("- **DeFi API**: Secondary source for tokens not found in CoinGecko")
        
        st.markdown("### Address Sources")
        st.markdown("- **CoinGecko**: Token addresses are sourced primarily from CoinGecko's comprehensive database")
        st.markdown("- **Manual Mappings**: Some token addresses are manually mapped for tokens not found in public APIs")
        
        st.markdown("### DEX and Pool Data")
        st.markdown("- **Raydium**: Liquidity pool data from Raydium DEX")
        st.markdown("- **Other Solana DEXes**: Additional pool data from other Solana DEXes")
        
        # Show data source stats
        st.subheader("Data Source Statistics")
        
        # Count tokens by price source
        price_sources = token_df['price_source'].value_counts().reset_index()
        price_sources.columns = ['Source', 'Count']
        
        # Count tokens by address source
        address_sources = token_df['address_source'].value_counts().reset_index()
        address_sources.columns = ['Source', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Sources")
            if not price_sources.empty:
                fig = px.pie(
                    price_sources, 
                    values='Count', 
                    names='Source',
                    title='Token Price Sources',
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No price source data available")
        
        with col2:
            st.markdown("#### Address Sources")
            if not address_sources.empty:
                fig = px.pie(
                    address_sources, 
                    values='Count', 
                    names='Source',
                    title='Token Address Sources',
                    color_discrete_sequence=px.colors.sequential.Greens
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No address source data available")
        
        # Display raw data option
        if st.checkbox("Show raw token data"):
            st.dataframe(token_df)
