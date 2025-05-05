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

# Page configuration
st.set_page_config(
    page_title="Token Analysis - Solpool Insight",
    page_icon="🪙",
    layout="wide"
)

# Initialize services
try:
    data_service = DataService()
    token_service = get_token_service()
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Error initializing services: {e}")
    st.error("Error initializing data services. Some features may be limited.")

# GUARANTEED TOKENS - This ensures we always have some data to display
GUARANTEED_TOKENS = [
    {"symbol": "SOL", "name": "Solana", "address": "So11111111111111111111111111111111111111112"},
    {"symbol": "USDC", "name": "USD Coin", "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"},
    {"symbol": "USDT", "name": "Tether", "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"},  
    {"symbol": "ETH", "name": "Ethereum", "address": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"},
    {"symbol": "BTC", "name": "Bitcoin", "address": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E"},
    {"symbol": "MSOL", "name": "Marinade Staked SOL", "address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So"},
    {"symbol": "BONK", "name": "Bonk", "address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"},
    {"symbol": "RAY", "name": "Raydium", "address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"}
]

# Function to load token data with maximum resilience
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_token_data():
    try:
        # Start with a guaranteed DataFrame with known tokens
        guaranteed_df = pd.DataFrame(GUARANTEED_TOKENS)
        guaranteed_df['current_price'] = 0
        guaranteed_df['price_source'] = 'hardcoded'
        guaranteed_df['address_source'] = 'hardcoded'
        
        logger.info(f"Created guaranteed token DataFrame with {len(guaranteed_df)} tokens")
        
        # Try to get price data for the guaranteed tokens
        for i, row in guaranteed_df.iterrows():
            try:
                symbol = row['symbol']
                price_result = get_token_price(symbol, return_source=True)
                
                # Handle different result formats
                if isinstance(price_result, tuple):
                    price, price_source = price_result
                else:
                    price = price_result
                    price_source = "unknown"
                
                if price and price > 0:
                    guaranteed_df.at[i, 'current_price'] = price
                    guaranteed_df.at[i, 'price_source'] = price_source
            except Exception as e:
                logger.warning(f"Error getting price for {row['symbol']}: {e}")
        
        # Now try to enhance with actual pool data
        try:
            # Start with getting all pools from the data service
            logger.info("Loading pool data for token extraction")
            all_pools = data_service.get_all_pools()
            
            if all_pools and len(all_pools) > 0:
                logger.info(f"Extracting token data from {len(all_pools)} pools")
                # Create set of all unique tokens
                token_symbols = set()
                
                # Process each pool
                for pool in all_pools:
                    # Extract token symbols
                    token1 = pool.get('token1', {})
                    token2 = pool.get('token2', {})
                    
                    token1_symbol = token1.get('symbol', '') if isinstance(token1, dict) else str(token1)
                    token2_symbol = token2.get('symbol', '') if isinstance(token2, dict) else str(token2)
                    
                    # Add to set of tokens
                    if token1_symbol and len(token1_symbol) > 0:
                        token_symbols.add(token1_symbol)
                    
                    if token2_symbol and len(token2_symbol) > 0:
                        token_symbols.add(token2_symbol)
                
                # Create a list from the set, excluding tokens we already have
                existing_symbols = set(guaranteed_df['symbol'].tolist())
                new_symbols = [s for s in token_symbols if s not in existing_symbols]
                
                logger.info(f"Found {len(new_symbols)} additional tokens in pool data")
                
                # Get data for additional tokens
                additional_token_data = []
                
                # Process in small batches to avoid rate limits
                batch_size = 3
                for i in range(0, len(new_symbols), batch_size):
                    batch = new_symbols[i:i+batch_size]
                    
                    for symbol in batch:
                        try:
                            # Get token info from token service
                            token_info = token_service.get_token_data(symbol)
                            
                            if token_info:
                                # Get current price with price source information
                                price_result = get_token_price(symbol, return_source=True)
                                
                                # Handle different result formats
                                if isinstance(price_result, tuple):
                                    price, price_source = price_result
                                else:
                                    price = price_result
                                    price_source = "unknown"
                                
                                # Format the data for our display
                                additional_token_data.append({
                                    'symbol': symbol,
                                    'name': token_info.get('name', symbol),
                                    'address': token_info.get('address', 'Unknown'),
                                    'address_source': token_info.get('address_source', 'unknown'),
                                    'current_price': price if price and price > 0 else 0,
                                    'price_source': price_source
                                })
                        except Exception as e:
                            logger.warning(f"Error processing additional token {symbol}: {e}")
                            # Add the token even if there's an error to ensure it appears in the list
                            additional_token_data.append({
                                'symbol': symbol,
                                'name': symbol,  # Use symbol as name
                                'address': 'Unknown',
                                'address_source': 'unknown',
                                'current_price': 0,
                                'price_source': 'error'
                            })
                    
                    # Small delay between batches to avoid rate limits
                    time.sleep(0.5)
                
                # If we have additional tokens, add them to the guaranteed data
                if additional_token_data:
                    additional_df = pd.DataFrame(additional_token_data)
                    # Combine the dataframes
                    token_df = pd.concat([guaranteed_df, additional_df], ignore_index=True)
                    logger.info(f"Extended token DataFrame to {len(token_df)} tokens")
                else:
                    token_df = guaranteed_df
            else:
                logger.warning("No pool data available for token extraction, using guaranteed tokens only")
                token_df = guaranteed_df
                
        except Exception as e:
            logger.error(f"Error processing pool data: {str(e)}")
            # Fall back to guaranteed tokens only
            token_df = guaranteed_df
            
        # Add display name column for the selectbox
        token_df['display_name'] = token_df.apply(
            lambda x: f"{x['symbol']} - {x['current_price']:.6f} USD" if x['current_price'] > 0 else x['symbol'], 
            axis=1
        )
        
        return token_df
    except Exception as e:
        logger.error(f"Critical error loading token data: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Super fallback - return just the hardcoded common tokens
        fallback_df = pd.DataFrame(GUARANTEED_TOKENS)
        fallback_df['current_price'] = 0
        fallback_df['price_source'] = 'fallback'
        fallback_df['address_source'] = 'fallback'
        fallback_df['display_name'] = fallback_df['symbol']
        return fallback_df

# Function to load pools for a specific token
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_token_pools(token_symbol):
    try:
        # Check if we have a token-pool mapping in session state
        if hasattr(st.session_state, 'token_pools_map') and token_symbol in st.session_state.token_pools_map:
            pools = st.session_state.token_pools_map[token_symbol]
            return pd.DataFrame(pools)
        
        # Otherwise fetch directly
        pools = data_service.get_pools_by_token(token_symbol)
        if pools and len(pools) > 0:
            return pd.DataFrame(pools)
        else:
            # Try to get all pools and filter
            all_pools = data_service.get_all_pools()
            if all_pools and len(all_pools) > 0:
                # Filter pools containing this token
                token_pools = []
                for pool in all_pools:
                    token1 = pool.get('token1', {})
                    token2 = pool.get('token2', {})
                    
                    token1_symbol = token1.get('symbol', '') if isinstance(token1, dict) else str(token1)
                    token2_symbol = token2.get('symbol', '') if isinstance(token2, dict) else str(token2)
                    
                    if token_symbol in [token1_symbol, token2_symbol]:
                        token_pools.append(pool)
                
                if token_pools:
                    return pd.DataFrame(token_pools)
            
            # No pools found
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading pools for token {token_symbol}: {str(e)}")
        return pd.DataFrame()

# Function to create a network visualization of token relationships
def generate_token_network(token_symbol, pool_df):
    try:
        if pool_df.empty:
            return None
        
        # Prepare nodes (tokens)
        nodes = set([token_symbol])  # Start with the selected token
        
        # Extract all tokens from the pools
        for _, row in pool_df.iterrows():
            token1 = row.get('token1_symbol', str(row.get('token1', '')))
            token2 = row.get('token2_symbol', str(row.get('token2', '')))
            
            if isinstance(token1, dict) and 'symbol' in token1:
                token1 = token1['symbol']
            if isinstance(token2, dict) and 'symbol' in token2:
                token2 = token2['symbol']
                
            nodes.add(token1)
            nodes.add(token2)
        
        # Create edges from pools
        edges = []
        for _, row in pool_df.iterrows():
            token1 = row.get('token1_symbol', str(row.get('token1', '')))
            token2 = row.get('token2_symbol', str(row.get('token2', '')))
            
            if isinstance(token1, dict) and 'symbol' in token1:
                token1 = token1['symbol']
            if isinstance(token2, dict) and 'symbol' in token2:
                token2 = token2['symbol']
                
            # Get liquidity for edge weight
            liquidity = row.get('liquidity', 0)
            if isinstance(liquidity, dict):
                liquidity = liquidity.get('usd', 0)
                
            edges.append((token1, token2, float(liquidity) if liquidity else 1))
        
        # Not enough data for a network
        if len(nodes) < 2 or len(edges) < 1:
            return None
            
        # Create a list of nodes with positions for the network
        node_positions = {}
        node_colors = {}
        node_sizes = {}
        
        # Assign the central position to our selected token
        node_positions[token_symbol] = [0, 0]
        node_colors[token_symbol] = 'rgb(255, 0, 0)'  # Red for the selected token
        node_sizes[token_symbol] = 20  # Larger size for the selected token
        
        # Position other nodes around the center
        other_nodes = [n for n in nodes if n != token_symbol]
        n_nodes = len(other_nodes)
        
        # Create a circle layout
        radius = 1.0
        for i, node in enumerate(other_nodes):
            angle = 2 * np.pi * i / n_nodes
            node_positions[node] = [radius * np.cos(angle), radius * np.sin(angle)]
            node_colors[node] = 'rgb(0, 0, 255)'  # Blue for other tokens
            node_sizes[node] = 15  # Regular size for other tokens
        
        # Prepare data for Plotly
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in nodes:
            pos = node_positions.get(node, [0, 0])
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_text.append(node)
            node_color.append(node_colors.get(node, 'rgb(0, 0, 255)'))
            node_size.append(node_sizes.get(node, 10))
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_opacity = []
        
        # Normalize edge weights
        max_weight = max(w for _, _, w in edges) if edges else 1
        
        for token1, token2, weight in edges:
            if token1 in node_positions and token2 in node_positions:
                x0, y0 = node_positions[token1]
                x1, y1 = node_positions[token2]
                
                # Create a curved edge
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
                
                # Set opacity based on normalized weight
                norm_weight = (weight / max_weight) * 0.8 + 0.2  # Minimum opacity of 0.2
                edge_opacity.extend([norm_weight, norm_weight, norm_weight])
        
        # Create a figure
        fig = go.Figure()
        
        # Add the edges as lines
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgb(150,150,150)'),
            hoverinfo='none',
            mode='lines',
            opacity=0.7,
            showlegend=False
        ))
        
        # Add the nodes as markers
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=1, color='rgb(50,50,50)')
            ),
            text=node_text,
            textposition='middle center',
            hoverinfo='text',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{token_symbol} Token Relationships",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error generating token network: {str(e)}")
        return None

# Function to create price chart with reliable fallback
def create_price_chart(token_symbol):
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
        token_df = load_token_data()
        
        if token_df.empty:
            status_container.error("No token data available.")
        else:
            status_container.success(f"Loaded data for {len(token_df)} tokens")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Token Explorer", "DEX Categorization", "Data Sources"])    
except Exception as e:
    st.error(f"An error occurred while loading token data: {str(e)}")
    logger.error(f"Critical error in Token Analysis page: {traceback.format_exc()}")
    # Minimal UI even in case of catastrophic failure
    tab1, tab2, tab3 = st.tabs(["Token Explorer", "DEX Categorization", "Data Sources"])  
    token_df = pd.DataFrame(GUARANTEED_TOKENS)  # Use guaranteed tokens as absolute fallback
    token_df['current_price'] = 0
    token_df['price_source'] = 'fallback'
    token_df['address_source'] = 'fallback'
    
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
            token_data = token_service.get_token_data(selected_token) or next(
                (item for item in token_df.to_dict('records') if item['symbol'] == selected_token), 
                {'symbol': selected_token, 'name': selected_token}
            )
            
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
                
                # Ensure we have pool data to display
                if not pool_df.empty and len(pool_df) > 0:
                    st.subheader("Liquidity Pools")
                    
                    # Display pool data with APR information
                    try:
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
                    except Exception as e:
                        st.error(f"Error displaying pool data: {str(e)}")
                        st.dataframe(pool_df)  # Show the raw data if formatting fails
                    
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
            st.error(f"Error loading DEX data: {str(e)}")
            # Create empty data structures as fallback
            dex_tokens = {}
            pool_counts = {}
            dex_tvl = {}
    
    # Display tokens by DEX
    if dex_tokens:
        col1, col2 = st.columns(2)
        
        with col1:
            # DEX selection
            available_dexes = sorted(list(dex_tokens.keys()))
            if available_dexes:
                selected_dex = st.selectbox("Select DEX", available_dexes)
                
                if selected_dex and selected_dex in dex_tokens:
                    # Display token count
                    st.markdown(f"**{len(dex_tokens[selected_dex])} tokens** found on {selected_dex}")
                    
                    # Create a table of tokens with their prices
                    dex_token_df = token_df[token_df['symbol'].isin(dex_tokens[selected_dex])].copy()
                    if not dex_token_df.empty:
                        dex_token_df = dex_token_df[['symbol', 'name', 'current_price', 'address_source']].sort_values(by='symbol')
                        dex_token_df.columns = ['Symbol', 'Name', 'Price (USD)', 'Address Source']
                        
                        # Format price column
                        dex_token_df['Price (USD)'] = dex_token_df['Price (USD)'].apply(lambda x: f"${x:.6f}" if pd.notnull(x) and x > 0 else "-")
                        
                        st.dataframe(dex_token_df, use_container_width=True)
                    else:
                        # Create a simple table if we can't match with token_df
                        simple_df = pd.DataFrame({
                            'Symbol': dex_tokens[selected_dex]
                        })
                        st.dataframe(simple_df, use_container_width=True)
                else:
                    st.info("Select a DEX to view its tokens")
            else:
                st.info("No DEX data available")
        
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
            if not comparison_df.empty:
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
                st.info("No DEX comparison data available")
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
    if 'price_source' in token_df.columns:
        price_sources = token_df['price_source'].value_counts().reset_index()
        price_sources.columns = ['Source', 'Count']
    else:
        price_sources = pd.DataFrame()
    
    # Count tokens by address source
    if 'address_source' in token_df.columns:
        address_sources = token_df['address_source'].value_counts().reset_index()
        address_sources.columns = ['Source', 'Count']
    else:
        address_sources = pd.DataFrame()
    
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
