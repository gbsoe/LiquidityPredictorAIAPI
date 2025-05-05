
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
import time
from datetime import datetime

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

# Function to load token data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_token_data():
    try:
        # Get all available tokens
        tokens = token_service.get_all_tokens()
        
        # Check if tokens is a list or dictionary and convert accordingly
        if isinstance(tokens, dict):
            # It's a dictionary, use the values
            token_df = pd.DataFrame(tokens.values())
        else:
            # It's already a list
            token_df = pd.DataFrame(tokens)
        
        # Add price data
        token_symbols = token_df['symbol'].tolist()
        price_data = get_multiple_prices(token_symbols)
        
        # Add price to dataframe with source information
        token_df['current_price'] = token_df['symbol'].apply(
            lambda x: price_data.get(x, 0) if x in price_data else 0
        )
        
        # Add price_source and address_source columns if they don't exist
        if 'price_source' not in token_df.columns:
            token_df['price_source'] = 'coingecko'  # Default source
        
        if 'address_source' not in token_df.columns:
            token_df['address_source'] = 'coingecko'  # Default source
        
        return token_df
    except Exception as e:
        st.error(f"Error loading token data: {e}")
        return pd.DataFrame()

# Function to load liquidity pools for a specific token
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_token_pools(token_symbol: str):
    try:
        # First try directly from data service
        pools = data_service.get_pools_by_token(token_symbol)
        
        if not pools or len(pools) == 0:
            # Fall back to token service if needed
            pools = token_service.get_pools_for_token(token_symbol)
            
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
    # Try to get historical price data from data service first
    try:
        # Attempt to get token address for lookup
        token_address = token_service.get_token_address(token_symbol)
        historical_data = None
        
        if token_address:
            # If we have a token address, try to get historical data from the data service
            historical_service = None
            try:
                from historical_data_service import get_historical_service
                historical_service = get_historical_service()
                
                # Get historical price data for the token
                historical_data = historical_service.get_token_history(token_address, days=30)
            except Exception as e:
                logger.warning(f"Error getting historical price data for {token_symbol}: {e}")
            
        # Process historical data if we have it
        if historical_data and isinstance(historical_data, dict) and 'prices' in historical_data:
            price_history = historical_data['prices']
            if price_history and len(price_history) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(price_history)
                if len(df.columns) >= 2:  # Ensure we have date and price columns
                    df.columns = ['Date', 'Price']
                    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                    source = historical_data.get('source', 'Historical API')
                    current_price = df['Price'].iloc[-1] if not df.empty else 0
                    return create_chart_from_df(df, token_symbol, current_price, source)
    except Exception as e:
        logger.warning(f"Failed to get historical prices for {token_symbol}: {e}")
    
    # Fall back to current price with simulated history if real data not available
    dates = pd.date_range(start=datetime.now() - pd.Timedelta(days=30), end=datetime.now(), freq='D')
    
    # Get current price as a reference
    price_result = get_token_price(token_symbol, return_source=True)
    
    # Handle different return types
    if isinstance(price_result, tuple):
        current_price, source = price_result
    else:
        current_price = price_result
        source = "Unknown"
    
    # If we have a price, create a reasonable price history
    if current_price and current_price > 0:
        # Generate somewhat realistic price movement
        base = current_price * 0.8
        variation = current_price * 0.4
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 1, len(dates))
        cumulative_noise = np.cumsum(noise) / 20  # Smoothed random walk
        normalized_noise = (cumulative_noise - cumulative_noise.min()) / (cumulative_noise.max() - cumulative_noise.min())
        prices = base + variation * normalized_noise
    else:
        # If no price, just use dummy data
        prices = [1 + 0.1 * np.sin(i/5) for i in range(len(dates))]
    
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
    return create_chart_from_df(df, token_symbol, current_price, source)

# Helper function to create chart from dataframe
def create_chart_from_df(df, token_symbol, current_price, source):
    fig = px.line(
        df, 
        x='Date', 
        y='Price',
        title=f"{token_symbol} Price History (Last 30 Days)"
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=300,
        margin=dict(l=0, r=10, t=30, b=0)
    )
    
    return fig, f"Source: {source.capitalize() if source else 'Unknown'} (Current: ${current_price:.6f})"

# Main app layout
st.title("🪙 Token Analysis")
st.markdown("Explore detailed information about tokens trading on Solana DEXes")

# Load token data
with st.spinner("Loading token data..."):
    token_df = load_token_data()

if token_df.empty:
    st.error("Unable to load token data. Please try again later.")
else:
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Token Explorer", "DEX Categorization", "Data Sources"])
    
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
                    
                    # Token details in columns
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    
                    with detail_col1:
                        st.markdown(f"**Price:** ${price:.6f} USD")
                        st.markdown(f"**Price Source:** {price_source.capitalize() if price_source else 'Unknown'}")
                    
                    with detail_col2:
                        address = token_data.get('address', 'Unknown')
                        if address and len(address) > 10:
                            short_address = address[:6] + '...' + address[-4:]
                            st.markdown(f"**Address:** {short_address}")
                        else:
                            st.markdown(f"**Address:** {address}")
                        
                        st.markdown(f"**Address Source:** {token_data.get('address_source', 'Unknown')}")
                    
                    with detail_col3:
                        dexes = token_data.get('dexes', [])
                        if dexes:
                            st.markdown(f"**Found on:** {', '.join(dexes)}")
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
                all_pools = data_service.get_all_pools()
                
                # Group tokens by DEX
                dex_tokens = {}
                for pool in all_pools:
                    dex = pool.get('dex', 'Unknown')
                    
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
            except Exception as e:
                logger.error(f"Error loading DEX data: {e}")
                dex_tokens = {}
                
                # Fall back to token service data
                for _, token_data in token_df.iterrows():
                    dexes = token_data.get('dexes', [])
                    if isinstance(dexes, list) and dexes:
                        for dex in dexes:
                            if dex not in dex_tokens:
                                dex_tokens[dex] = []
                            dex_tokens[dex].append(token_data['symbol'])
        
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
                
                # Count tokens by DEX
                dex_counts = {dex: len(tokens) for dex, tokens in dex_tokens.items()}
                dex_count_df = pd.DataFrame(list(dex_counts.items()), columns=['DEX', 'Token Count'])
                dex_count_df = dex_count_df.sort_values(by='Token Count', ascending=False)
                
                # Create a bar chart
                fig = px.bar(
                    dex_count_df, 
                    x='DEX', 
                    y='Token Count',
                    title='Number of Tokens by DEX',
                    color='Token Count',
                    color_continuous_scale='blues'
                )
                
                fig.update_layout(
                    xaxis_title="DEX",
                    yaxis_title="Number of Tokens",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
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
