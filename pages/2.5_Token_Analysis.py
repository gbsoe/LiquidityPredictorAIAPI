"""
Token Analysis - Comprehensive Token Explorer for SolPool Insight

This page provides detailed analysis for tokens across Solana DEXes,
with comprehensive metadata and visualization of token relationships.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our token service
from token_data_service import get_token_service

# Page configuration
st.set_page_config(
    page_title="Token Analysis - SolPool Insight",
    page_icon="💰",
    layout="wide"
)

# Initialize token service
token_service = get_token_service()

def format_address(address: str, max_length: int = 12) -> str:
    """Format a token address for display"""
    if not address or len(address) <= max_length:
        return address
    return f"{address[:6]}...{address[-4:]}"

def format_price(price: float) -> str:
    """Format a price value for display"""
    if price < 0.01:
        return f"${price:.6f}"
    elif price < 100:
        return f"${price:.2f}"
    else:
        return f"${price:,.2f}"

def get_token_color(token_symbol: str) -> str:
    """Get a consistent color for a token"""
    colors = {
        "SOL": "#14F195",
        "USDC": "#2775CA",
        "RAY": "#9D45FF",
        "mSOL": "#F4B731",
        "BTC": "#F7931A",
        "ETH": "#627EEA"
    }
    return colors.get(token_symbol.upper(), "#AAAAAA")

def main():
    """Main function for the Token Analysis page"""
    st.title("🪙 Token Analysis and DEX Mapping")
    st.markdown("""
    Explore comprehensive token data across Solana DEXes, view token metrics, 
    and analyze relationships between tokens and liquidity pools.
    """)
    
    # Fetch all tokens
    with st.spinner("Loading token data..."):
        tokens = token_service.get_all_tokens()
        
        if not tokens:
            st.error("Failed to load token data. Please check your API connection.")
            st.stop()
    
    # Header metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        active_tokens = [t for t in tokens if t.get("active", False)]
        st.metric("Active Tokens", f"{len(active_tokens):,}")
    
    with col2:
        dex_categories = token_service.get_token_categories()
        total_dexes = len(dex_categories)
        st.metric("DEXes Integrated", f"{total_dexes:,}")
    
    with col3:
        tokens_with_price = [t for t in tokens if t.get("price", 0) > 0]
        st.metric("Tokens With Price", f"{len(tokens_with_price):,}")
    
    # Token Explorer
    st.header("📊 Token Explorer")
    
    # Create tabs for different token views
    tab1, tab2, tab3, tab4 = st.tabs([
        "All Tokens", 
        "DEX Categorization", 
        "Token Relationships", 
        "Token Details"
    ])
    
    with tab1:
        st.subheader("Complete Token Database")
        
        # Process and enrich tokens before creating DataFrame
        enriched_tokens = []
        
        # Import token price service to get direct CoinGecko prices
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from token_price_service import get_token_price
        
        # First, filter out tokens with no symbol
        filtered_tokens = [t for t in tokens if t.get("symbol")]
        
        # Filter out duplicates by symbol
        seen_symbols = set()
        for t in filtered_tokens:
            symbol = t.get("symbol", "").upper()
            if symbol and symbol not in seen_symbols:
                seen_symbols.add(symbol)
                
                # Try to get price from token_service if needed
                if t.get("price", 0) == 0:
                    try:
                        # Force a fresh fetch for the token
                        fresh_token = token_service.get_token_data(symbol, force_refresh=True)
                        if fresh_token.get("price", 0) > 0:
                            t["price"] = fresh_token["price"]
                            t["price_source"] = fresh_token.get("price_source", "defi_api")
                    except Exception as e:
                        # Don't show error, just continue with direct price fetch
                        pass
                
                # If token still doesn't have a price, try CoinGecko directly
                if t.get("price", 0) == 0:
                    try:
                        # Using direct CoinGecko price fetching for tokens shown in UI
                        price, source = get_token_price(symbol, return_source=True)
                        if price and price > 0:
                            t["price"] = float(price)
                            t["price_source"] = source
                    except Exception as e:
                        st.error(f"Error getting CoinGecko price for {symbol}: {e}")
                
                enriched_tokens.append(t)
        
        # Sort tokens: first by price (non-zero first), then by symbol
        enriched_tokens.sort(key=lambda t: (t.get("price", 0) == 0, t.get("symbol", "").upper()))
        
        # Create a DataFrame for display
        token_df = pd.DataFrame([
            {
                "Symbol": t.get("symbol", "").upper(),
                "Name": t.get("name", "Unknown") if t.get("name") else t.get("symbol", "Unknown"),
                "Address": format_address(t.get("address", "")),
                "Decimals": t.get("decimals", 0),
                "Price": format_price(float(t.get("price", 0))),
                "Price Source": "CoinGecko" if t.get("price_source", "") == "coingecko" else ("DeFi API" if t.get("price", 0) > 0 else "None"),
                "Active": "✓" if t.get("active", False) else "✗"
            }
            for t in enriched_tokens
        ])
        
        # Add search functionality
        token_search = st.text_input("Search tokens by symbol or name", "")
        
        # Filter based on search
        if token_search:
            filtered_df = token_df[
                token_df["Symbol"].str.contains(token_search, case=False) |
                token_df["Name"].str.contains(token_search, case=False)
            ]
        else:
            filtered_df = token_df
        
        # Display token count
        st.write(f"Showing {len(filtered_df)} of {len(token_df)} tokens")
        
        # Display the dataframe with a copy button for addresses
        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "Address": st.column_config.TextColumn("Address", width="medium"),
                "Decimals": st.column_config.NumberColumn("Decimals", width="small"),
                "Price": st.column_config.TextColumn("Price", width="small"),
                "Price Source": st.column_config.TextColumn(
                    "Price Source", 
                    width="small",
                    help="Source of the token price data: CoinGecko, DeFi API, or None"
                ),
                "Active": st.column_config.TextColumn("Active", width="small")
            }
        )
    
    with tab2:
        st.subheader("Tokens by DEX")
        
        # Get DEX categories
        dex_categories = token_service.get_token_categories()
        
        # Create a list of DEXes for selection
        dex_list = list(dex_categories.keys())
        
        # Let user select a DEX
        selected_dex = st.selectbox("Select a DEX", dex_list, index=0)
        
        if selected_dex:
            # Get tokens for the selected DEX
            dex_tokens = token_service.get_tokens_by_dex(selected_dex)
            
            # Display token count
            st.write(f"Found {len(dex_tokens)} tokens used by {selected_dex.capitalize()}")
            
            # Enhance tokens with fresh price data if needed
            enhanced_dex_tokens = {}
            for symbol, token in dex_tokens.items():
                # Copy the token to avoid modifying the original
                enhanced_token = token.copy()
                
                # Try to get fresh price if current price is 0
                if enhanced_token.get("price", 0) == 0:
                    try:
                        # Force a fresh fetch for the token
                        fresh_token = token_service.get_token_data(symbol, force_refresh=True)
                        if fresh_token.get("price", 0) > 0:
                            enhanced_token["price"] = fresh_token["price"]
                            enhanced_token["price_source"] = fresh_token.get("price_source", "defi_api")
                    except Exception as e:
                        # Don't show error, just continue with direct price fetch
                        pass
                
                # If token still doesn't have a price, try CoinGecko directly
                if enhanced_token.get("price", 0) == 0:
                    try:
                        # Using direct CoinGecko price fetching for tokens shown in UI
                        price, source = get_token_price(symbol, return_source=True)
                        if price and price > 0:
                            enhanced_token["price"] = float(price)
                            enhanced_token["price_source"] = source
                    except Exception as e:
                        st.error(f"Error getting CoinGecko price for {symbol}: {e}")
                
                enhanced_dex_tokens[symbol] = enhanced_token
            
            # Sort tokens: first by price (non-zero first), then by symbol
            sorted_tokens = sorted(
                enhanced_dex_tokens.items(), 
                key=lambda item: (item[1].get("price", 0) == 0, item[0].upper())
            )
            
            # Create a DataFrame for the tokens
            dex_token_df = pd.DataFrame([
                {
                    "Symbol": symbol,
                    "Name": token.get("name", "Unknown") if token.get("name") else symbol,
                    "Address": format_address(token.get("address", "")),
                    "Decimals": token.get("decimals", 0),
                    "Price": format_price(float(token.get("price", 0))),
                    "Price Source": "CoinGecko" if token.get("price_source", "") == "coingecko" else ("DeFi API" if token.get("price", 0) > 0 else "None"),
                    "Active": "✓" if token.get("active", False) else "✗"
                }
                for symbol, token in sorted_tokens
            ])
            
            # Display the DataFrame
            st.dataframe(
                dex_token_df,
                use_container_width=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Name": st.column_config.TextColumn("Name", width="medium"),
                    "Address": st.column_config.TextColumn("Address", width="medium"),
                    "Decimals": st.column_config.NumberColumn("Decimals", width="small"),
                    "Price": st.column_config.TextColumn("Price", width="small"),
                    "Price Source": st.column_config.TextColumn(
                        "Price Source", 
                        width="small",
                        help="Source of the token price data: CoinGecko, DeFi API, or None"
                    ),
                    "Active": st.column_config.TextColumn("Active", width="small")
                }
            )
            
            # Show token price comparison chart
            if len(dex_tokens) > 1:
                st.subheader(f"Price Comparison - {selected_dex.capitalize()} Tokens")
                
                # Filter tokens with prices
                tokens_with_prices = {symbol: token for symbol, token in dex_tokens.items() 
                                      if token.get("price", 0) > 0}
                
                if tokens_with_prices:
                    # Create data for chart
                    chart_data = pd.DataFrame([
                        {"Token": symbol, "Price (USD)": float(token.get("price", 0))}
                        for symbol, token in tokens_with_prices.items()
                    ])
                    
                    # Exclude tokens with extremely high prices that would skew the chart
                    top_price = chart_data["Price (USD)"].max()
                    if top_price > 1000:
                        chart_data_filtered = chart_data[chart_data["Price (USD)"] < 1000]
                        if not chart_data_filtered.empty:
                            st.info(f"Some high-price tokens have been filtered from the chart for better visualization.")
                            chart_data = chart_data_filtered
                    
                    # Create chart
                    fig = px.bar(
                        chart_data, 
                        x="Token", 
                        y="Price (USD)",
                        color="Token",
                        title=f"Token Prices for {selected_dex.capitalize()}",
                        color_discrete_map={token: get_token_color(token) for token in chart_data["Token"]}
                    )
                    
                    # Customize layout
                    fig.update_layout(
                        xaxis_title="Token Symbol",
                        yaxis_title="Price (USD)",
                        legend_title="Tokens",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No tokens with price data available for chart visualization.")
    
    with tab3:
        st.subheader("Token Relationships")
        
        # This would be a visualization of token relationships, e.g., which tokens are paired in pools
        # Since we don't have the complete pool data here, we'll display a conceptual visualization
        st.write("This visualization shows token relationships across DEXes.")
        
        # Create a simplified network diagram of tokens across DEXes
        nodes = []
        edges = []
        
        # Add DEXes as nodes
        for dex in dex_categories.keys():
            nodes.append({
                "id": dex,
                "label": dex.capitalize(),
                "group": "dex"
            })
        
        # Add tokens as nodes
        for token in tokens:
            symbol = token.get("symbol", "").upper()
            if symbol:
                nodes.append({
                    "id": symbol,
                    "label": symbol,
                    "group": "token",
                    "value": token.get("price", 0.1) or 0.1  # Size based on price, with minimum
                })
        
        # Add edges from DEXes to tokens
        for dex, dex_tokens in dex_categories.items():
            for token_symbol in dex_tokens:
                edges.append({
                    "from": dex,
                    "to": token_symbol,
                    "value": 1
                })
        
        # Initialize visualization placeholder
        st.write("Select tokens of interest to see their relationships:")
        
        # Let user select token(s) to highlight
        token_symbols = [t.get("symbol", "").upper() for t in tokens if t.get("symbol")]
        selected_tokens = st.multiselect(
            "Select tokens to highlight",
            options=token_symbols,
            default=["SOL", "USDC"] if "SOL" in token_symbols and "USDC" in token_symbols else []
        )
        
        if selected_tokens:
            # Create a mapping of DEXes that use the selected tokens
            token_dex_mapping = {}
            for token_symbol in selected_tokens:
                token_dex_mapping[token_symbol] = []
                for dex, dex_tokens in dex_categories.items():
                    if token_symbol in dex_tokens:
                        token_dex_mapping[token_symbol].append(dex)
            
            # Create a Sankey diagram
            source = []
            target = []
            value = []
            label = []
            
            # Define colors
            colors = []
            
            # Add the selected tokens and their DEXes
            for token_symbol, dexes in token_dex_mapping.items():
                token_idx = len(label)
                label.append(token_symbol)
                colors.append(get_token_color(token_symbol))
                
                for dex in dexes:
                    if dex not in label:
                        label.append(dex.capitalize())
                        colors.append("#CCCCCC")  # Grey for DEXes
                    
                    dex_idx = label.index(dex.capitalize())
                    source.append(token_idx)
                    target.append(dex_idx)
                    value.append(1)
            
            # Create the Sankey diagram
            if source and target and value:
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=label,
                        color=colors
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value
                    )
                )])
                
                fig.update_layout(
                    title_text=f"Token to DEX Relationships",
                    font_size=10,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                st.info("""
                The diagram above shows which DEXes use the selected tokens. 
                Thicker lines indicate more token usage across DEXes.
                """)
            else:
                st.warning("Not enough data to create visualization.")
        else:
            st.info("Please select at least one token to visualize relationships.")
    
    with tab4:
        st.subheader("Token Details Explorer")
        
        # Token selection
        token_symbols = [t.get("symbol", "").upper() for t in tokens if t.get("symbol")]
        selected_token = st.selectbox(
            "Select a token for detailed analysis",
            options=token_symbols
        )
        
        if selected_token:
            # Get comprehensive token metadata
            token_metadata = token_service.get_token_metadata(selected_token)
            
            if token_metadata:
                # Create columns for token details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Token Identity")
                    st.write(f"**Symbol:** {token_metadata.get('symbol', '')}")
                    st.write(f"**Name:** {token_metadata.get('name', 'Unknown')}")
                    st.write(f"**Decimals:** {token_metadata.get('decimals', 0)}")
                    
                    # Address with copy button
                    address = token_metadata.get('address', '')
                    if address:
                        st.write(f"**Address:** `{address}`")
                    
                    # DEXes that use this token
                    dexes = token_metadata.get('dexes', [])
                    if dexes:
                        st.write("**Used by DEXes:**")
                        for dex in dexes:
                            st.write(f"- {dex.capitalize()}")
                
                with col2:
                    st.write("### Token Economics")
                    price = token_metadata.get('price', 0)
                    price_source = token_metadata.get('price_source', '')
                    
                    # Format the price source display
                    if price_source == 'coingecko':
                        display_source = "CoinGecko"
                    elif price > 0:
                        display_source = "DeFi API"
                    else:
                        display_source = "None"
                    
                    # Display price with source indicator
                    st.write(f"**Current Price:** {format_price(price)}")
                    st.write(f"**Price Source:** {display_source}")
                    
                    # Active status
                    active = token_metadata.get('active', False)
                    st.write(f"**Active Status:** {'Active ✓' if active else 'Inactive ✗'}")
                    
                    # Address source if available
                    address_source = token_metadata.get('address_source', '')
                    if address_source:
                        st.write(f"**Address Source:** {address_source}")
                    
                    # Note about data source
                    if price_source == 'coingecko':
                        st.info(f"Price data provided by CoinGecko API for higher accuracy and reliability.")
                    elif price > 0:
                        st.info(f"Data sourced from the DeFi API.")
                    else:
                        st.info(f"No price data available for this token from our data sources.")
            else:
                st.warning(f"No detailed metadata available for {selected_token}.")
    
    # Footer with documentation
    st.markdown("---")
    with st.expander("About Token Analysis"):
        st.markdown("""
        ### Token Data Documentation
        
        This page provides comprehensive token data from multiple reliable sources with a focus on CoinGecko for data authenticity.
        
        **Data Fields:**
        - **Symbol**: The token's symbol (e.g., SOL, RAY)
        - **Name**: The full name of the token
        - **Address**: The Solana address of the token (from CoinGecko when available)
        - **Decimals**: The number of decimal places for the token
        - **Price**: Current price in USD
        - **Active**: Whether the token is active in the ecosystem
        
        **DEX Categorization:**
        Tokens are categorized by the DEXes where they're commonly used:
        - **Raydium**: Tokens like RAY, SOL, USDC
        - **Meteora**: Tokens like mSOL, BTC, USDC
        - **Orca**: Tokens like SOL, ETH, USDC
        
        **Price Data:**
        Token prices are obtained from multiple reliable sources with emphasis on authentic data:
        - **CoinGecko**: Primary source for token prices and metadata
        - **DeFi API**: Secondary source when CoinGecko data is unavailable
        
        Each token's price source is clearly labeled to provide transparency about data origin and ensure you have access to trustworthy financial information.
        """)

if __name__ == "__main__":
    main()