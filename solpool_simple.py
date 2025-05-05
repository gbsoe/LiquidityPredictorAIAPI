import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="SolPool Insight - Simple Version",
    page_icon="🧪",
    layout="wide"
)

# Performance tracking
start_time = time.time()

# Title and intro
st.title("SolPool Insight - Simple Version")
st.write("A simple version of the Solana Liquidity Pool Analytics Platform")

# Create tabs
tab1, tab2 = st.tabs(["Pool Data", "Token Info"])

with tab1:
    st.header("Pool Data")
    
    # Create sample pool data
    pools = [
        {
            "name": "SOL-USDC",
            "dex": "Raydium",
            "liquidity": 12500000,
            "volume_24h": 4300000,
            "apr": 15.2,
            "fee": 0.25,
            "token1": "SOL",
            "token2": "USDC"
        },
        {
            "name": "BTC-USDC",
            "dex": "Raydium",
            "liquidity": 8700000,
            "volume_24h": 2100000,
            "apr": 12.1,
            "fee": 0.3,
            "token1": "BTC",
            "token2": "USDC"
        },
        {
            "name": "ETH-USDC",
            "dex": "Orca",
            "liquidity": 7200000,
            "volume_24h": 1850000,
            "apr": 9.8,
            "fee": 0.2,
            "token1": "ETH",
            "token2": "USDC"
        },
        {
            "name": "SOL-BONK",
            "dex": "Raydium",
            "liquidity": 950000,
            "volume_24h": 320000,
            "apr": 42.5,
            "fee": 0.3,
            "token1": "SOL",
            "token2": "BONK"
        },
        {
            "name": "SOL-RAY",
            "dex": "Raydium",
            "liquidity": 3800000,
            "volume_24h": 980000,
            "apr": 18.7,
            "fee": 0.25,
            "token1": "SOL",
            "token2": "RAY"
        }
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(pools)
    
    # Format columns for display
    def format_currency(value):
        if value >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"${value/1_000:.2f}K"
        else:
            return f"${value:.2f}"
    
    def format_percentage(value):
        return f"{value:.2f}%"
    
    display_df = df.copy()
    display_df["liquidity_fmt"] = display_df["liquidity"].apply(format_currency)
    display_df["volume_24h_fmt"] = display_df["volume_24h"].apply(format_currency)
    display_df["apr_fmt"] = display_df["apr"].apply(format_percentage)
    display_df["fee_fmt"] = display_df["fee"].apply(lambda x: f"{x:.2f}%")
    
    # Display the pool data
    st.subheader("Liquidity Pools")
    st.dataframe(
        display_df[[
            "name", "dex", "liquidity_fmt", "volume_24h_fmt", 
            "apr_fmt", "fee_fmt"
        ]].rename(columns={
            "name": "Pool",
            "dex": "DEX",
            "liquidity_fmt": "Liquidity",
            "volume_24h_fmt": "Volume (24h)",
            "apr_fmt": "APR",
            "fee_fmt": "Fee"
        }),
        use_container_width=True
    )
    
    # Display statistics
    st.subheader("Pool Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pools", len(df))
    
    with col2:
        st.metric("Total Liquidity", format_currency(df["liquidity"].sum()))
    
    with col3:
        st.metric("Average APR", format_percentage(df["apr"].mean()))
    
    with col4:
        st.metric("24h Volume", format_currency(df["volume_24h"].sum()))
    
    # DEX distribution
    st.subheader("DEX Distribution")
    dex_counts = df["dex"].value_counts().reset_index()
    dex_counts.columns = ["DEX", "Count"]
    st.bar_chart(dex_counts.set_index("DEX"))

with tab2:
    st.header("Token Information")
    
    # Create sample token data
    tokens = {
        "SOL": {"price": 172.45, "name": "Solana"},
        "BTC": {"price": 68432.12, "name": "Bitcoin"},
        "ETH": {"price": 3027.89, "name": "Ethereum"},
        "USDC": {"price": 1.0, "name": "USD Coin"},
        "BONK": {"price": 0.00002453, "name": "Bonk"},
        "RAY": {"price": 0.521, "name": "Raydium"}
    }
    
    # Token selection
    selected_token = st.selectbox(
        "Select a token",
        options=list(tokens.keys())
    )
    
    if selected_token:
        token_data = tokens[selected_token]
        
        # Display token info
        st.subheader(f"{selected_token} Information")
        st.metric(
            f"{selected_token} Price",
            f"${token_data['price']:.4f}" if token_data['price'] < 1 else f"${token_data['price']:.2f}",
            delta=None
        )
        
        st.write(f"Name: {token_data['name']}")
        
        # Find pools with this token
        token_pools = df[
            (df["token1"] == selected_token) | 
            (df["token2"] == selected_token)
        ]
        
        st.subheader(f"Pools Containing {selected_token}")
        
        if len(token_pools) > 0:
            # Create display data
            pool_data = []
            
            for _, pool in token_pools.iterrows():
                pool_data.append({
                    "Pool Name": pool["name"],
                    "DEX": pool["dex"],
                    "Liquidity": format_currency(pool["liquidity"]),
                    "APR": format_percentage(pool["apr"]),
                    "Paired With": pool["token2"] if pool["token1"] == selected_token else pool["token1"]
                })
            
            # Display table
            st.dataframe(pd.DataFrame(pool_data), use_container_width=True)
        else:
            st.info(f"No pools found containing {selected_token}")

# Footer
st.markdown("---")
st.markdown("SolPool Insight - Simple Version | Solana Liquidity Pool Analytics Platform")

# Performance stats
end_time = time.time()
total_time = end_time - start_time
st.write(f"Loaded in {total_time:.2f} seconds")
