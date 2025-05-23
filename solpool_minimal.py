import streamlit as st
import pandas as pd
import os
import psycopg2
import time
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('solpool_insight')

# Set page config
st.set_page_config(
    page_title="SolPool Insight - Minimal Version",
    page_icon="🧪",
    layout="wide"
)

st.title("SolPool Insight - Minimal Version")
st.write("A simplified version of the Solana Liquidity Pool Analytics Platform")

# Performance monitoring
start_time = time.time()
loading_start = datetime.now()

# Simple token price service
def get_token_price(symbol, return_source=False):
    """Get a token price (simplified version)"""
    # Use a dictionary with a few sample prices
    sample_prices = {
        "SOL": 172.45,
        "BTC": 68432.12,
        "ETH": 3027.89,
        "USDC": 1.0,
        "USDT": 1.0,
        "RAY": 0.521,
        "BONK": 0.00002453
    }
    
    price = sample_prices.get(symbol.upper())
    source = "sample_data"
    
    if return_source:
        return (price, source) if price else (None, "none")
    return price

def get_multiple_prices(symbols):
    """Get prices for multiple tokens at once (simplified version)"""
    return {symbol: get_token_price(symbol) for symbol in symbols}

def update_pool_with_token_prices(pool):
    """Update a pool dictionary with token prices (simplified version)"""
    token1_symbol = pool.get('token1_symbol')
    token2_symbol = pool.get('token2_symbol')
    
    # Find prices
    if token1_symbol:
        token1_price = get_token_price(token1_symbol)
        pool['token1_price'] = token1_price if token1_price else 0.0
    
    if token2_symbol:
        token2_price = get_token_price(token2_symbol)
        pool['token2_price'] = token2_price if token2_price else 0.0
    
    return pool

# Utility functions
def format_currency(value):
    """Format a value as currency"""
    if value is None:
        return "N/A"
    
    try:
        value = float(value)
        if value >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"${value/1_000:.2f}K"
        else:
            return f"${value:.2f}"
    except (ValueError, TypeError):
        return "N/A"

def format_percentage(value):
    """Format a value as percentage"""
    if value is None:
        return "N/A"
    
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return "N/A"

def get_trend_icon(value):
    """Return an arrow icon based on trend direction"""
    if value is None:
        return ""
    elif value > 0:
        return "↑" # Up arrow
    elif value < 0:
        return "↓" # Down arrow
    else:
        return "→" # Right arrow

def get_category_badge(category):
    """Return HTML for a category badge"""
    colors = {
        "Stable": "#1E88E5",
        "Volatile": "#F44336",
        "Farm": "#43A047",
        "Popular": "#FFB300",
        "New": "#8E24AA",
        "Low Volume": "#757575",
        "High APR": "#FB8C00",
        "Large Cap": "#26A69A" 
    }
    
    color = colors.get(category, "#757575")
    return f"<span style='background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px;'>{category}</span>"

# Connection function
def get_db_connection():
    """Get a connection to the PostgreSQL database"""
    db_url = os.environ.get("DATABASE_URL")
    
    if db_url:
        return psycopg2.connect(db_url)
    else:
        # Fall back to individual connection parameters
        return psycopg2.connect(
            host=os.environ.get("PGHOST"),
            port=os.environ.get("PGPORT"),
            database=os.environ.get("PGDATABASE"),
            user=os.environ.get("PGUSER"),
            password=os.environ.get("PGPASSWORD")
        )

# Function to retrieve pools from the database
def load_pools_from_database():
    """Load liquidity pool data from the PostgreSQL database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if the pools table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'pools'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            # Create a simple pools table
            cursor.execute("""
                CREATE TABLE pools (
                    id SERIAL PRIMARY KEY,
                    pool_id VARCHAR(255) UNIQUE,
                    name VARCHAR(255),
                    dex VARCHAR(100),
                    token1_symbol VARCHAR(20),
                    token2_symbol VARCHAR(20),
                    liquidity NUMERIC,
                    volume_24h NUMERIC,
                    apr NUMERIC,
                    fee NUMERIC,
                    category VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            
            # Add some sample data
            sample_pools = [
                {
                    "pool_id": "sol_usdc_raydium",
                    "name": "SOL-USDC",
                    "dex": "Raydium",
                    "token1_symbol": "SOL",
                    "token2_symbol": "USDC",
                    "liquidity": 12500000,
                    "volume_24h": 4300000,
                    "apr": 15.2,
                    "fee": 0.25,
                    "category": "Popular"
                },
                {
                    "pool_id": "btc_usdc_raydium",
                    "name": "BTC-USDC", 
                    "dex": "Raydium",
                    "token1_symbol": "BTC", 
                    "token2_symbol": "USDC",
                    "liquidity": 8700000,
                    "volume_24h": 2100000,
                    "apr": 12.1,
                    "fee": 0.3,
                    "category": "Stable"
                },
                {
                    "pool_id": "eth_usdc_orca",
                    "name": "ETH-USDC",
                    "dex": "Orca",
                    "token1_symbol": "ETH",
                    "token2_symbol": "USDC",
                    "liquidity": 7200000,
                    "volume_24h": 1850000,
                    "apr": 9.8,
                    "fee": 0.2,
                    "category": "Stable"
                },
                {
                    "pool_id": "sol_bonk_raydium",
                    "name": "SOL-BONK",
                    "dex": "Raydium",
                    "token1_symbol": "SOL",
                    "token2_symbol": "BONK",
                    "liquidity": 950000,
                    "volume_24h": 320000,
                    "apr": 42.5,
                    "fee": 0.3,
                    "category": "Volatile"
                },
                {
                    "pool_id": "sol_ray_raydium",
                    "name": "SOL-RAY",
                    "dex": "Raydium",
                    "token1_symbol": "SOL",
                    "token2_symbol": "RAY",
                    "liquidity": 3800000,
                    "volume_24h": 980000,
                    "apr": 18.7,
                    "fee": 0.25,
                    "category": "Farm"
                }
            ]
            
            for pool in sample_pools:
                cursor.execute("""
                    INSERT INTO pools 
                        (pool_id, name, dex, token1_symbol, token2_symbol, 
                         liquidity, volume_24h, apr, fee, category) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    pool["pool_id"], pool["name"], pool["dex"], 
                    pool["token1_symbol"], pool["token2_symbol"],
                    pool["liquidity"], pool["volume_24h"], pool["apr"],
                    pool["fee"], pool["category"]
                ))
            
            conn.commit()
            st.success("Sample data created in the database!")
        
        # Fetch the pool data
        cursor.execute("SELECT * FROM pools")
        columns = [desc[0] for desc in cursor.description]
        pools = cursor.fetchall()
        
        # Convert to list of dictionaries
        pools_data = [dict(zip(columns, pool)) for pool in pools]
        
        # Close the connection
        cursor.close()
        conn.close()
        
        # Update pools with token prices
        for pool in pools_data:
            update_pool_with_token_prices(pool)
        
        return pools_data
    
    except Exception as e:
        st.error(f"Database error: {e}")
        logger.error(f"Error loading pools from database: {e}")
        return []

# Main application function
def main():
    """Main application function"""
    try:
        # Sidebar
        st.sidebar.title("SolPool Insight")
        st.sidebar.write("Solana Liquidity Pool Analytics")
        
        # Load data
        with st.spinner("Loading pool data..."):
            pools = load_pools_from_database()
            
            # Debug information about pools
            st.write("Retrieved pool data:")
            st.json(pools[0] if pools else {"message": "No pools found"})
            
            # Create DataFrame
            df = pd.DataFrame(pools)
        
        # Check if data was loaded
        if df.empty:
            st.warning("No pool data available. Please check the database connection.")
            return
        
        # Debug information
        st.write("DataFrame columns:", list(df.columns))
        st.write("DataFrame types:", df.dtypes.to_dict())
        
        # Basic data check
        st.write("Sample row:", df.iloc[0].to_dict() if len(df) > 0 else "No data")
        
        st.success(f"Loaded {len(df)} pools from database.")
        
        # Dashboard tabs
        tab1, tab2, tab3 = st.tabs(["Pool Explorer", "Analytics", "Tokens"])
        
        # Pool Explorer tab
        with tab1:
            st.header("Pool Explorer")
            
            # Filters
            st.subheader("Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dex_filter = st.multiselect(
                    "DEX",
                    options=sorted(df["dex"].unique()),
                    default=[]
                )
            
            with col2:
                min_liquidity = st.number_input(
                    "Min Liquidity ($)",
                    min_value=0.0,
                    value=0.0,
                    step=100000.0,
                    format="%f"
                )
            
            with col3:
                min_apr = st.number_input(
                    "Min APR (%)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    format="%f"
                )
            
            # Apply filters
            filtered_df = df.copy()
            
            # Add debug information
            st.write("DataFrame columns:", list(df.columns))
            
            if dex_filter:
                filtered_df = filtered_df[filtered_df["dex"].isin(dex_filter)]
            
            # Check if liquidity column exists and has correct type
            if "liquidity" in filtered_df.columns:
                # Convert to numeric if needed
                filtered_df["liquidity"] = pd.to_numeric(filtered_df["liquidity"], errors="coerce")
                
                if min_liquidity > 0:
                    filtered_df = filtered_df[filtered_df["liquidity"] >= min_liquidity]
            
            # Check if apr column exists and has correct type
            if "apr" in filtered_df.columns:
                # Convert to numeric if needed
                filtered_df["apr"] = pd.to_numeric(filtered_df["apr"], errors="coerce")
                
                if min_apr > 0:
                    filtered_df = filtered_df[filtered_df["apr"] >= min_apr]
            
            # Display pools
            st.subheader("Pools")
            st.write(f"Showing {len(filtered_df)} of {len(df)} pools")
            
            # Prepare display data
            display_df = filtered_df.copy()
            
            # Format columns for display
            if not display_df.empty:
                # Check and ensure all required columns exist
                for col in ["liquidity", "volume_24h", "apr", "fee"]:
                    if col not in display_df.columns:
                        display_df[col] = 0
                    else:
                        # Convert to numeric to ensure they are numbers
                        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(0)
                
                # Format columns for display
                display_df["liquidity_fmt"] = display_df["liquidity"].apply(format_currency)
                display_df["volume_24h_fmt"] = display_df["volume_24h"].apply(format_currency)
                display_df["apr_fmt"] = display_df["apr"].apply(format_percentage)
                display_df["fee_fmt"] = display_df["fee"].apply(lambda x: f"{x:.2f}%")
                
                # Sort by liquidity (descending)
                display_df = display_df.sort_values("liquidity", ascending=False)
                
                # Create a clean display table
                st.dataframe(
                    display_df[[
                        "name", "dex", "liquidity_fmt", "volume_24h_fmt", 
                        "apr_fmt", "fee_fmt", "category"
                    ]].rename(columns={
                        "name": "Pool",
                        "dex": "DEX",
                        "liquidity_fmt": "Liquidity",
                        "volume_24h_fmt": "Volume (24h)",
                        "apr_fmt": "APR",
                        "fee_fmt": "Fee",
                        "category": "Category"
                    }),
                    use_container_width=True
                )
                
                # Pool details
                st.subheader("Pool Details")
                selected_pool_id = st.selectbox(
                    "Select a pool for details",
                    options=display_df["pool_id"].tolist(),
                    format_func=lambda x: display_df[display_df["pool_id"] == x]["name"].iloc[0]
                )
                
                if selected_pool_id:
                    pool = display_df[display_df["pool_id"] == selected_pool_id].iloc[0].to_dict()
                    
                    # Display pool details in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### {pool['name']} ({pool['dex']})")
                        st.write(f"**Category:** {pool['category']}")
                        st.write(f"**Liquidity:** {format_currency(pool['liquidity'])}")
                        st.write(f"**Volume (24h):** {format_currency(pool['volume_24h'])}")
                    
                    with col2:
                        st.write(f"**APR:** {format_percentage(pool['apr'])}")
                        st.write(f"**Fee:** {format_percentage(pool['fee'])}")
                        
                        # Calculate token values
                        token1_price = pool.get('token1_price', 0)
                        token2_price = pool.get('token2_price', 0)
                        
                        if token1_price:
                            st.write(f"**{pool['token1_symbol']} Price:** ${token1_price:,.4f}")
                        
                        if token2_price:
                            st.write(f"**{pool['token2_symbol']} Price:** ${token2_price:,.4f}")
        
        # Analytics tab
        with tab2:
            st.header("Analytics")
            
            # Simple analytics
            st.subheader("Pool Statistics")
            
            # Calculate some stats safely
            try:
                # Convert data types to numeric for calculations
                if "liquidity" in df.columns:
                    df["liquidity"] = pd.to_numeric(df["liquidity"], errors="coerce")
                    total_liquidity = df["liquidity"].sum()
                else:
                    total_liquidity = 0
                    
                if "apr" in df.columns:
                    df["apr"] = pd.to_numeric(df["apr"], errors="coerce")
                    avg_apr = df["apr"].mean()
                else:
                    avg_apr = 0
                    
                if "volume_24h" in df.columns:
                    df["volume_24h"] = pd.to_numeric(df["volume_24h"], errors="coerce")
                    total_volume = df["volume_24h"].sum()
                else:
                    total_volume = 0
                    
                pool_count = len(df)
            except Exception as e:
                st.error(f"Error calculating stats: {e}")
                total_liquidity = 0
                avg_apr = 0
                total_volume = 0
                pool_count = len(df)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Pools", pool_count)
            
            with col2:
                st.metric("Total Liquidity", format_currency(total_liquidity))
            
            with col3:
                st.metric("Average APR", format_percentage(avg_apr))
            
            with col4:
                st.metric("24h Volume", format_currency(total_volume))
            
            # DEX breakdown
            st.subheader("DEX Distribution")
            try:
                if "dex" in df.columns and not df.empty:
                    # Get value counts
                    dex_counts = df["dex"].value_counts().reset_index()
                    dex_counts.columns = ["DEX", "Count"]
                    
                    # Display chart
                    st.bar_chart(dex_counts.set_index("DEX"))
                else:
                    st.info("No DEX data available for distribution chart.")
            except Exception as e:
                st.error(f"Error creating DEX distribution: {e}")
                st.info("DEX distribution could not be calculated.")
                
                # Display a simple message instead
                st.write("DEX Distribution not available due to data format issues.")
            
            # APR distribution
            st.subheader("APR Distribution")
            try:
                if "apr" in df.columns and not df.empty:
                    # Make sure APR is numeric
                    df["apr"] = pd.to_numeric(df["apr"], errors="coerce")
                    
                    # Define APR range bins
                    apr_ranges = [
                        "0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"
                    ]
                    
                    # Calculate counts with safety checks
                    apr_counts = [
                        len(df[df["apr"] < 5]),
                        len(df[(df["apr"] >= 5) & (df["apr"] < 10)]),
                        len(df[(df["apr"] >= 10) & (df["apr"] < 15)]),
                        len(df[(df["apr"] >= 15) & (df["apr"] < 20)]),
                        len(df[(df["apr"] >= 20) & (df["apr"] < 30)]),
                        len(df[df["apr"] >= 30])
                    ]
                    
                    # Create a DataFrame for the chart
                    apr_df = pd.DataFrame({
                        "APR Range": apr_ranges,
                        "Count": apr_counts
                    })
                    
                    # Display chart
                    st.bar_chart(apr_df.set_index("APR Range"))
                else:
                    st.info("No APR data available for distribution chart.")
            except Exception as e:
                st.error(f"Error creating APR distribution: {e}")
                st.info("APR distribution could not be calculated.")
                
                # Display a simple message instead
                st.write("APR Distribution not available due to data format issues.")
            
        
        # Tokens tab
        with tab3:
            st.header("Token Explorer")
            
            # Extract unique tokens safely
            try:
                all_tokens = []
                
                if "token1_symbol" in df.columns:
                    all_tokens.extend(list(df["token1_symbol"].unique()))
                    
                if "token2_symbol" in df.columns:
                    all_tokens.extend(list(df["token2_symbol"].unique()))
                    
                # Remove duplicates, None values, and sort
                all_tokens = sorted([t for t in set(all_tokens) if t])
                
                if not all_tokens:
                    # Provide a fallback if no tokens found
                    all_tokens = ["SOL", "BTC", "ETH", "USDC", "USDT"]
                    st.info("Using sample tokens as no token data was found.")
            except Exception as e:
                st.error(f"Error extracting tokens: {e}")
                # Fallback to sample tokens
                all_tokens = ["SOL", "BTC", "ETH", "USDC", "USDT"]
                st.info("Using sample tokens due to data format issues.")
            
            # Token selection
            selected_token = st.selectbox(
                "Select a token",
                options=all_tokens
            )
            
            if selected_token:
                # Get token price
                token_price, price_source = get_token_price(selected_token, return_source=True)
                
                # Token info
                st.subheader(f"{selected_token} Information")
                
                if token_price:
                    st.metric(
                        f"{selected_token} Price", 
                        f"${token_price:,.4f}",
                        delta=None,
                        help=f"Price source: {price_source}"
                    )
                else:
                    st.warning(f"Could not retrieve price for {selected_token}")
                
                # Find pools containing this token
                token_pools = df[
                    (df["token1_symbol"] == selected_token) | 
                    (df["token2_symbol"] == selected_token)
                ]
                
                st.subheader(f"Pools Containing {selected_token}")
                
                if len(token_pools) > 0:
                    # Prepare display data
                    pool_data = []
                    
                    for _, pool in token_pools.iterrows():
                        pool_data.append({
                            "Pool Name": pool["name"],
                            "DEX": pool["dex"],
                            "Liquidity": format_currency(pool["liquidity"]),
                            "Volume (24h)": format_currency(pool["volume_24h"]),
                            "APR": format_percentage(pool["apr"]),
                            "Paired With": pool["token2_symbol"] if pool["token1_symbol"] == selected_token else pool["token1_symbol"]
                        })
                    
                    # Create a DataFrame for better display
                    pool_df = pd.DataFrame(pool_data)
                    
                    # Display table
                    st.dataframe(pool_df, use_container_width=True)
                else:
                    st.info(f"No pools found containing {selected_token}")
        
        # Footer
        st.markdown("---")
        st.markdown("SolPool Insight - Minimal Version | Solana Liquidity Pool Analytics Platform")
        
        # Performance stats
        end_time = time.time()
        loading_end = datetime.now()
        total_time = end_time - start_time
        
        st.write(f"Loaded in {total_time:.2f} seconds")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {e}")

# Run the application
if __name__ == "__main__":
    main()
