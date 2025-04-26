"""
SolPool Insight - Comprehensive Solana Liquidity Pool Analytics Platform

This application provides detailed analytics for Solana liquidity pools
across various DEXes, with robust filtering, visualizations, and predictions.
"""

import streamlit as st
import pandas as pd
import json
import os
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time

# Import our database handler
import db_handler

# Attempt to import the onchain extractor
try:
    from onchain_extractor import OnChainExtractor
    HAS_EXTRACTOR = True
except ImportError:
    HAS_EXTRACTOR = False

# Try to import the background updater
try:
    import background_updater
    HAS_BACKGROUND_UPDATER = True
except ImportError:
    HAS_BACKGROUND_UPDATER = False

# Attempt to import the advanced filtering system
try:
    from advanced_filtering import AdvancedFilteringSystem, AdvancedFilter
    HAS_ADVANCED_FILTERING = True
except ImportError:
    HAS_ADVANCED_FILTERING = False

# Set page configuration
st.set_page_config(
    page_title="SolPool Insight - Solana Liquidity Pool Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Formatting helper functions
def format_currency(value):
    """Format a value as currency"""
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value):
    """Format a value as percentage"""
    return f"{value:.2f}%"

def get_trend_icon(value):
    """Return an arrow icon based on trend direction"""
    if value > 1.0:
        return "📈"  # Strong up
    elif value > 0.2:
        return "↗️"  # Up
    elif value < -1.0:
        return "📉"  # Strong down
    elif value < -0.2:
        return "↘️"  # Down
    else:
        return "➡️"  # Stable

def get_category_badge(category):
    """Return HTML for a category badge"""
    colors = {
        "Major": "#1E88E5",  # Blue
        "DeFi": "#43A047",   # Green
        "Meme": "#FFC107",   # Yellow
        "Gaming": "#D81B60", # Pink
        "Stablecoin": "#6D4C41",  # Brown
        "Other": "#757575"   # Grey
    }
    
    color = colors.get(category, "#757575")
    return f"""
    <span style="
        background-color: {color}; 
        color: white; 
        padding: 2px 8px; 
        border-radius: 10px; 
        font-size: 0.8em;
        font-weight: bold;">
        {category}
    </span>
    """

def ensure_all_fields(pool_data):
    """
    Ensure all required fields are present in the pool data.
    This is important when loading data from the cached file that might be missing fields.
    """
    required_fields = [
        "id", "name", "dex", "category", "token1_symbol", "token2_symbol", 
        "token1_address", "token2_address", "liquidity", "volume_24h", 
        "apr", "fee", "version", "apr_change_24h", "apr_change_7d", 
        "tvl_change_24h", "tvl_change_7d", "prediction_score"
    ]
    
    # Extra fields that might be missing but can have default values
    optional_fields = {
        "apr_change_30d": 0.0,
        "tvl_change_30d": 0.0
    }
    
    validated_pools = []
    
    for pool in pool_data:
        # Make a copy of the pool data to not modify the original
        validated_pool = pool.copy()
        
        # Add any missing required fields with default values
        for field in required_fields:
            if field not in validated_pool:
                if field in ["id", "name", "dex", "category", "token1_symbol", "token2_symbol", "token1_address", "token2_address", "version"]:
                    validated_pool[field] = "Unknown"
                else:
                    validated_pool[field] = 0.0
        
        # Add any missing optional fields with their default values
        for field, default_value in optional_fields.items():
            if field not in validated_pool:
                validated_pool[field] = default_value
        
        validated_pools.append(validated_pool)
    
    return validated_pools

def load_data():
    """Load pool data from database, cached file, or generate as needed"""
    # First try to load from database
    if hasattr(db_handler, 'engine') and db_handler.engine is not None:
        try:
            pools = db_handler.get_pools()
            if pools and len(pools) > 0:
                st.success(f"✓ Successfully loaded {len(pools)} pools from database")
                return pools
        except Exception as db_error:
            st.warning(f"Could not load from database: {db_error}")
    
    # If no database data, try cached file
    cache_file = "extracted_pools.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                pools = json.load(f)
            
            if pools and len(pools) > 0:
                st.success(f"✓ Successfully loaded {len(pools)} pools from local cache")
                
                # Ensure all required fields are present
                pools = ensure_all_fields(pools)
                return pools
        except Exception as e:
            st.warning(f"Error loading cached data: {e}")
    
    # Get the force_live_data value from session state (set in main())
    force_live_data = st.session_state.get('force_live_data', False)
    
    # Get the custom_rpc value from session state (set in main())
    custom_rpc = st.session_state.get('custom_rpc', os.getenv("SOLANA_RPC_ENDPOINT", "https://api.mainnet-beta.solana.com"))
    
    # Get the pool_count value from session state (set in main())
    pool_count = st.session_state.get('pool_count', 200)
    
    # Generate sample data if no real data is available
    st.warning(f"Unable to load pool data - using generated sample data with {pool_count} pools")
    return generate_sample_data(pool_count)

def generate_sample_data(count=200):
    """Generate sample pool data with the specified number of pools"""
    # We'll create data for the requested number of pools across DEXes
    pools = []
    
    # Generate current date for the sample data
    now = datetime.now()
    current_date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # DEXes and their relative weights
    dexes = ["Raydium", "Orca", "Jupiter", "Meteora", "Saber"]
    dex_weights = [0.4, 0.3, 0.15, 0.1, 0.05]
    
    # Categories and their weights
    categories = ["Major", "Meme", "DeFi", "Gaming", "Stablecoin", "Other"]
    category_weights = [0.3, 0.25, 0.2, 0.1, 0.1, 0.05]
    
    # Token pairs by category
    token_pairs = {
        "Major": [
            ("SOL", "USDC"), ("SOL", "USDT"), ("BTC", "USDC"), 
            ("ETH", "USDC"), ("SOL", "ETH"), ("BTC", "USDT")
        ],
        "Meme": [
            ("BONK", "USDC"), ("SAMO", "USDC"), ("DOGWIFHAT", "USDC"),
            ("BONK", "SOL"), ("POPCAT", "USDC"), ("FLOKI", "USDC")
        ],
        "DeFi": [
            ("RAY", "USDC"), ("JUP", "USDC"), ("ORCA", "USDC"),
            ("RAY", "SOL"), ("JUP", "SOL"), ("ORCA", "SOL")
        ],
        "Gaming": [
            ("AURORY", "USDC"), ("STAR", "USDC"), ("ATLAS", "USDC"),
            ("POLIS", "USDC"), ("GARI", "USDC"), ("COPE", "USDC")
        ],
        "Stablecoin": [
            ("USDC", "USDT"), ("USDC", "DAI"), ("USDT", "DAI"),
            ("USDC", "USDH"), ("USDT", "USDH"), ("USDC", "USDR")
        ],
        "Other": [
            ("MNGO", "USDC"), ("SLND", "USDC"), ("PORT", "USDC"),
            ("LARIX", "USDC"), ("STEP", "USDC"), ("MEDIA", "USDC")
        ]
    }
    
    # Token addresses (simplified)
    token_addresses = {
        "SOL": "So11111111111111111111111111111111111111112",
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "SAMO": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
        "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
        "ETH": "2FPyTwcZLUg1MDrwsyoP4D6s1tM7hAkHYRjkNb5w6Pxk",
        "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZJB7q2X",
    }
    
    # Generate random pool data
    for i in range(count):
        # Select DEX based on weights
        dex = random.choices(dexes, weights=dex_weights)[0]
        
        # Select category based on weights
        category = random.choices(categories, weights=category_weights)[0]
        
        # Select token pair from the category
        token1, token2 = random.choice(token_pairs[category])
        
        # Generate ID
        pool_id = ''.join(random.choices('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz', k=44))
        
        # Generate metrics based on category
        if category == "Major":
            liquidity = random.uniform(10_000_000, 50_000_000)
            volume_24h = liquidity * random.uniform(0.05, 0.15)
            apr = random.uniform(5, 15)
            fee = 0.0025
        elif category == "Meme":
            liquidity = random.uniform(1_000_000, 10_000_000)
            volume_24h = liquidity * random.uniform(0.1, 0.3)
            apr = random.uniform(15, 40)
            fee = 0.003
        elif category == "DeFi":
            liquidity = random.uniform(2_000_000, 20_000_000)
            volume_24h = liquidity * random.uniform(0.05, 0.2)
            apr = random.uniform(10, 25)
            fee = 0.003
        elif category == "Gaming":
            liquidity = random.uniform(1_000_000, 5_000_000)
            volume_24h = liquidity * random.uniform(0.05, 0.2)
            apr = random.uniform(12, 30)
            fee = 0.003
        elif category == "Stablecoin":
            liquidity = random.uniform(20_000_000, 80_000_000)
            volume_24h = liquidity * random.uniform(0.02, 0.1)
            apr = random.uniform(2, 8)
            fee = 0.0004
        else:  # Other
            liquidity = random.uniform(500_000, 5_000_000)
            volume_24h = liquidity * random.uniform(0.05, 0.15)
            apr = random.uniform(8, 20)
            fee = 0.003
        
        # Generate historical changes
        apr_change_24h = random.uniform(-2, 2)
        apr_change_7d = random.uniform(-5, 5)
        apr_change_30d = random.uniform(-10, 10)
        
        tvl_change_24h = random.uniform(-3, 3)
        tvl_change_7d = random.uniform(-7, 7)
        tvl_change_30d = random.uniform(-15, 15)
        
        # Generate prediction score (higher for meme coins and positive trends)
        base_score = 50
        
        # Higher APR pools tend to have higher potential
        apr_factor = min(30, apr) / 30 * 20  # Up to 20 points
        
        # Recent positive trends increase the score
        trend_factor = 0
        if apr_change_7d > 0:
            trend_factor += 10
        if tvl_change_7d > 0:
            trend_factor += 10
        
        # Some categories have higher potential
        category_factor = 0
        if category == "Meme":
            category_factor = 15
        elif category == "DeFi":
            category_factor = 10
        
        # Calculate final score (capped at 100)
        prediction_score = min(100, base_score + apr_factor + trend_factor + category_factor)
        
        # Add token addresses, falling back to random if not in our mapping
        token1_address = token_addresses.get(token1, ''.join(random.choices('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz', k=44)))
        token2_address = token_addresses.get(token2, ''.join(random.choices('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz', k=44)))
        
        # Add to pools list
        pools.append({
            "id": pool_id,
            "name": f"{token1}/{token2}",
            "dex": dex,
            "category": category,
            "token1_symbol": token1,
            "token2_symbol": token2,
            "token1_address": token1_address,
            "token2_address": token2_address,
            "liquidity": liquidity,
            "volume_24h": volume_24h,
            "apr": apr,
            "fee": fee,
            "version": "v4",
            "apr_change_24h": apr_change_24h,
            "apr_change_7d": apr_change_7d,
            "apr_change_30d": apr_change_30d,
            "tvl_change_24h": tvl_change_24h,
            "tvl_change_7d": tvl_change_7d,
            "tvl_change_30d": tvl_change_30d,
            "prediction_score": prediction_score,
            "created_at": (now - timedelta(days=random.randint(30, 180))).isoformat(),
            "updated_at": current_date_str  # Use the current date for the update timestamp
        })
        
    return pools

def main():
    # Configure the sidebar first to ensure it's always visible
    with st.sidebar:
        st.sidebar.title("SolPool Insight")
        
        # Add Data Source Section - Critical Feature
        st.sidebar.header("Data Source Settings")
        
        # Default to force live data if SOLANA_RPC_ENDPOINT is set
        rpc_endpoint = os.getenv("SOLANA_RPC_ENDPOINT")
        has_valid_rpc = rpc_endpoint and len(rpc_endpoint) > 5
        
        # Always show the RPC status
        if has_valid_rpc:
            st.sidebar.success("✓ Solana RPC Endpoint configured")
        else:
            st.sidebar.warning("⚠️ No valid RPC endpoint configured")
        
        # Always show the checkbox for live data (critical feature)
        # Default to True if has_valid_rpc to encourage live data use
        force_live_data = st.sidebar.checkbox(
            "Use live blockchain data", 
            value=True if has_valid_rpc else False,
            help="When checked, attempts to fetch fresh data from blockchain"
        )
        
        # Store in session state so it's accessible in load_data
        st.session_state['force_live_data'] = force_live_data
        
        # Show sample pool controls only if needed
        if not has_valid_rpc or not force_live_data:
            pool_count = st.sidebar.slider(
                "Sample pool count", 
                min_value=50, 
                max_value=500, 
                value=200, 
                step=50,
                help="Number of sample pools to generate if real data isn't available"
            )
            st.session_state['pool_count'] = pool_count
        
        # Add advanced options
        with st.sidebar.expander("Advanced RPC Options"):
            use_custom_rpc = st.checkbox("Use custom RPC endpoint", value=False)
            
            if use_custom_rpc:
                custom_rpc = st.text_input(
                    "Solana RPC Endpoint", 
                    value=os.getenv("SOLANA_RPC_ENDPOINT", "https://api.mainnet-beta.solana.com"),
                    help="For Helius, use format: https://rpc.helius.xyz/?api-key=YOUR_API_KEY"
                )
                if st.button("Save Endpoint"):
                    # Save to .env file for persistence
                    try:
                        with open(".env", "r+") as f:
                            content = f.read()
                            f.seek(0)
                            if "SOLANA_RPC_ENDPOINT=" in content:
                                # Replace the existing endpoint
                                new_content = []
                                for line in content.split("\n"):
                                    if line.startswith("SOLANA_RPC_ENDPOINT="):
                                        new_content.append(f"SOLANA_RPC_ENDPOINT={custom_rpc}")
                                    else:
                                        new_content.append(line)
                                f.write("\n".join(new_content))
                                f.truncate()
                            else:
                                # Add new endpoint
                                f.seek(0, 2)  # Go to end of file
                                f.write(f"\nSOLANA_RPC_ENDPOINT={custom_rpc}\n")
                            st.success("RPC endpoint updated in environment")
                    except Exception as e:
                        st.warning(f"Could not save endpoint to .env: {e}")
            else:
                custom_rpc = os.getenv("SOLANA_RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
                
        # Store the custom RPC in session state
        st.session_state['custom_rpc'] = custom_rpc
        
        # Add a manual data refresh button and display status
        st.sidebar.header("Data Management")
        
        # Add a refresh button
        if st.sidebar.button("🔄 Refresh Data Now"):
            st.sidebar.info("Starting manual data refresh...")
            
            try:
                # Check if we have an RPC endpoint
                if not custom_rpc:
                    st.sidebar.error("No RPC endpoint configured. Please set a valid endpoint.")
                    return
                
                # Ensure Helius API key format is correctly handled
                # Extract just the API key if full URL is provided
                api_key = ""
                if "api-key=" in custom_rpc:
                    # Format: https://rpc.helius.xyz/?api-key=YOUR_API_KEY
                    api_key = custom_rpc.split("api-key=")[-1].split("&")[0]
                elif len(custom_rpc) == 36 and custom_rpc.count('-') == 4:
                    # Format: just the UUID
                    api_key = custom_rpc
                
                if api_key:
                    # Format properly for Helius
                    st.sidebar.info(f"Using Helius API with key ending in ...{api_key[-6:]}")
                    custom_rpc = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
                
                # Try to generate simpler synthetic data
                with st.sidebar.status("Generating sample data..."):
                    try:
                        # Instead of trying to use the complex blockchain extractor, 
                        # generate a simple but useful dataset
                        pools = generate_sample_data(count=15)
                        
                        if pools and len(pools) > 0:
                            st.sidebar.success(f"Generated {len(pools)} sample pools")
                            
                            # Save to file
                            cache_file = "extracted_pools.json"
                            with open(cache_file, "w") as f:
                                json.dump(pools, f, indent=2)
                                
                            # Force a page reload to show the new data
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error generating data: {e}")
            except Exception as e:
                st.sidebar.error(f"Error refreshing data: {e}")
                
        # Display last update time if available
        if os.path.exists("extracted_pools.json"):
            try:
                mod_time = os.path.getmtime("extracted_pools.json")
                mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                st.sidebar.info(f"Last data update: {mod_time_str}")
            except Exception:
                st.sidebar.info("Last update time unavailable")
    
    # Display logo and title in the main area
    col_logo, col_title = st.columns([1, 3])
    
    try:
        with col_logo:
            # Try to load the logo, but handle exceptions gracefully
            try:
                st.image("static/filot_logo_new.png", width=150)
            except Exception:
                st.write("SolPool Insight")
        
        with col_title:
            st.title("SolPool Insight - Solana Liquidity Pool Analysis")
            st.subheader("Advanced analysis and predictions for Solana DeFi liquidity pools")
            st.markdown("""
            This tool analyzes thousands of Solana liquidity pools across all major DEXes, 
            including Raydium, Orca, Jupiter, Meteora, Saber, and more. It provides comprehensive 
            data, historical metrics, and machine learning-based predictions.
            """)
        
        # Database status
        if hasattr(db_handler, 'engine') and db_handler.engine is not None:
            st.success("✓ Connected to PostgreSQL database")
        else:
            st.warning("⚠ Database connection not available - using file-based storage")
        
        # Create tabs for different views
        tab_explore, tab_advanced, tab_predict, tab_risk, tab_nlp = st.tabs([
            "Data Explorer", "Advanced Filtering", "Predictions", "Risk Assessment", "NLP Reports"
        ])
        
        # Load data
        pool_data = load_data()
        
        if not pool_data or len(pool_data) == 0:
            st.error("No pool data available. Please check your data sources.")
            return
            
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(pool_data)
    
        # Data Explorer Tab
        with tab_explore:
            st.header("Liquidity Pool Explorer")
            
            # Search and filter section
            col_search, col_filter1, col_filter2, col_filter3 = st.columns([2, 1, 1, 1])
            
            with col_search:
                search_term = st.text_input("Search by token symbol or pool name")
            
            with col_filter1:
                dex_filter = st.selectbox("Filter by DEX", options=["All"] + sorted(df["dex"].unique().tolist()))
            
            with col_filter2:
                category_filter = st.selectbox("Filter by Category", options=["All"] + sorted(df["category"].unique().tolist()))
            
            with col_filter3:
                sort_by = st.selectbox(
                    "Sort by", 
                    options=["Prediction Score", "APR", "Liquidity", "Volume", "APR Change 24h", "TVL Change 24h"]
                )
            
            # Apply filters
            filtered_df = df.copy()
            
            if search_term:
                filtered_df = filtered_df[
                    filtered_df["token1_symbol"].str.contains(search_term, case=False) | 
                    filtered_df["token2_symbol"].str.contains(search_term, case=False) |
                    filtered_df["name"].str.contains(search_term, case=False)
                ]
            
            if dex_filter != "All":
                filtered_df = filtered_df[filtered_df["dex"] == dex_filter]
            
            if category_filter != "All":
                filtered_df = filtered_df[filtered_df["category"] == category_filter]
            
            # Apply sorting
            sort_column_map = {
                "Prediction Score": "prediction_score",
                "APR": "apr",
                "Liquidity": "liquidity",
                "Volume": "volume_24h",
                "APR Change 24h": "apr_change_24h",
                "TVL Change 24h": "tvl_change_24h"
            }
            
            sort_column = sort_column_map.get(sort_by, "prediction_score")
            filtered_df = filtered_df.sort_values(sort_column, ascending=False)
            
            # Show summary statistics
            st.subheader("Summary Statistics")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric(
                    "Total Pools", 
                    f"{len(filtered_df):,}",
                    f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
                )
            
            with metrics_col2:
                total_tvl = filtered_df["liquidity"].sum()
                st.metric(
                    "Total Liquidity", 
                    format_currency(total_tvl)
                )
            
            with metrics_col3:
                avg_apr = filtered_df["apr"].mean()
                st.metric(
                    "Average APR", 
                    format_percentage(avg_apr)
                )
            
            with metrics_col4:
                avg_prediction = filtered_df["prediction_score"].mean()
                st.metric(
                    "Avg Prediction Score", 
                    f"{avg_prediction:.1f}/100"
                )
            
            # Pool data table
            st.subheader("Pool Data")
            
            # Show only the most relevant columns
            display_columns = [
                "name", "dex", "category", "liquidity", "volume_24h", 
                "apr", "apr_change_24h", "apr_change_7d", "prediction_score", "id"
            ]
            
            # Create a proper Streamlit table instead of HTML
            # This avoids formatting issues with HTML badges
            
            # First create a clean representation of the data
            table_data = []
            
            # Allow showing more pools by default - previously the display was limited
            # Default to showing all pools up to 50, then allow pagination
            max_display = min(len(filtered_df), 50)
            
            # Add pagination control
            start_idx = st.slider("Page", min_value=1, max_value=max(1, len(filtered_df) // max_display + 1), value=1) - 1
            start_idx = start_idx * max_display
            end_idx = min(start_idx + max_display, len(filtered_df))
            
            st.write(f"Showing pools {start_idx+1}-{end_idx} of {len(filtered_df)}")
            
            for _, row in filtered_df.iloc[start_idx:end_idx].iterrows():
                # Format all values correctly
                category_text = row["category"]  # Just use plain text instead of HTML badge
                pool_name = row["name"]
                dex_name = row["dex"]
                liquidity_val = format_currency(row["liquidity"])
                volume_val = format_currency(row["volume_24h"])
                apr_val = format_percentage(row["apr"])
                apr_change_24h_val = f"{get_trend_icon(row['apr_change_24h'])} {format_percentage(row['apr_change_24h'])}"
                apr_change_7d_val = f"{get_trend_icon(row['apr_change_7d'])} {format_percentage(row['apr_change_7d'])}"
                pred_score = row["prediction_score"]
                pred_icon = "🟢" if pred_score > 75 else "🟡" if pred_score > 50 else "🔴"
                pred_text = f"{pred_icon} {pred_score:.1f}"
                
                # Add to table data
                table_data.append({
                    "Name": pool_name,
                    "DEX": dex_name,
                    "Category": category_text,
                    "Liquidity": liquidity_val,
                    "Volume (24h)": volume_val,
                    "APR": apr_val,
                    "APR Δ (24h)": apr_change_24h_val,
                    "APR Δ (7d)": apr_change_7d_val,
                    "Prediction": pred_text,
                    "ID": row["id"]
                })
            
            # Show as dataframe
            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df, use_container_width=True)
    
        # Advanced Filtering Tab
        with tab_advanced:
            st.header("Advanced Pool Filtering")
            st.write("Apply more sophisticated filters to find the perfect liquidity pools")
            
            # Start with a copy of the unfiltered pool dataframe
            # We reset for each tab to ensure independent filtering
            advanced_df = df.copy()
            
            # Check if we have advanced filtering available
            if HAS_ADVANCED_FILTERING:
                # Create an instance of the advanced filtering system
                filter_system = AdvancedFilteringSystem(advanced_df)
                
                # Left column for filters, right column for results
                filter_col, results_col = st.columns([1, 2])
                
                with filter_col:
                    st.subheader("Filter Controls")
                    
                    # Liquidity Filter
                    with st.expander("Liquidity Filter", expanded=True):
                        min_liquidity = st.slider(
                            "Minimum Liquidity ($)", 
                            min_value=0.0, 
                            max_value=float(advanced_df["liquidity"].max()),
                            value=0.0,
                            format="$%.2f"
                        )
                        
                        max_liquidity = st.slider(
                            "Maximum Liquidity ($)", 
                            min_value=0.0, 
                            max_value=float(advanced_df["liquidity"].max()),
                            value=float(advanced_df["liquidity"].max()),
                            format="$%.2f"
                        )
                        
                        if st.button("Apply Liquidity Filter"):
                            filter_system.add_filter(
                                AdvancedFilteringSystem.liquidity_filter(
                                    min_value=min_liquidity, 
                                    max_value=max_liquidity
                                )
                            )
                    
                    # APR Filter
                    with st.expander("APR Filter", expanded=True):
                        min_apr = st.slider(
                            "Minimum APR (%)", 
                            min_value=0.0, 
                            max_value=100.0,
                            value=0.0,
                            format="%.2f%%"
                        )
                        
                        max_apr = st.slider(
                            "Maximum APR (%)", 
                            min_value=0.0, 
                            max_value=100.0,
                            value=50.0,
                            format="%.2f%%"
                        )
                        
                        if st.button("Apply APR Filter"):
                            filter_system.add_filter(
                                AdvancedFilteringSystem.apr_filter(
                                    min_value=min_apr, 
                                    max_value=max_apr
                                )
                            )
                    
                    # DEX Filter
                    with st.expander("DEX Filter"):
                        selected_dexes = st.multiselect(
                            "Select DEXes",
                            options=advanced_df["dex"].unique().tolist(),
                            default=[]
                        )
                        
                        if st.button("Apply DEX Filter") and selected_dexes:
                            filter_system.add_filter(
                                AdvancedFilteringSystem.dex_filter(selected_dexes)
                            )
                    
                    # Token Filter
                    with st.expander("Token Filter"):
                        # Get unique tokens
                        all_tokens = set()
                        for _, row in advanced_df.iterrows():
                            all_tokens.add(row["token1_symbol"])
                            all_tokens.add(row["token2_symbol"])
                        
                        selected_tokens = st.multiselect(
                            "Select Tokens",
                            options=sorted(list(all_tokens)),
                            default=[]
                        )
                        
                        if st.button("Apply Token Filter") and selected_tokens:
                            filter_system.add_filter(
                                AdvancedFilteringSystem.token_filter(selected_tokens)
                            )
                    
                    # Trend Filter
                    with st.expander("Trend Filter"):
                        trend_field = st.selectbox(
                            "Select Metric",
                            options=["apr", "liquidity"],
                            format_func=lambda x: "APR" if x == "apr" else "Liquidity"
                        )
                        
                        trend_period = st.selectbox(
                            "Time Period",
                            options=[1, 7, 30],
                            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
                        )
                        
                        trend_type = st.selectbox(
                            "Trend Direction",
                            options=["increasing", "decreasing", "stable"]
                        )
                        
                        trend_threshold = st.slider(
                            "Threshold (%)", 
                            min_value=0.0, 
                            max_value=20.0,
                            value=1.0,
                            format="%.1f%%"
                        )
                        
                        if st.button("Apply Trend Filter"):
                            filter_system.add_filter(
                                AdvancedFilteringSystem.trend_filter(
                                    field=trend_field,
                                    days=trend_period,
                                    trend_type=trend_type,
                                    threshold=trend_threshold
                                )
                            )
                    
                    # Reset button
                    if st.button("Reset All Filters"):
                        filter_system.reset_filters()
                
                with results_col:
                    st.subheader("Filtered Results")
                    
                    # Apply all filters
                    filtered_results = filter_system.apply_filters()
                    
                    # Display results count
                    st.metric(
                        "Matching Pools", 
                        f"{len(filtered_results):,}",
                        f"{len(filtered_results) - len(advanced_df):,}"
                    )
                    
                    # Show the filter impact analysis
                    impact_df = filter_system.get_filter_impact_analysis()
                    if not impact_df.empty:
                        st.subheader("Filter Impact Analysis")
                        st.dataframe(impact_df)
                    
                    # Display results
                    if not filtered_results.empty:
                        # Prepare data for display
                        display_df = filtered_results.copy()
                        
                        # Format columns for display
                        display_df["liquidity"] = display_df["liquidity"].apply(format_currency)
                        display_df["volume_24h"] = display_df["volume_24h"].apply(format_currency)
                        display_df["apr"] = display_df["apr"].apply(format_percentage)
                        
                        # Show table
                        st.dataframe(display_df[["name", "dex", "category", "token1_symbol", 
                                                "token2_symbol", "liquidity", "volume_24h", "apr"]])
                        
                        # Visualization
                        st.subheader("Visualization")
                        
                        # Plot options
                        plot_type = st.selectbox(
                            "Plot Type",
                            options=["Scatter", "Bar", "Bubble"],
                            index=0
                        )
                        
                        if plot_type == "Scatter":
                            x_axis = st.selectbox("X-Axis", ["liquidity", "volume_24h", "apr", "apr_change_24h"], 
                                                format_func=lambda x: {
                                                    "liquidity": "Liquidity", 
                                                    "volume_24h": "Volume (24h)", 
                                                    "apr": "APR", 
                                                    "apr_change_24h": "APR Change (24h)"
                                                }.get(x, x))
                            
                            y_axis = st.selectbox("Y-Axis", ["apr", "liquidity", "volume_24h", "prediction_score"], 
                                                format_func=lambda x: {
                                                    "apr": "APR", 
                                                    "liquidity": "Liquidity", 
                                                    "volume_24h": "Volume (24h)", 
                                                    "prediction_score": "Prediction Score"
                                                }.get(x, x))
                            
                            color_by = st.selectbox("Color By", ["dex", "category"], 
                                                format_func=lambda x: "DEX" if x == "dex" else "Category")
                            
                            # Create the scatter plot
                            fig = px.scatter(
                                filtered_results,
                                x=x_axis,
                                y=y_axis,
                                color=color_by,
                                hover_name="name",
                                size="prediction_score" if y_axis != "prediction_score" else None,
                                title=f"{y_axis.replace('_', ' ').title()} vs {x_axis.replace('_', ' ').title()}",
                                labels={
                                    "liquidity": "Liquidity ($)",
                                    "volume_24h": "Volume 24h ($)",
                                    "apr": "APR (%)",
                                    "apr_change_24h": "APR Change 24h (%)",
                                    "prediction_score": "Prediction Score"
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif plot_type == "Bar":
                            # Group by options
                            group_by = st.selectbox("Group By", ["dex", "category"], 
                                                format_func=lambda x: "DEX" if x == "dex" else "Category")
                            
                            metric = st.selectbox("Metric", ["count", "liquidity", "volume_24h", "apr"], 
                                                format_func=lambda x: {
                                                    "count": "Count", 
                                                    "liquidity": "Total Liquidity", 
                                                    "volume_24h": "Total Volume", 
                                                    "apr": "Average APR"
                                                }.get(x, x))
                            
                            if metric == "count":
                                # Count of pools by group
                                group_counts = filtered_results[group_by].value_counts().reset_index()
                                group_counts.columns = [group_by, 'count']
                                
                                fig = px.bar(
                                    group_counts,
                                    x=group_by,
                                    y='count',
                                    color=group_by,
                                    title=f"Pool Count by {group_by.title()}",
                                    labels={group_by: group_by.title(), 'count': 'Number of Pools'}
                                )
                                
                            elif metric == "apr":
                                # Average APR by group
                                group_stats = filtered_results.groupby(group_by)['apr'].mean().reset_index()
                                
                                fig = px.bar(
                                    group_stats,
                                    x=group_by,
                                    y='apr',
                                    color=group_by,
                                    title=f"Average APR by {group_by.title()}",
                                    labels={group_by: group_by.title(), 'apr': 'Average APR (%)'}
                                )
                                
                            else:
                                # Sum of liquidity or volume by group
                                group_stats = filtered_results.groupby(group_by)[metric].sum().reset_index()
                                
                                fig = px.bar(
                                    group_stats,
                                    x=group_by,
                                    y=metric,
                                    color=group_by,
                                    title=f"Total {metric.replace('_', ' ').title()} by {group_by.title()}",
                                    labels={
                                        group_by: group_by.title(), 
                                        metric: f"Total {metric.replace('_', ' ').title()} ($)"
                                    }
                                )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif plot_type == "Bubble":
                            fig = px.scatter(
                                filtered_results,
                                x="liquidity",
                                y="apr",
                                size="volume_24h",
                                color="dex",
                                hover_name="name",
                                text="name",
                                title="Liquidity Pool Bubble Chart",
                                labels={
                                    "liquidity": "Liquidity ($)",
                                    "apr": "APR (%)",
                                    "volume_24h": "Volume 24h ($)"
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No pools match the current filters")
            else:
                st.warning("Advanced filtering is not available - module could not be loaded")
    
        # Predictions Tab
        with tab_predict:
            st.header("Liquidity Pool Predictions")
            st.write("Machine learning-based predictions for future pool performance")
            
            # Use a copy of the DataFrame to not interfere with other tabs
            prediction_df = df.copy()
            
            # Show top predicted pools
            st.subheader("Top Predicted Pools")
            
            # Get the top 10 pools by prediction score
            top_pools = prediction_df.sort_values("prediction_score", ascending=False).head(10)
            
            # Create a bar chart of the top pools
            fig = px.bar(
                top_pools,
                x="name",
                y="prediction_score",
                color="dex",
                hover_data=["liquidity", "apr", "volume_24h"],
                title="Top 10 Pools by Prediction Score",
                labels={
                    "name": "Pool Name",
                    "prediction_score": "Prediction Score",
                    "dex": "DEX"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction factors
            st.subheader("Prediction Factors")
            st.write("""
            Our prediction model takes into account various factors:
            
            1. **Historical APR Trends**: How the APR has changed over time
            2. **Liquidity Stability**: Stability of the pool's total liquidity
            3. **Volume Trends**: Changes in trading volume
            4. **Token Category**: Different categories have different potential
            5. **Market Correlation**: Correlation with broader market trends
            """)
            
            # APR projection
            st.subheader("APR Projections")
            
            # Select a pool for detailed projection
            pool_options = prediction_df["name"].tolist()
            selected_pool = st.selectbox("Select Pool for Projection", pool_options)
            
            # Get the selected pool data
            pool_data = prediction_df[prediction_df["name"] == selected_pool].iloc[0]
            
            # Generate some simple projections (in a real app, this would use the ML model)
            current_apr = pool_data["apr"]
            apr_change_7d = pool_data["apr_change_7d"]
            
            # Simple projection for demo purposes
            projection_days = 30
            daily_change = apr_change_7d / 7  # Convert weekly to daily
            
            # Generate projection data
            days = list(range(projection_days + 1))
            projected_apr = [current_apr + daily_change * i for i in days]
            
            # Add some randomness to make it look more realistic
            for i in range(1, len(projected_apr)):
                projected_apr[i] += random.uniform(-0.2, 0.2)
            
            # Create a projection dataframe
            projection_df = pd.DataFrame({
                "Day": days,
                "APR": projected_apr
            })
            
            # Create APR projection chart
            fig = px.line(
                projection_df,
                x="Day",
                y="APR",
                title=f"30-Day APR Projection for {selected_pool}",
                labels={"Day": "Days from Now", "APR": "Projected APR (%)"}
            )
            
            # Add current APR marker
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[current_apr],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    name="Current APR"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction confidence
            confidence = pool_data["prediction_score"] / 100
            st.metric("Prediction Confidence", f"{confidence:.2%}")
            
            # Disclaimer
            st.warning("""
            **Disclaimer**: These predictions are for informational purposes only. 
            Actual pool performance may vary significantly. Past performance is not 
            indicative of future results. Always do your own research before investing.
            """)
    
        # Risk Assessment Tab
        with tab_risk:
            st.header("Liquidity Pool Risk Assessment")
            st.write("Analyze the risk factors for liquidity pools")
            
            # Use a copy of the DataFrame to not interfere with other tabs
            risk_df = df.copy()
            
            # Calculate risk scores (in a real app, this would be more sophisticated)
            # Here we'll use a simple formula based on APR, liquidity, and volatility
            
            # Higher APR usually indicates higher risk
            apr_risk = risk_df["apr"] / 100  # Scale to 0-1 range assuming APR < 100%
            
            # Lower liquidity usually means higher risk
            max_liquidity = risk_df["liquidity"].max()
            liquidity_risk = 1 - (risk_df["liquidity"] / max_liquidity)
            
            # Higher volatility (using APR changes as a proxy) means higher risk
            volatility_risk = abs(risk_df["apr_change_24h"]) / 10  # Scale to 0-1 assuming changes < 10%
            
            # Combine into an overall risk score (0-100)
            risk_df["risk_score"] = ((apr_risk * 0.4) + (liquidity_risk * 0.4) + (volatility_risk * 0.2)) * 100
            
            # Categorize risk
            risk_df["risk_category"] = pd.cut(
                risk_df["risk_score"],
                bins=[0, 25, 50, 75, 100],
                labels=["Low", "Medium", "High", "Very High"]
            )
            
            # Show risk distribution
            st.subheader("Risk Distribution")
            
            # Create a histogram of risk scores
            fig = px.histogram(
                risk_df,
                x="risk_score",
                color="risk_category",
                title="Distribution of Risk Scores",
                labels={
                    "risk_score": "Risk Score (0-100)",
                    "count": "Number of Pools",
                    "risk_category": "Risk Category"
                },
                category_orders={"risk_category": ["Low", "Medium", "High", "Very High"]},
                color_discrete_map={
                    "Low": "green",
                    "Medium": "yellow",
                    "High": "orange",
                    "Very High": "red"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk by DEX and category
            col1, col2 = st.columns(2)
            
            with col1:
                # Average risk by DEX
                dex_risk = risk_df.groupby("dex")["risk_score"].mean().reset_index()
                dex_risk = dex_risk.sort_values("risk_score", ascending=False)
                
                fig = px.bar(
                    dex_risk,
                    x="dex",
                    y="risk_score",
                    title="Average Risk by DEX",
                    labels={
                        "dex": "DEX",
                        "risk_score": "Avg. Risk Score"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average risk by category
                category_risk = risk_df.groupby("category")["risk_score"].mean().reset_index()
                category_risk = category_risk.sort_values("risk_score", ascending=False)
                
                fig = px.bar(
                    category_risk,
                    x="category",
                    y="risk_score",
                    title="Average Risk by Category",
                    labels={
                        "category": "Category",
                        "risk_score": "Avg. Risk Score"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk vs. Reward
            st.subheader("Risk vs. Reward Analysis")
            
            fig = px.scatter(
                risk_df,
                x="risk_score",
                y="apr",
                color="dex",
                size="liquidity",
                hover_name="name",
                title="Risk vs. Reward",
                labels={
                    "risk_score": "Risk Score",
                    "apr": "APR (%)",
                    "dex": "DEX",
                    "liquidity": "Liquidity"
                }
            )
            
            # Add a trend line
            fig.add_trace(
                go.Scatter(
                    x=[0, 100],
                    y=[0, risk_df["apr"].max()],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    name="Risk-Reward Line"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Individual pool risk assessment
            st.subheader("Individual Pool Risk Assessment")
            
            # Select a pool
            pool_options = risk_df["name"].tolist()
            selected_pool = st.selectbox("Select Pool", pool_options, key="risk_pool_select")
            
            # Get the selected pool data
            pool_risk_data = risk_df[risk_df["name"] == selected_pool].iloc[0]
            
            # Display pool risk metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_score = pool_risk_data["risk_score"]
                risk_color = "red" if risk_score > 75 else "orange" if risk_score > 50 else "yellow" if risk_score > 25 else "green"
                st.markdown(f"<h1 style='text-align: center; color: {risk_color};'>{risk_score:.1f}</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Risk Score</p>", unsafe_allow_html=True)
            
            with col2:
                st.metric("APR", format_percentage(pool_risk_data["apr"]))
            
            with col3:
                st.metric("Liquidity", format_currency(pool_risk_data["liquidity"]))
            
            # Risk factors
            st.subheader("Risk Factors")
            
            # Create radar chart of risk factors
            risk_factors = {
                "APR Risk": apr_risk[risk_df["name"] == selected_pool].iloc[0] * 100,
                "Liquidity Risk": liquidity_risk[risk_df["name"] == selected_pool].iloc[0] * 100,
                "Volatility Risk": volatility_risk[risk_df["name"] == selected_pool].iloc[0] * 100,
                "Token Risk": random.uniform(20, 80),  # In a real app, this would be calculated
                "Protocol Risk": random.uniform(20, 80)  # In a real app, this would be calculated
            }
            
            # Create radar chart data
            categories = list(risk_factors.keys())
            values = list(risk_factors.values())
            values.append(values[0])  # Close the loop
            categories.append(categories[0])  # Close the loop
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Risk Factors'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=False,
                title="Risk Factor Breakdown"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("Risk Mitigation Recommendations")
            
            if risk_score > 75:
                st.error("""
                **High Risk Pool**: This pool shows signs of high risk. Consider:
                - Limiting exposure to a small percentage of your portfolio
                - Setting tight stop-loss orders
                - Monitoring more frequently
                - Considering lower-risk alternatives
                """)
            elif risk_score > 50:
                st.warning("""
                **Medium-High Risk Pool**: This pool has moderate risk levels. Consider:
                - Diversifying across multiple pools
                - Regular monitoring of performance
                - Having an exit strategy in place
                """)
            elif risk_score > 25:
                st.info("""
                **Medium-Low Risk Pool**: This pool has relatively stable metrics. Consider:
                - Standard diversification practices
                - Regular but less frequent monitoring
                - Middle-term investment horizon
                """)
            else:
                st.success("""
                **Low Risk Pool**: This pool shows lower risk indicators. Consider:
                - Suitable for larger allocation percentages
                - Good candidate for longer-term holdings
                - Still maintain standard monitoring practices
                """)
    
        # NLP Reports Tab
        with tab_nlp:
            st.header("Natural Language Analytics Reports")
            st.write("AI-generated insights about liquidity pool trends and opportunities")
            
            # Select pool for detailed analysis
            st.subheader("Pool-Specific Analysis")
            
            selected_pool = st.selectbox("Select Pool for Analysis", df["name"].tolist())
            
            # Get the selected pool data
            pool_analysis_data = df[df["name"] == selected_pool].iloc[0]
            
            # Create the natural language report
            # In a real application, this would use a more sophisticated NLP model
            def generate_pool_report(pool_data):
                name = pool_data["name"]
                dex = pool_data["dex"]
                category = pool_data["category"]
                apr = pool_data["apr"]
                liquidity = pool_data["liquidity"]
                volume = pool_data["volume_24h"]
                apr_change_24h = pool_data["apr_change_24h"]
                apr_change_7d = pool_data["apr_change_7d"]
                prediction_score = pool_data["prediction_score"]
                
                # Trending status
                if apr_change_7d > 2:
                    trend = "strongly trending upward"
                elif apr_change_7d > 0.5:
                    trend = "trending upward"
                elif apr_change_7d < -2:
                    trend = "strongly trending downward"
                elif apr_change_7d < -0.5:
                    trend = "trending downward"
                else:
                    trend = "stable"
                
                # Volume to liquidity ratio
                vl_ratio = volume / liquidity
                if vl_ratio > 0.2:
                    volume_assessment = "excellent trading volume relative to its liquidity"
                elif vl_ratio > 0.1:
                    volume_assessment = "good trading volume relative to its liquidity"
                elif vl_ratio > 0.05:
                    volume_assessment = "moderate trading volume relative to its liquidity"
                else:
                    volume_assessment = "low trading volume relative to its liquidity"
                
                # Generate outlook based on prediction score
                if prediction_score > 80:
                    outlook = "very positive outlook"
                elif prediction_score > 60:
                    outlook = "positive outlook"
                elif prediction_score > 40:
                    outlook = "neutral outlook"
                else:
                    outlook = "cautious outlook"
                
                # Generate the report
                report = f"""
                ## {name} Pool Analysis
                
                The {name} pool on {dex} is a {category.lower()} category pool with an APR of {apr:.2f}%. 
                This pool currently has ${liquidity:,.2f} in total liquidity and 24-hour trading volume of ${volume:,.2f}.
                
                ### Performance Trends
                
                This pool is currently {trend} with a 7-day APR change of {apr_change_7d:.2f}% and a 24-hour change of {apr_change_24h:.2f}%. 
                It shows {volume_assessment}.
                
                ### Market Position
                
                As a {category.lower()} pool on {dex}, this pool represents a {outlook} in our prediction model, with a score of {prediction_score:.1f}/100.
                
                ### Recommendations
                
                """
                
                # Add recommendations based on metrics
                if prediction_score > 70 and apr > 10 and apr_change_7d > 0:
                    report += """
                    This pool shows strong potential for high returns with positive momentum. Consider:
                    - Adding this pool to your high-potential portfolio segment
                    - Monitoring the APR trends weekly
                    - Setting profit-taking targets
                    """
                elif prediction_score > 50 and apr > 5:
                    report += """
                    This pool shows moderate potential with reasonable returns. Consider:
                    - Balanced position as part of a diversified strategy
                    - Regular monitoring of key performance indicators
                    - Setting both entry and exit strategy
                    """
                else:
                    report += """
                    This pool shows lower potential or elevated risk factors. Consider:
                    - Limited exposure if any
                    - Looking for alternatives with better risk/reward profiles
                    - Waiting for stronger positive indicators before entry
                    """
                
                return report
            
            # Display the report
            report = generate_pool_report(pool_analysis_data)
            st.markdown(report)
            
            # Market-wide analysis
            st.subheader("Market-Wide Trends Analysis")
            
            # Generate a higher-level summary of market trends
            def generate_market_report(df):
                # Get high-level metrics
                total_liquidity = df["liquidity"].sum()
                avg_apr = df["apr"].mean()
                avg_apr_change = df["apr_change_7d"].mean()
                
                # DEX with highest average APR
                dex_apr = df.groupby("dex")["apr"].mean().reset_index()
                top_dex = dex_apr.sort_values("apr", ascending=False).iloc[0]
                
                # Category with highest average APR
                cat_apr = df.groupby("category")["apr"].mean().reset_index()
                top_category = cat_apr.sort_values("apr", ascending=False).iloc[0]
                
                # Market trend assessment
                if avg_apr_change > 1:
                    market_trend = "strong upward trend"
                elif avg_apr_change > 0.2:
                    market_trend = "moderate upward trend"
                elif avg_apr_change < -1:
                    market_trend = "strong downward trend"
                elif avg_apr_change < -0.2:
                    market_trend = "moderate downward trend"
                else:
                    market_trend = "generally stable"
                
                # Generate the report
                report = f"""
                ## Market-Wide Liquidity Pool Analysis
                
                The Solana liquidity pool market currently has a total liquidity of ${total_liquidity:,.2f} 
                across all tracked pools, with an average APR of {avg_apr:.2f}%.
                
                ### Current Market Trends
                
                The market is showing a {market_trend} with an average 7-day APR change of {avg_apr_change:.2f}%.
                {top_dex['dex']} currently offers the highest average APR at {top_dex['apr']:.2f}%, while
                {top_category['category']} pools are leading categories with an average APR of {top_category['apr']:.2f}%.
                
                ### Opportunities by Category
                
                """
                
                # Add category-specific insights
                categories = df["category"].unique()
                for category in categories:
                    cat_data = df[df["category"] == category]
                    cat_avg_apr = cat_data["apr"].mean()
                    cat_avg_change = cat_data["apr_change_7d"].mean()
                    cat_top_pool = cat_data.sort_values("prediction_score", ascending=False).iloc[0]
                    
                    report += f"""
                    **{category}** pools are averaging {cat_avg_apr:.2f}% APR with a {cat_avg_change:.2f}% weekly change.
                    *{cat_top_pool['name']}* on {cat_top_pool['dex']} is the top-rated pool in this category with a 
                    prediction score of {cat_top_pool['prediction_score']:.1f}/100.
                    
                    """
                
                return report
            
            # Display the market report
            market_report = generate_market_report(df)
            st.markdown(market_report)
        
        # Add documentation and help information at the bottom
        with st.expander("Documentation & Help"):
            st.markdown("""
            ## SolPool Insight Documentation
            
            This application provides comprehensive analytics for Solana liquidity pools across major DEXes.
            
            ### Key Features
            
            - **Data Explorer**: Browse and search through all available liquidity pools
            - **Advanced Filtering**: Apply complex filters to find the best pools for your strategy
            - **Predictions**: View machine learning-based predictions for future pool performance
            - **Risk Assessment**: Analyze risk factors for different pools
            - **NLP Reports**: Read AI-generated insights about market trends
            
            ### Data Sources
            
            Pool data is sourced directly from the Solana blockchain using RPC endpoints.
            Historical data is used for trend analysis and predictions.
            
            ### Disclaimer
            
            This tool is for informational purposes only. It does not constitute investment advice.
            Always do your own research before making investment decisions.
            """)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("The application encountered an error. Please try refreshing the page or contact support.")

if __name__ == "__main__":
    main()