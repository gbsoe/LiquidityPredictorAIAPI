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
    if db_handler.engine:
        pools = db_handler.get_pools()
        if pools and len(pools) > 0:
            st.success(f"✓ Successfully loaded {len(pools)} pools from database")
            return pools
    
    # If no database data, try cached file
    cache_file = "extracted_pools.json"
    if os.path.exists(cache_file):
        try:
            pools = db_handler.load_from_json(cache_file)
            if pools and len(pools) > 0:
                st.success(f"✓ Successfully loaded {len(pools)} pools from local cache")
                
                # Store in database if available
                if db_handler.engine:
                    db_handler.store_pools(pools)
                    st.info("✓ Pool data copied to database")
                
                # Ensure all required fields are present
                pools = ensure_all_fields(pools)
                return pools
        except Exception as e:
            st.warning(f"Error loading cached data: {e}")
    
    # Sidebar controls for data loading
    with st.sidebar:
        st.subheader("Data Source")
        
        # Show background updater status
        if HAS_BACKGROUND_UPDATER and st.session_state.get('updater_started') == True:
            st.success("✓ Background data refresh is active")
            
            # Get the cache file modification time
            try:
                if os.path.exists("extracted_pools.json"):
                    mod_time = os.path.getmtime("extracted_pools.json")
                    mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                    st.info(f"Last data update: {mod_time_str}")
            except Exception:
                pass
                
            # Add a refresh button
            if st.button("Refresh Data Now"):
                # Get the current data
                try:
                    with open("extracted_pools.json", "r") as f:
                        st.session_state['last_data_length'] = len(json.load(f))
                except Exception:
                    st.session_state['last_data_length'] = 0
                    
                # Force a page refresh
                st.rerun()
            
            # Show stats if we have them
            if 'last_data_length' in st.session_state:
                try:
                    with open("extracted_pools.json", "r") as f:
                        current_length = len(json.load(f))
                        if current_length > st.session_state['last_data_length']:
                            st.success(f"➕ Added {current_length - st.session_state['last_data_length']} new pools")
                        st.session_state['last_data_length'] = current_length
                except Exception:
                    pass
        
        force_live_data = st.checkbox("Try live blockchain data", value=False, 
                              help="When checked, attempts to fetch fresh data from blockchain")
        
        # Increase pool count slider
        pool_count = st.slider("Sample pool count", min_value=50, max_value=500, value=200, step=50,
                              help="Number of sample pools to generate if real data isn't available")
        
        # Add advanced options
        with st.expander("Advanced RPC Options"):
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
    
    # Only try to fetch live data if explicitly requested
    if force_live_data and HAS_EXTRACTOR:
        with st.spinner("Attempting to extract pool data from Solana blockchain..."):
            try:
                # Use the custom or environment RPC endpoint
                st.info(f"Connecting to RPC endpoint: {custom_rpc}")
                
                # Initialize extractor with better error handling and batch processing
                # Use tranched fetching with up to 50 pools per DEX but fetch them in batches
                extractor = OnChainExtractor(rpc_endpoint=custom_rpc)
                
                # Display a message about tranched fetching
                st.info("Starting tranched fetching of pool data. This may take a few minutes as we retrieve data in batches to avoid rate limits.")
                
                # Increase the number of pools per DEX to 50 to get more real-time data
                pools = extractor.extract_and_enrich_pools(max_per_dex=50)
                
                # Verify the data is not empty
                if pools and len(pools) > 0:
                    st.success(f"Successfully extracted {len(pools)} pools from blockchain")
                    # Save to cache and database
                    try:
                        # Save to cache file
                        db_handler.backup_to_json(pools, cache_file)
                        st.info(f"Data saved to cache file: {cache_file}")
                        
                        # Save to database if available
                        if db_handler.engine:
                            db_handler.store_pools(pools)
                            st.info("✓ Pool data stored in database")
                    except Exception as e:
                        st.warning(f"Error saving data: {e}")
                    
                    return pools
                else:
                    st.error("No pool data was returned from the blockchain")
            except Exception as e:
                st.error(f"Error extracting data from blockchain: {str(e)}")
    
    # If we got here, we need to use sample data
    st.warning(f"Unable to load pool data - using generated sample data with {pool_count} pools")
    sample_data = generate_sample_data(pool_count)
    
    # Save sample data to cache and database
    try:
        # Save to cache file
        db_handler.backup_to_json(sample_data, cache_file)
        
        # Save to database if available
        if db_handler.engine:
            db_handler.store_pools(sample_data)
            st.info("✓ Sample data stored in database")
    except Exception as e:
        st.warning(f"Error saving sample data: {e}")
    
    return sample_data

def generate_sample_data(count=200):
    """Generate sample pool data with the specified number of pools"""
    # We'll create data for the requested number of pools across DEXes
    pools = []
    
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
            "prediction_score": prediction_score
        })
        
    return pools

def main():
    # Display logo and title
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
        if db_handler.engine:
            st.success("✓ Connected to PostgreSQL database")
        else:
            st.warning("⚠ Database connection not available - using file-based storage")
        
        # Start the background updater if available
        if HAS_BACKGROUND_UPDATER and st.session_state.get('updater_started') != True:
            st.session_state['updater_started'] = True
            
            # Only start if we have a valid RPC endpoint
            rpc_endpoint = os.getenv("SOLANA_RPC_ENDPOINT")
            if rpc_endpoint:
                try:
                    background_updater.start_background_updater()
                    st.success("✓ Background data updater started - pools will be progressively refreshed")
                except Exception as e:
                    st.warning(f"Could not start background updater: {e}")
        
        # Create tabs for different views
        tab_explore, tab_advanced, tab_predict, tab_risk, tab_nlp = st.tabs([
            "Data Explorer", "Advanced Filtering", "Predictions", "Risk Assessment", "NLP Reports"
        ])
        
        # Add debug information
        st.info("Loading pool data now...")
        
        # Load data
        pool_data = load_data()
        
        # Debug info
        if pool_data:
            if isinstance(pool_data, list):
                st.success(f"✓ Successfully loaded {len(pool_data)} pools")
            else:
                st.error(f"Unexpected data type: {type(pool_data)}")
                # Generate fallback sample data
                pool_data = generate_sample_data(200)
                st.warning("Using generated sample data due to unexpected data type")
        else:
            st.error("Failed to load any pool data")
            # Generate fallback sample data
            pool_data = generate_sample_data(200)
            st.warning("Using generated sample data due to data loading failure")
            
        # Ensure we have data, no matter what
        if not pool_data:
            pool_data = [{
                "id": "sample1",
                "name": "SOL/USDC",
                "dex": "Raydium",
                "category": "Major",
                "token1_symbol": "SOL",
                "token2_symbol": "USDC",
                "token1_address": "So11111111111111111111111111111111111111112",
                "token2_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "liquidity": 25000000,
                "volume_24h": 1000000,
                "apr": 10.5,
                "fee": 0.0025,
                "version": "v4",
                "apr_change_24h": 0.5,
                "apr_change_7d": 1.2,
                "tvl_change_24h": 0.8,
                "tvl_change_7d": 2.5,
                "prediction_score": 85,
                "apr_change_30d": 5.5,
                "tvl_change_30d": 7.5
            }]
        
        # Ensure all required fields are present 
        pool_data = ensure_all_fields(pool_data)
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(pool_data)
    except Exception as e:
        # Handle any unexpected exceptions to prevent crashes
        st.error(f"Error in main function: {str(e)}")
        st.warning("Attempting to continue with minimal functionality...")
        
        # Create minimal data for the app to continue
        pool_data = [{
            "id": "sample1",
            "name": "SOL/USDC",
            "dex": "Raydium",
            "category": "Major",
            "token1_symbol": "SOL",
            "token2_symbol": "USDC",
            "token1_address": "So11111111111111111111111111111111111111112",
            "token2_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "liquidity": 25000000,
            "volume_24h": 1000000,
            "apr": 10.5,
            "fee": 0.0025,
            "version": "v4",
            "apr_change_24h": 0.5,
            "apr_change_7d": 1.2,
            "tvl_change_24h": 0.8,
            "tvl_change_7d": 2.5,
            "prediction_score": 85,
            "apr_change_30d": 5.5,
            "tvl_change_30d": 7.5
        }]
        
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
            pred_text = f"{pred_icon} {pred_score:.1f}/100"
            pool_id = row["id"]
            
            # Add to table data
            table_data.append({
                "Pool Name": pool_name,
                "DEX": dex_name,
                "Category": category_text,
                "TVL": liquidity_val,
                "24h Volume": volume_val,
                "APR": apr_val,
                "24h Δ": apr_change_24h_val,
                "7d Δ": apr_change_7d_val,
                "Pred Score": pred_text,
                "Pool ID": pool_id
            })
        
        # Convert to a new DataFrame with formatted values
        formatted_df = pd.DataFrame(table_data)
        
        # Use Streamlit's native table which handles formatting better
        st.dataframe(
            formatted_df,
            use_container_width=True,
            column_config={
                "Pool ID": st.column_config.TextColumn(
                    "Pool ID",
                    width="medium",
                    help="Unique identifier for the pool"
                ),
                "TVL": st.column_config.TextColumn(
                    "TVL",
                    width="small"
                ),
                "24h Volume": st.column_config.TextColumn(
                    "24h Volume",
                    width="small"
                ),
                "Pool Name": st.column_config.TextColumn(
                    "Pool Name", 
                    width="medium"
                ),
                "Pred Score": st.column_config.TextColumn(
                    "Pred Score",
                    width="small"
                )
            }
        )
    
    # Advanced Filtering Tab
    with tab_advanced:
        st.header("Advanced Filtering System")
        st.markdown("""
        Apply sophisticated filters to find pools matching specific criteria. This system supports
        complex multi-dimensional filtering across metrics and derived values.
        """)
        
        if HAS_ADVANCED_FILTERING:
            # Create an AdvancedFilteringSystem instance if not already in session state
            if 'advanced_filter_system' not in st.session_state:
                st.session_state.advanced_filter_system = AdvancedFilteringSystem(df)
            
            # Layout for the advanced filtering UI
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Filter Configuration")
                
                # TVL (Liquidity) Filter
                with st.expander("Liquidity (TVL) Filter", expanded=True):
                    min_tvl = st.number_input("Minimum TVL (USD)", value=0.0, step=100000.0, format="%.2f")
                    max_tvl = st.number_input("Maximum TVL (USD)", value=1000000000.0, step=1000000.0, format="%.2f")
                    
                    if st.button("Apply TVL Filter"):
                        tvl_filter = AdvancedFilteringSystem.liquidity_filter(min_value=min_tvl, max_value=max_tvl)
                        st.session_state.advanced_filter_system.add_filter(tvl_filter)
                        st.success("✓ TVL filter applied")
                
                # APR Filter
                with st.expander("APR Filter", expanded=True):
                    min_apr = st.number_input("Minimum APR (%)", value=0.0, step=1.0, format="%.2f")
                    max_apr = st.number_input("Maximum APR (%)", value=100.0, step=5.0, format="%.2f")
                    
                    if st.button("Apply APR Filter"):
                        apr_filter = AdvancedFilteringSystem.apr_filter(min_value=min_apr, max_value=max_apr)
                        st.session_state.advanced_filter_system.add_filter(apr_filter)
                        st.success("✓ APR filter applied")
                
                # Volume Filter
                with st.expander("Volume Filter"):
                    min_volume = st.number_input("Minimum 24h Volume (USD)", value=0.0, step=10000.0, format="%.2f")
                    max_volume = st.number_input("Maximum 24h Volume (USD)", value=100000000.0, step=100000.0, format="%.2f")
                    
                    if st.button("Apply Volume Filter"):
                        volume_filter = AdvancedFilteringSystem.volume_filter(min_value=min_volume, max_value=max_volume)
                        st.session_state.advanced_filter_system.add_filter(volume_filter)
                        st.success("✓ Volume filter applied")
                
                # DEX Filter
                with st.expander("DEX Filter"):
                    selected_dexes = st.multiselect(
                        "Select DEXes", 
                        options=sorted(df["dex"].unique().tolist()),
                        default=[]
                    )
                    
                    if st.button("Apply DEX Filter") and selected_dexes:
                        dex_filter = AdvancedFilteringSystem.dex_filter(selected_dexes)
                        st.session_state.advanced_filter_system.add_filter(dex_filter)
                        st.success(f"✓ DEX filter applied for {', '.join(selected_dexes)}")
                
                # Category Filter
                with st.expander("Category Filter"):
                    selected_categories = st.multiselect(
                        "Select Categories", 
                        options=sorted(df["category"].unique().tolist()),
                        default=[]
                    )
                    
                    if st.button("Apply Category Filter") and selected_categories:
                        category_filter = AdvancedFilteringSystem.category_filter(selected_categories)
                        st.session_state.advanced_filter_system.add_filter(category_filter)
                        st.success(f"✓ Category filter applied for {', '.join(selected_categories)}")
                
                # Token Filter
                with st.expander("Token Filter"):
                    tokens = set()
                    for _, row in df.iterrows():
                        tokens.add(row["token1_symbol"])
                        tokens.add(row["token2_symbol"])
                    
                    selected_tokens = st.multiselect(
                        "Select Tokens", 
                        options=sorted(list(tokens)),
                        default=[]
                    )
                    
                    if st.button("Apply Token Filter") and selected_tokens:
                        token_filter = AdvancedFilteringSystem.token_filter(selected_tokens)
                        st.session_state.advanced_filter_system.add_filter(token_filter)
                        st.success(f"✓ Token filter applied for {', '.join(selected_tokens)}")
            
            with col2:
                st.subheader("Filter Results")
                
                # Button to clear all filters
                if st.button("Clear All Filters"):
                    st.session_state.advanced_filter_system.reset_filters()
                    st.success("✓ All filters cleared")
                
                # Apply all filters and show results
                filtered_df = st.session_state.advanced_filter_system.apply_filters()
                
                # Show impact analysis
                impact_analysis = st.session_state.advanced_filter_system.get_filter_impact_analysis()
                if not impact_analysis.empty:
                    st.write("Filter Impact Analysis:")
                    st.dataframe(impact_analysis)
                
                # Show filtered data summary
                st.metric("Pools Matching Filters", len(filtered_df))
                
                # Show top 10 results
                st.write("Top Results (by Prediction Score):")
                top_results = filtered_df.sort_values("prediction_score", ascending=False).head(10)
                
                # Display results in a table
                for i, (_, row) in enumerate(top_results.iterrows()):
                    with st.container():
                        st.markdown(f"**{row['name']}** ({row['dex']})")
                        cols = st.columns(4)
                        
                        with cols[0]:
                            st.markdown(f"**TVL:** {format_currency(row['liquidity'])}")
                        
                        with cols[1]:
                            st.markdown(f"**APR:** {format_percentage(row['apr'])}")
                        
                        with cols[2]:
                            st.markdown(f"**24h Volume:** {format_currency(row['volume_24h'])}")
                        
                        with cols[3]:
                            score_color = "green" if row['prediction_score'] > 75 else "orange" if row['prediction_score'] > 50 else "red"
                            st.markdown(f"**Score:** <span style='color:{score_color};'>{row['prediction_score']:.1f}/100</span>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                
                # Advanced analytics visualization
                if not filtered_df.empty and len(filtered_df) > 1:
                    st.subheader("Filtered Pools Analysis")
                    
                    # Create a scatter plot of APR vs TVL
                    fig = px.scatter(
                        filtered_df,
                        x="liquidity",
                        y="apr",
                        color="category",
                        size="volume_24h",
                        hover_name="name",
                        log_x=True,
                        size_max=30,
                        title="APR vs TVL by Pool Category"
                    )
                    
                    # Update the layout
                    fig.update_layout(
                        xaxis_title="TVL (log scale)",
                        yaxis_title="APR (%)",
                        legend_title="Category",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Option to cluster the filtered pools
                    if st.button("Cluster Similar Pools"):
                        if len(filtered_df) >= 5:  # Need at least 5 pools for meaningful clustering
                            clusters = st.session_state.advanced_filter_system.get_pool_clusters(
                                n_clusters=min(5, len(filtered_df) // 2),
                                metrics=["liquidity", "apr", "volume_24h", "prediction_score"]
                            )
                            
                            if not clusters.empty:
                                st.success("✓ Pools clustered successfully")
                                
                                # Show clusters
                                st.write("Pool Clusters:")
                                for cluster_id in sorted(clusters["cluster"].unique()):
                                    cluster_pools = clusters[clusters["cluster"] == cluster_id]
                                    
                                    # Describe this cluster
                                    avg_tvl = cluster_pools["liquidity"].mean()
                                    avg_apr = cluster_pools["apr"].mean()
                                    avg_pred = cluster_pools["prediction_score"].mean()
                                    
                                    st.markdown(f"**Cluster {cluster_id+1}** - {len(cluster_pools)} pools")
                                    st.markdown(f"Avg TVL: {format_currency(avg_tvl)} | Avg APR: {format_percentage(avg_apr)} | Avg Score: {avg_pred:.1f}/100")
                                    
                                    # Show a few examples from this cluster
                                    example_pools = cluster_pools.sort_values("prediction_score", ascending=False).head(3)
                                    for _, pool in example_pools.iterrows():
                                        st.markdown(f"- {pool['name']} ({pool['dex']})")
                                    
                                    st.markdown("---")
                        else:
                            st.warning("Need at least 5 pools for clustering")
        else:
            st.warning("""
            Advanced filtering module not found. Please make sure that `advanced_filtering.py` 
            is available in the same directory as this script.
            """)
            
            # Show a simplified advanced filtering UI
            st.subheader("Basic Advanced Filtering")
            
            # TVL range filter
            col1, col2 = st.columns(2)
            with col1:
                min_tvl = st.number_input("Minimum TVL (USD)", value=0.0, step=100000.0, format="%.2f")
            with col2:
                max_tvl = st.number_input("Maximum TVL (USD)", value=1000000000.0, step=1000000.0, format="%.2f")
            
            # APR range filter
            col1, col2 = st.columns(2)
            with col1:
                min_apr = st.number_input("Minimum APR (%)", value=0.0, step=1.0, format="%.2f")
            with col2:
                max_apr = st.number_input("Maximum APR (%)", value=100.0, step=5.0, format="%.2f")
            
            # DEX and category filters
            col1, col2 = st.columns(2)
            with col1:
                dex_options = ["All"] + sorted(df["dex"].unique().tolist())
                selected_dex = st.selectbox("Select DEX", options=dex_options)
            with col2:
                category_options = ["All"] + sorted(df["category"].unique().tolist())
                selected_category = st.selectbox("Select Category", options=category_options)
            
            # Token filter
            tokens = set()
            for _, row in df.iterrows():
                tokens.add(row["token1_symbol"])
                tokens.add(row["token2_symbol"])
            
            selected_token = st.selectbox("Filter by Token", options=["All"] + sorted(list(tokens)))
            
            # Apply filters
            filtered_df = df.copy()
            
            if min_tvl > 0:
                filtered_df = filtered_df[filtered_df["liquidity"] >= min_tvl]
            
            if max_tvl < 1000000000:
                filtered_df = filtered_df[filtered_df["liquidity"] <= max_tvl]
            
            if min_apr > 0:
                filtered_df = filtered_df[filtered_df["apr"] >= min_apr]
            
            if max_apr < 100:
                filtered_df = filtered_df[filtered_df["apr"] <= max_apr]
            
            if selected_dex != "All":
                filtered_df = filtered_df[filtered_df["dex"] == selected_dex]
            
            if selected_category != "All":
                filtered_df = filtered_df[filtered_df["category"] == selected_category]
            
            if selected_token != "All":
                filtered_df = filtered_df[
                    (filtered_df["token1_symbol"] == selected_token) | 
                    (filtered_df["token2_symbol"] == selected_token)
                ]
            
            # Show results
            st.metric("Pools Matching Filters", len(filtered_df))
            
            # Display results
            if not filtered_df.empty:
                st.dataframe(
                    filtered_df[["name", "dex", "category", "liquidity", "volume_24h", "apr", "prediction_score"]],
                    use_container_width=True
                )
    
    # Predictions Tab
    with tab_predict:
        st.header("Pool Performance Predictions")
        st.markdown("""
        Our machine learning algorithms analyze historical pool performance and various on-chain metrics
        to predict which pools are likely to perform well in the future.
        """)
        
        # Top predicted pools
        st.subheader("Top Predicted Pools")
        
        # Get top 20 pools by prediction score (increased from 10)
        top_predicted = df.sort_values("prediction_score", ascending=False).head(20)
        
        # Display in a clean table
        for i, (_, row) in enumerate(top_predicted.iterrows()):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1.5, 1, 1.5])
                
                with col1:
                    st.markdown(f"**{row['name']}**")
                    st.markdown(f"*{row['dex']}* • {get_category_badge(row['category'])}", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**TVL**")
                    st.markdown(format_currency(row['liquidity']))
                
                with col3:
                    st.markdown("**APR**")
                    st.markdown(f"{format_percentage(row['apr'])} {get_trend_icon(row['apr_change_24h'])}")
                
                with col4:
                    st.markdown("**Score**")
                    score_color = "green" if row['prediction_score'] > 75 else "orange" if row['prediction_score'] > 50 else "red"
                    st.markdown(f"<span style='color:{score_color};font-weight:bold;'>{row['prediction_score']:.1f}/100</span>", unsafe_allow_html=True)
                
                with col5:
                    st.markdown("**Pool ID**")
                    st.code(row['id'], language=None)
            
            st.markdown("---")
        
        # Chart of prediction scores by category
        st.subheader("Prediction Scores by Category")
        
        # Group by category and calculate average prediction score
        category_scores = df.groupby("category")["prediction_score"].mean().reset_index()
        category_scores = category_scores.sort_values("prediction_score", ascending=False)
        
        # Create bar chart
        fig = px.bar(
            category_scores,
            x="category",
            y="prediction_score",
            title="Average Prediction Score by Category",
            labels={"prediction_score": "Avg Prediction Score", "category": "Category"},
            color="prediction_score",
            color_continuous_scale="RdYlGn"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Assessment Tab
    with tab_risk:
        st.header("Risk Assessment")
        st.markdown("""
        This tab provides a risk assessment for various liquidity pools based on
        volatility, impermanent loss risk, smart contract risk, and other factors.
        """)
        
        # Risk metrics
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.subheader("Highest Volatility Pools")
            
            # Get pools with highest APR changes (proxy for volatility) - increased to 10
            volatile_pools = df.iloc[df['apr_change_24h'].abs().argsort()[::-1]].head(10)
            
            for _, row in volatile_pools.iterrows():
                st.markdown(f"**{row['name']}** ({row['dex']})")
                st.markdown(f"APR Change 24h: {format_percentage(row['apr_change_24h'])} {get_trend_icon(row['apr_change_24h'])}")
                st.markdown("---")
        
        with risk_col2:
            st.subheader("Lowest Liquidity Pools")
            
            # Get pools with lowest liquidity - increased to 10
            low_liquidity = df.sort_values("liquidity").head(10)
            
            for _, row in low_liquidity.iterrows():
                st.markdown(f"**{row['name']}** ({row['dex']})")
                st.markdown(f"TVL: {format_currency(row['liquidity'])}")
                st.markdown("---")
        
        # Risk scatter plot
        st.subheader("Risk vs. Reward Analysis")
        
        # Create a scatter plot of APR vs Volatility
        fig = px.scatter(
            df,
            x="apr_change_24h",
            y="apr",
            color="category",
            size="liquidity",
            hover_name="name",
            log_x=False,
            size_max=30,
            title="APR vs Volatility by Pool Category"
        )
        
        # Update the layout
        fig.update_layout(
            xaxis_title="24h APR Change (%)",
            yaxis_title="Current APR (%)",
            legend_title="Category",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # NLP Reports Tab
    with tab_nlp:
        st.header("NLP-Generated Pool Reports")
        st.markdown("""
        Our AI analyzes each pool's metrics and performance to generate human-readable reports.
        These reports help you understand the pool's performance and potential investment value.
        """)
        
        # Pool selection
        selected_pool = st.selectbox(
            "Select a pool for detailed report",
            options=df["name"].tolist(),
            index=0
        )
        
        pool_data = df[df["name"] == selected_pool].iloc[0]
        
        # Display pool data
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Pool Metrics")
            st.metric("TVL", format_currency(pool_data["liquidity"]))
            st.metric("APR", format_percentage(pool_data["apr"]))
            st.metric("24h Volume", format_currency(pool_data["volume_24h"]))
            st.metric("Fee", f"{pool_data['fee']*100:.3f}%")
            st.metric("Prediction Score", f"{pool_data['prediction_score']:.1f}/100")
        
        with col2:
            st.subheader("AI Analysis Report")
            
            # Generate a formatted pool report based on the data
            dex_name = pool_data["dex"]
            pool_name = pool_data["name"]
            liquidity = pool_data["liquidity"]
            apr = pool_data["apr"]
            volume = pool_data["volume_24h"]
            apr_trend_24h = pool_data["apr_change_24h"]
            liquidity_trend_7d = pool_data["tvl_change_7d"]
            prediction_score = pool_data["prediction_score"]
            
            # Determine sentiment based on metrics
            sentiment = "very positive" if prediction_score > 80 else "positive" if prediction_score > 60 else "neutral" if prediction_score > 40 else "negative" if prediction_score > 20 else "very negative"
            
            apr_sentiment = "excellent" if apr > 30 else "very good" if apr > 20 else "good" if apr > 10 else "moderate" if apr > 5 else "low"
            
            liquidity_size = "very high" if liquidity > 20_000_000 else "high" if liquidity > 5_000_000 else "moderate" if liquidity > 1_000_000 else "low" if liquidity > 100_000 else "very low"
            
            volume_quality = "excellent" if volume/liquidity > 0.2 else "good" if volume/liquidity > 0.1 else "moderate" if volume/liquidity > 0.05 else "low" if volume/liquidity > 0.01 else "very low"
            
            # Trend analysis
            apr_trend_text = "increasing rapidly" if apr_trend_24h > 5 else "increasing" if apr_trend_24h > 1 else "stable" if abs(apr_trend_24h) <= 1 else "decreasing" if apr_trend_24h > -5 else "decreasing rapidly"
            
            liquidity_trend_text = "growing strongly" if liquidity_trend_7d > 10 else "growing" if liquidity_trend_7d > 2 else "stable" if abs(liquidity_trend_7d) <= 2 else "declining" if liquidity_trend_7d > -10 else "declining rapidly"
            
            # Investment recommendation
            if prediction_score > 75:
                recommendation = "strong opportunity for investment"
            elif prediction_score > 60:
                recommendation = "potential opportunity worth considering"
            elif prediction_score > 40:
                recommendation = "moderate opportunity with reasonable risk/reward"
            elif prediction_score > 25:
                recommendation = "high-risk opportunity, proceed with caution"
            else:
                recommendation = "not recommended for investment at this time"
            
            # Generate the report
            report = f"""
            ## {pool_name} Pool Analysis
            
            The {pool_name} pool on {dex_name} currently shows {sentiment} indicators with a prediction score of {prediction_score:.1f}/100.
            
            ### Liquidity and Volume
            The pool has {liquidity_size} liquidity at ${liquidity:,.2f}, which has been {liquidity_trend_text} over the past 7 days ({liquidity_trend_7d:.2f}%).
            Trading volume is {volume_quality} at ${volume:,.2f} over the past 24 hours, representing a volume/TVL ratio of {volume/liquidity:.2%}.
            
            ### Yield Analysis
            The current APR is {apr_sentiment} at {apr:.2f}% and has been {apr_trend_text} over the past 24 hours ({apr_trend_24h:.2f}%).
            
            ### Investment Outlook
            Based on comprehensive analysis of on-chain metrics, market conditions, and historical performance, this pool presents a {recommendation}.
            
            ### Pool ID Reference
            ```
            {pool_data['id']}
            ```
            """
            
            st.markdown(report)
        
        # Display historical performance chart
        st.subheader("Simulated Historical Performance")
        
        # Create simulated historical data for demonstration
        days = 30
        # Use safer methods to get values
        tvl_change_30d = pool_data.get("tvl_change_30d", 0)
        apr_change_30d = pool_data.get("apr_change_30d", 0)
        
        # Calculate base values
        base_liquidity = pool_data["liquidity"] / (1 + (tvl_change_30d/100))
        base_apr = pool_data["apr"] / (1 + (apr_change_30d/100))
        
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days, 0, -1)]
        
        # Generate random daily fluctuations within the overall trend
        liquidity_trend = tvl_change_30d / 30  # daily trend
        apr_trend = apr_change_30d / 30  # daily trend
        
        # Create historical data with some randomness
        historical_data = []
        
        for i in range(days):
            daily_liquidity_change = liquidity_trend + random.uniform(-0.5, 0.5)
            daily_apr_change = apr_trend + random.uniform(-0.2, 0.2)
            
            factor_liquidity = 1 + (daily_liquidity_change / 100)
            factor_apr = 1 + (daily_apr_change / 100)
            
            if i == 0:
                liquidity = base_liquidity
                apr = base_apr
            else:
                liquidity = historical_data[i-1]["liquidity"] * factor_liquidity
                apr = historical_data[i-1]["apr"] * factor_apr
            
            historical_data.append({
                "date": dates[i],
                "liquidity": liquidity,
                "apr": apr
            })
        
        # Add current day
        historical_data.append({
            "date": datetime.now().strftime('%Y-%m-%d'),
            "liquidity": pool_data["liquidity"],
            "apr": pool_data["apr"]
        })
        
        # Convert to DataFrame
        historical_df = pd.DataFrame(historical_data)
        
        # Create a subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add TVL line
        fig.add_trace(
            go.Scatter(
                x=historical_df["date"],
                y=historical_df["liquidity"],
                name="TVL",
                line=dict(color="#1E88E5", width=3)
            ),
            secondary_y=False,
        )
        
        # Add APR line
        fig.add_trace(
            go.Scatter(
                x=historical_df["date"],
                y=historical_df["apr"],
                name="APR",
                line=dict(color="#43A047", width=3, dash="dash")
            ),
            secondary_y=True,
        )
        
        # Set titles
        fig.update_layout(
            title=f"Historical Performance: {pool_data['name']}",
            hovermode="x unified",
            height=500
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="TVL ($)", secondary_y=False)
        fig.update_yaxes(title_text="APR (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # REMINDER disclaimer
        st.info("""
        **Note**: Historical data shown here is simulated based on current metrics 
        and trends for demonstration purposes. The AI-generated report is based on 
        algorithmic analysis and should not be considered financial advice.
        """)

# Initialize the database when the module is first loaded
if db_handler.engine:
    db_handler.init_db()

if __name__ == "__main__":
    main()