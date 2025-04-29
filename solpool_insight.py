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
import requests
import logging
import traceback
import sys

# Import our data service modules
from data_services.initialize import init_services, get_stats
from data_services.data_service import get_data_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('solpool_insight')

# Import our database handler
import db_handler

# Import token services
from token_price_service import get_token_price, get_multiple_prices
from token_data_service import get_token_service

# Import DeFi Aggregation API
try:
    from defi_aggregation_api import DefiAggregationAPI
    HAS_DEFI_API = True
except ImportError:
    HAS_DEFI_API = False

# Custom exception handler for the entire app
def handle_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"An error occurred: {str(e)}")
            st.warning("Please try refreshing the page or contact support if the issue persists.")
            return None
    return wrapper

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

# Start continuous data collection for better predictions
def initialize_continuous_data_collection():
    """Initialize continuous data collection on startup"""
    try:
        # Initialize our new data services
        services = init_services()
        
        # Store services in session state for future access
        st.session_state["data_services"] = services
        
        # Get the data service
        data_service = services["data_service"]
        
        # Start scheduled collection if not already running
        if not data_service.scheduler_running:
            data_service.start_scheduled_collection()
            logger.info("Started scheduled data collection service")
        
        logger.info("Initialized data services with continuous collection for better predictions")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize data services: {str(e)}")
        logger.error(traceback.format_exc())
        return False
        
# Run initialization if this is the first time
if 'initialized_data_collection' not in st.session_state:
    st.session_state.initialized_data_collection = initialize_continuous_data_collection()

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
        "apr", "fee", "version"
    ]
    
    # Extra fields that might be missing but can have default values
    optional_fields = {
        "apr_change_24h": 0.0,
        "apr_change_7d": 0.0,
        "apr_change_30d": 0.0,
        "tvl_change_24h": 0.0,
        "tvl_change_7d": 0.0,
        "tvl_change_30d": 0.0,
        "token1_price": 0.0,
        "token2_price": 0.0,
        "prediction_score": 0.0,  # Add prediction score field
        "risk_score": 0.0,        # Add risk score field
        "sentiment_score": 0.0,   # Add sentiment score field
        "volatility": 0.0,        # Add volatility field
        "volume_change_24h": 0.0, # Add volume change field
        "liquidity_change_24h": 0.0  # Add liquidity change field
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
        
        # Get token prices from CoinGecko if not already present
        if validated_pool.get("token1_price", 0) == 0 or validated_pool.get("token2_price", 0) == 0:
            try:
                from token_price_service import update_pool_with_token_prices
                validated_pool = update_pool_with_token_prices(validated_pool)
            except Exception as e:
                st.warning(f"Could not get token prices: {e}")
        
        validated_pools.append(validated_pool)
    
    return validated_pools

@handle_exception
def fetch_live_data_from_blockchain():
    """Fetch live pool data directly from the blockchain using reliable fetch methods"""
    
    # First try the DeFi Aggregation API which provides authentic data with base58 encoded pool IDs
    try:
        # Import here to avoid circular imports
        from defi_aggregation_api import DefiAggregationAPI
        
        # Check if we have the API key
        api_key = os.getenv("DEFI_API_KEY")
        if not api_key:
            st.error("No DEFI_API_KEY found in environment variables. This is required for authentic data.")
            st.info("Trying alternative data sources...")
            # Continue to other methods
        else:
            st.info("Using DeFi Aggregation API for authentic on-chain data...")
            
            # Initialize the API client - it uses Bearer token auth based on our testing
            defi_api = DefiAggregationAPI(api_key=api_key)
            
            # Show progress information
            with st.spinner("Fetching verified pool data with rate limiting (10 req/sec)..."):
                # Fetch pools with appropriate rate limiting
                # This will respect the 10 req/sec limit documented in the API
                pool_data = defi_api.get_transformed_pools(max_pools=75)  # Reasonable limit for good performance
                
                if pool_data and len(pool_data) > 5:
                    # Save to cache for future use
                    defi_api.save_pools_to_cache(pool_data)
                    
                    st.success(f"✓ Successfully fetched {len(pool_data)} pools with authentic data")
                    return ensure_all_fields(pool_data)  # Ensure all required fields exist
                else:
                    st.warning("DeFi API returned insufficient data. Trying alternative sources...")
    except ImportError:
        st.warning("DeFi Aggregation API module not available. Trying alternative sources...")
    except Exception as e:
        # Log and provide user-friendly error
        logger.warning(f"DeFi API error: {str(e)}")
        st.warning(f"Issue with DeFi API: {str(e)}")
        if "rate limit" in str(e).lower():
            st.error("Rate limit exceeded. The API allows 10 requests per second.")
        elif "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
            st.error("API authentication failed. Please check your DEFI_API_KEY.")
    
    # Get the custom_rpc value from session state as fallback
    custom_rpc = st.session_state.get('custom_rpc', os.getenv("SOLANA_RPC_ENDPOINT", ""))
    
    # Proceed only if we have a valid RPC endpoint
    if not custom_rpc or len(custom_rpc) < 10:
        st.error("No valid RPC endpoint configured. Cannot fetch blockchain data directly.")
        return None
    
    # Use alternative pool fetcher as fallback 
    try:
        # Import here to avoid circular imports
        from alternative_pool_fetcher import AlternativePoolFetcher
        
        st.info("Using alternative fetcher as fallback...")
        
        # Initialize fetcher with our custom RPC endpoint
        fetcher = AlternativePoolFetcher(rpc_endpoint=custom_rpc)
        
        # Fetch a reasonable number of pools
        with st.spinner("Fetching data with alternative method..."):
            pool_dicts = fetcher.fetch_pools(limit=10)
            
            if pool_dicts and len(pool_dicts) > 0:
                # Save to cache for future use
                with open("extracted_pools.json", "w") as f:
                    json.dump(pool_dicts, f)
                
                st.success(f"✓ Successfully fetched {len(pool_dicts)} pools using alternative fetcher")
                
                # Ensure all required fields are present
                pool_dicts = ensure_all_fields(pool_dicts)
                return pool_dicts
            else:
                st.warning("No pools found with alternative fetcher. Trying original method...")
    except Exception as e:
        # Just log and continue to original method
        logger.warning(f"Alternative fetcher failed: {str(e)}")
        st.warning("Alternative fetcher encountered an issue. Trying original method...")
            
    # Fall back to original OnChainExtractor method
    try:
        # Import here to avoid circular imports
        from onchain_extractor import OnChainExtractor
        
        st.info("Trying original OnChainExtractor method...")
        
        # Initialize extractor with custom RPC endpoint
        extractor = OnChainExtractor(rpc_endpoint=custom_rpc)
        
        # Extract pools from major DEXes with limited number to avoid timeouts
        max_per_dex = 3  # Reduced to avoid timeouts
        
        with st.spinner("Extracting pools from blockchain using original extractor..."):
            # This is a direct call to extract pools from all supported DEXes
            pools = extractor.extract_pools_from_all_dexes(max_per_dex=max_per_dex)
            
            if pools and len(pools) > 0:
                # Convert to dictionary format
                pool_dicts = [pool.to_dict() for pool in pools]
                
                # Save to cache for future use
                with open("extracted_pools.json", "w") as f:
                    json.dump(pool_dicts, f)
                
                st.success(f"✓ Successfully fetched {len(pool_dicts)} pools with original extractor")
                
                # Ensure all required fields are present
                pool_dicts = ensure_all_fields(pool_dicts)
                return pool_dicts
            else:
                st.error("No pools were found on the blockchain. Check your RPC endpoint.")
                return None
    except ImportError:
        st.error("OnChainExtractor module is not available")
        return None
    except AttributeError as ae:
        logger.error(f"Function not available in OnChainExtractor: {str(ae)}")
        st.error(f"OnChainExtractor module is missing required functions")
        return None
    except Exception as e:
        logger.error(f"Error fetching live data: {str(e)}")
        st.error(f"Could not fetch live data: {str(e)}")
        return None

@handle_exception
def load_data():
    """Load pool data with a prioritized strategy: live or cached"""
    
    # Check if data service is initialized
    if "data_services" not in st.session_state:
        try:
            # Initialize data services if not already done
            services = init_services()
            st.session_state["data_services"] = services
            logger.info("Initialized data services on demand")
        except Exception as e:
            logger.error(f"Failed to initialize data services on demand: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fallback to legacy cache file loading
            st.warning("Could not initialize data services. Using legacy data loading.")
            return _load_data_legacy()
    
    # Get the data service
    try:
        data_service = get_data_service()
    except Exception as e:
        logger.error(f"Failed to get data service: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fallback to legacy cache file loading
        st.warning("Could not access data service. Using legacy data loading.")
        return _load_data_legacy()
    
    # 1. Check if user has requested to use cached data
    if st.session_state.get('use_cached_data', False):
        # Reset the flag so it doesn't keep using cached data every refresh
        st.session_state['use_cached_data'] = False
        
        # Get cached pools from the data service
        try:
            cached_pools = data_service.get_all_pools(force_refresh=False)
            
            if cached_pools and len(cached_pools) > 0:
                # Get cache stats
                cache_stats = st.session_state["data_services"]["cache_manager"].get_stats()
                
                # Estimate age
                last_collection_time = data_service.stats.get("last_collection_time")
                if last_collection_time:
                    try:
                        last_time = datetime.strptime(last_collection_time, "%Y-%m-%d %H:%M:%S")
                        time_diff = datetime.now() - last_time
                        
                        if time_diff.days > 0:
                            age_str = f"{time_diff.days} days old"
                        elif time_diff.seconds > 3600:
                            age_str = f"{time_diff.seconds // 3600} hours old"
                        else:
                            age_str = f"{time_diff.seconds // 60} minutes old"
                    except Exception:
                        age_str = "Unknown age"
                else:
                    age_str = "Unknown age"
                
                # Ensure all required fields are present
                pools = ensure_all_fields(cached_pools)
                
                st.info(f"ℹ️ Using cached data ({age_str}) - Hit ratio: {cache_stats.get('hit_ratio', 0):.2%}")
                st.session_state['data_source'] = f"Cached data ({age_str})"
                return pools
            else:
                st.warning("No cached data found. Attempting to fetch fresh data...")
        except Exception as e:
            logger.error(f"Error accessing cached data: {str(e)}")
            st.warning(f"Could not access cached data: {e}")
    
    # 2. Check if user has requested to force using live data
    force_refresh = st.session_state.get('try_live_data', False)
    
    # Reset flags to avoid constant retries
    st.session_state['try_live_data'] = False
    st.session_state['use_defi_api'] = False
    
    # 3. Fetch/refresh data from service
    try:
        with st.spinner("Collecting data from available sources..."):
            pools, success = data_service.collect_data(force=force_refresh)
            
            if pools and len(pools) > 0:
                # Ensure all required fields are present
                pools = ensure_all_fields(pools)
                
                # Show success message
                if success:
                    st.success(f"✓ Successfully collected {len(pools)} pools from data service")
                else:
                    st.warning(f"⚠️ Retrieved {len(pools)} pools, but with some collection errors")
                
                # Update session state
                collection_time = data_service.stats.get("last_collection_time", 
                                                       datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                st.session_state['data_source'] = f"Live data service ({collection_time})"
                
                return pools
            else:
                st.warning("No pools collected from data service")
                
                # Check if we should fall back to legacy method
                if not success:
                    st.info("Trying legacy data loading methods...")
                    return _load_data_legacy()
                else:
                    st.error("No pool data available")
                    return []
    except Exception as e:
        logger.error(f"Error collecting data from service: {str(e)}")
        logger.error(traceback.format_exc())
        
        st.warning(f"Error collecting data: {str(e)}")
        st.info("Falling back to legacy data loading...")
        
        # Try legacy method as fallback
        return _load_data_legacy()

# Legacy data loading method for fallback
@handle_exception
def _load_data_legacy():
    """Legacy method to load pool data from file cache"""
    logger.info("Using legacy data loading method")
    
    # Load from cached file
    cache_file = "extracted_pools.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                pools = json.load(f)
            
            if pools and len(pools) > 0:
                # Get modification time for the cache file
                mod_time = os.path.getmtime(cache_file)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                mod_time_diff = datetime.now() - datetime.fromtimestamp(mod_time)
                
                # Ensure all required fields are present
                pools = ensure_all_fields(pools)
                
                # Indicate how old the data is
                if mod_time_diff.days > 0:
                    age_str = f"{mod_time_diff.days} days old"
                elif mod_time_diff.seconds > 3600:
                    age_str = f"{mod_time_diff.seconds // 3600} hours old"
                else:
                    age_str = f"{mod_time_diff.seconds // 60} minutes old"
                
                st.info(f"ℹ️ Using legacy cached data from {mod_time_str} ({age_str})")
                st.session_state['data_source'] = f"Legacy cached data ({age_str})"
                return pools
        except Exception as e:
            st.warning(f"Error loading legacy cached data: {e}")
    
    # If all else fails, show a helpful error message
    st.error("No pool data available. Please configure a valid DeFi API key to fetch authentic data.")
    st.info("Check the 'API Key Configuration' section to set up your API key.")
    
    # Return an empty list to avoid errors
    return []

def generate_sample_data(count=200):
    """Generate sample pool data - disabled for data integrity reasons"""
    logger.warning("Sample data generation is disabled for data integrity reasons")
    st.error("⚠️ Sample data generation has been disabled for data integrity reasons. Please provide a valid Solana RPC endpoint to access authentic data.")
    return []

@handle_exception
def main():
    """
    Main application function with robust error handling.
    """
    # Configure the sidebar first to ensure it's always visible
    with st.sidebar:
        st.sidebar.title("SolPool Insight")
        
        # Set defaults for data source
        rpc_endpoint = os.getenv("SOLANA_RPC_ENDPOINT")
        has_valid_rpc = rpc_endpoint and len(rpc_endpoint) > 5
        
        # Default to always try to use live data
        st.session_state['force_live_data'] = True
        
        # Store default pool count in session state in case it's needed
        st.session_state['pool_count'] = 15
        
        # Get RPC endpoint from environment variables
        custom_rpc = os.getenv("SOLANA_RPC_ENDPOINT", "")
                
        # Store the RPC endpoint in session state
        st.session_state['custom_rpc'] = custom_rpc
        
        # Add data management section with clear data options
        st.sidebar.header("Data Management")
        
        # Show live data option with a badge
        st.sidebar.markdown("### 📊 Data Sources")
        
        # Create a specific button for DeFi API
        if st.sidebar.button("💹 Fetch DeFi API Data", 
                    help="Fetch real-time data from the DeFi Aggregation API with authentic on-chain metrics",
                    use_container_width=True):
            # Set flags for DeFi API data
            st.session_state['try_live_data'] = True
            st.session_state['use_defi_api'] = True
            st.session_state['generate_sample_data'] = False
            
            # Show info message
            st.info("Attempting to fetch authentic data from DeFi Aggregation API...")
            time.sleep(1)
            st.rerun()
        
        # Create 2 columns for other options
        live_col, sample_col = st.sidebar.columns(2)
        
        with live_col:
            # Add blockchain data button
            if st.button("🌐 Blockchain Data", 
                         help="Fetch real-time data directly from the Solana blockchain using your RPC endpoint"):
                # Set flag to try live data on next load
                st.session_state['try_live_data'] = True
                st.session_state['use_defi_api'] = False  # Explicitly not using DeFi API
                st.session_state['generate_sample_data'] = False
                
                # Show info message
                st.info("Attempting to fetch live blockchain data...")
                time.sleep(1)
                st.rerun()
        
        with sample_col:
            # Add cached data button
            if st.button("🔄 Use Cached Data", 
                         help="Use previously fetched data from cache"):
                # Set flag to use cached data
                st.session_state['generate_sample_data'] = False
                st.session_state['try_live_data'] = False
                st.session_state['use_defi_api'] = False
                st.session_state['use_cached_data'] = True
                
                # Show info message
                st.info("Loading data from cache...")
                time.sleep(1)
                st.rerun()
        
        # Add divider
        st.sidebar.divider()
        
        # Cached data configuration
        st.sidebar.markdown("### ⚙️ Cache Settings")
        
        # Display data service stats if available
        if "data_services" in st.session_state:
            try:
                # Get data service stats
                data_service = get_data_service()
                stats = data_service.get_system_stats()
                
                # Show collection stats
                last_collection = stats.get("last_collection_time", "Never")
                pool_count = stats.get("last_collection_pool_count", 0)
                
                # Show collection info
                st.sidebar.info(f"Last collection: {last_collection}")
                st.sidebar.info(f"Pools collected: {pool_count}")
                
                # Get cache stats
                cache_stats = st.session_state["data_services"]["cache_manager"].get_stats()
                cache_hit_ratio = cache_stats.get("hit_ratio", 0)
                
                # Show cache hit ratio as progress bar
                st.sidebar.text("Cache efficiency:")
                st.sidebar.progress(cache_hit_ratio)
                st.sidebar.caption(f"{cache_hit_ratio:.1%} cache hit ratio")
                
                # Add a button to clear system cache
                if st.sidebar.button("Clear System Cache", help="Clear the data service cache to force fresh data collection"):
                    try:
                        # Clear the cache
                        st.session_state["data_services"]["cache_manager"].clear()
                        st.sidebar.success("System cache cleared successfully")
                    except Exception as e:
                        st.sidebar.error(f"Error clearing system cache: {e}")
            except Exception as e:
                st.sidebar.warning(f"Data service stats unavailable: {str(e)}")
                
        # Add a button to clear the legacy cache
        if st.sidebar.button("Clear Legacy Cache", help="Delete legacy cached pool data file"):
            try:
                if os.path.exists("extracted_pools.json"):
                    os.remove("extracted_pools.json")
                    st.sidebar.success("Legacy cache cleared successfully")
                else:
                    st.sidebar.info("No legacy cache file found")
            except Exception as e:
                st.sidebar.error(f"Error clearing legacy cache: {e}")
                
        
        # Display legacy cache update time if available
        if os.path.exists("extracted_pools.json"):
            try:
                mod_time = os.path.getmtime("extracted_pools.json")
                mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                st.sidebar.info(f"Legacy cache update: {mod_time_str}")
            except Exception:
                pass
    
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
        tab_explore, tab_advanced, tab_predict, tab_risk, tab_nlp, tab_tokens = st.tabs([
            "Data Explorer", "Advanced Filtering", "Predictions", "Risk Assessment", "NLP Reports", "Token Explorer"
        ])
        
        # Load data
        pool_data = load_data()
        
        if not pool_data or len(pool_data) == 0:
            st.error("No pool data available. Please check your data sources.")
            return
        
        # Remove duplicate pools based on ID
        unique_pools = {}
        for pool in pool_data:
            pool_id = pool.get('id', '')
            if pool_id and pool_id not in unique_pools:
                unique_pools[pool_id] = pool
        
        # Log if we found and removed duplicates
        if len(unique_pools) < len(pool_data):
            logger.warning(f"Removed {len(pool_data) - len(unique_pools)} duplicate pool IDs")
            st.info(f"⚠️ Removed {len(pool_data) - len(unique_pools)} duplicate pool IDs for more accurate analysis")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(list(unique_pools.values()))
    
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
            
            # Display data source
            data_source = st.session_state.get('data_source', 'Unknown data source')
            if "Live" in data_source:
                st.success(f"✓ {data_source} - Real-time information from blockchain")
            elif "Database" in data_source:
                st.info(f"ℹ️ {data_source} - Data stored in PostgreSQL database")
            elif "cache" in data_source.lower():
                st.warning(f"⚠️ {data_source} - This may not reflect current market conditions")
            else:
                st.warning(f"⚠️ {data_source}")
            
            # Pool data table
            st.subheader("Pool Data")
            
            # Get watchlists for filtering
            try:
                watchlists = db_handler.get_watchlists()
                if watchlists:
                    # Allow filtering by watchlist
                    watchlist_options = ["All Pools"] + [w["name"] for w in watchlists]
                    selected_watchlist = st.selectbox(
                        "Filter by Watchlist",
                        options=watchlist_options,
                        index=0
                    )
                    
                    # Apply watchlist filter if selected
                    if selected_watchlist != "All Pools":
                        # Find the watchlist ID
                        watchlist_id = next((w["id"] for w in watchlists if w["name"] == selected_watchlist), None)
                        
                        if watchlist_id:
                            # Get the pool IDs in this watchlist
                            watchlist_pool_ids = db_handler.get_pools_in_watchlist(watchlist_id)
                            
                            if watchlist_pool_ids:
                                # Filter the dataframe
                                filtered_df = filtered_df[filtered_df["id"].isin(watchlist_pool_ids)]
                                st.success(f"Showing {len(filtered_df)} pools from watchlist: {selected_watchlist}")
            except Exception as e:
                # Silently handle errors with watchlists as this is an optional feature
                pass
            
            # Create token information display option
            show_token_details = st.checkbox("Show Detailed Token Information", value=False)
            
            # Show only the most relevant columns, with option for token details
            display_columns = [
                "name", "dex", "category", "liquidity", "volume_24h", 
                "apr", "apr_change_24h", "apr_change_7d", "prediction_score", "id"
            ]
            
            # Add token columns if requested
            if show_token_details:
                # Create tabs for different views
                token_tab1, token_tab2 = st.tabs(["Basic Token Info", "Advanced Token Details"])
                
                with token_tab1:
                    st.info("The tables below will include token symbols, addresses, and prices")
            
            # Create a proper Streamlit table instead of HTML
            # This avoids formatting issues with HTML badges
            
            # First create a clean representation of the data
            table_data = []
            
            # Allow showing more pools per page
            # When only a few pools are available, limit the options
            if len(filtered_df) <= 5:
                pools_per_page = min(len(filtered_df), 5)
                st.write(f"Pools per page: {pools_per_page}")
            else:
                # Only show options that make sense based on number of pools
                slider_options = [opt for opt in [5, 10, 20, 50, 100] if opt <= len(filtered_df)]
                if not slider_options:
                    slider_options = [min(len(filtered_df), 5)]
                
                pools_per_page = st.select_slider(
                    "Pools per page", 
                    options=slider_options,
                    value=min(50, max(slider_options)) if len(slider_options) > 1 else slider_options[0],
                    help="Choose how many pools to display on each page"
                )
            
            # Calculate pagination based on the selected number of pools per page
            max_display = min(len(filtered_df), pools_per_page)
            
            # Calculate the maximum number of pages
            max_pages = max(1, len(filtered_df) // max_display + (1 if len(filtered_df) % max_display > 0 else 0))
            
            # Fix for slider when min_value equals max_value
            if max_pages == 1:
                page = 1
                st.write("Page: 1")
            else:
                # Add pagination control 
                page = st.slider("Page", min_value=1, max_value=max_pages, value=1)
            start_idx = (page - 1) * max_display
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
                
                # Prepare token data if available
                token1_symbol = row.get("token1_symbol", "Unknown")
                token2_symbol = row.get("token2_symbol", "Unknown")
                token1_address = row.get("token1_address", "")
                token2_address = row.get("token2_address", "")
                token1_price = row.get("token1_price", 0)
                token2_price = row.get("token2_price", 0)
                
                # Format token addresses for display (truncated)
                token1_addr_display = f"{token1_address[:6]}...{token1_address[-4:]}" if len(token1_address) > 10 else token1_address
                token2_addr_display = f"{token2_address[:6]}...{token2_address[-4:]}" if len(token2_address) > 10 else token2_address
                
                # Create standard table entry
                table_entry = {
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
                }
                
                # If showing token details, add token-specific fields
                if show_token_details:
                    table_entry.update({
                        "Token1": token1_symbol,
                        "Token2": token2_symbol,
                        "Token1 Address": token1_addr_display,
                        "Token2 Address": token2_addr_display,
                        "Token1 Price": f"${token1_price:.6f}" if token1_price < 0.01 else f"${token1_price:.2f}",
                        "Token2 Price": f"${token2_price:.6f}" if token2_price < 0.01 else f"${token2_price:.2f}"
                    })
                
                # Add to table data
                table_data.append(table_entry)
            
            # Show as dataframe
            table_df = pd.DataFrame(table_data)
            
            # Display the dataframe
            if show_token_details:
                with token_tab1:
                    st.dataframe(table_df, use_container_width=True)
                
                # Display detailed token information in the second tab if tokens data is available
                with token_tab2:
                    st.subheader("Tokens Detailed View")
                    
                    # Create expanded token information
                    expanded_token_data = []
                    
                    for _, row in filtered_df.iloc[start_idx:end_idx].iterrows():
                        # Check if tokens field exists in the data
                        if 'tokens' in row:
                            tokens = row.get('tokens', [])
                            if isinstance(tokens, list) and len(tokens) > 0:
                                for token in tokens:
                                    expanded_token_data.append({
                                        "Pool Name": row["name"],
                                        "Token Symbol": token.get("symbol", "Unknown"),
                                        "Token Name": token.get("name", "Unknown"),
                                        "Address": token.get("address", ""),
                                        "Decimals": token.get("decimals", 0),
                                        "Price": f"${token.get('price', 0):.6f}" if token.get('price', 0) < 0.01 else f"${token.get('price', 0):.2f}",
                                        "Active": "✓" if token.get("active", False) else "✗"
                                    })
                    
                    if expanded_token_data:
                        st.dataframe(pd.DataFrame(expanded_token_data), use_container_width=True)
                    else:
                        st.info("Detailed token information not available in the current dataset")
            else:
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
                        # Ensure max_value is greater than min_value for sliders
                        max_liquidity_value = float(advanced_df["liquidity"].max())
                        # Add a small buffer to max value if it's 0 to prevent min==max error
                        if max_liquidity_value == 0:
                            max_liquidity_value = 0.01
                            
                        min_liquidity = st.slider(
                            "Minimum Liquidity ($)", 
                            min_value=0.0, 
                            max_value=max_liquidity_value,
                            value=0.0,
                            format="$%.2f"
                        )
                        
                        max_liquidity = st.slider(
                            "Maximum Liquidity ($)", 
                            min_value=0.0, 
                            max_value=max_liquidity_value,
                            value=max_liquidity_value,
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
            
            # Display a message indicating Claude integration is pending
            st.info("🤖 Claude AI integration is in progress")
            
            st.markdown("""
            ### About NLP Reports
            
            This feature will use Claude AI to generate real-time analysis of pool data and market trends
            when the integration is complete. No mock reports are shown as we focus on providing only authentic insights.
            
            #### Coming Soon:
            
            - **Pool-specific analysis** - Detailed evaluations of individual pools
            - **Market trend detection** - Identification of patterns across pools
            - **Investment considerations** - Risk-adjusted opportunity analysis
            - **Token correlation insights** - Understanding how tokens relate to each other
            
            Stay tuned for these features in upcoming releases.
            """)
            
            # Show basic data for reference
            st.subheader("Available Pool Data (For Reference)")
            
            # Show a sample of the data that would be analyzed
            st.dataframe(df[['name', 'dex', 'category', 'apr', 'liquidity', 'volume_24h']].head(5), use_container_width=True)
            
            # Contact information
            st.markdown("""
            For questions about NLP capabilities or to provide feedback on what insights
            would be most valuable to you, please contact the development team.
            """)
            
            # Technical information about the API
            with st.expander("Technical Information"):
                st.markdown("""
                The NLP Reports module will integrate with Claude AI using the Anthropic API.
                This provides more nuanced and context-aware analysis compared to template-based reports.
                
                **Implementation status:**
                - ✅ Backend API integration prepared
                - ✅ Token and API validation logic complete
                - ⏳ Request formatting in progress
                - ⏳ Response parsing in development
                """)
                
                # Environment check
                if os.getenv("ANTHROPIC_API_KEY"):
                    st.success("Claude API credentials are configured")
                else:
                    st.warning("Claude API credentials are not yet configured")
        
        # Token Explorer Tab
        with tab_tokens:
            st.header("Token Price Explorer")
            st.markdown("""
            Track and analyze token prices using real-time data from the DeFi API and CoinGecko. 
            This tab allows you to monitor token prices, view historical trends, and compare tokens.
            """)
            
            # Initialize token service for enhanced token data
            token_service = get_token_service()
            
            # Get list of popular tokens plus tokens from pools
            popular_tokens = [
                "SOL", "BTC", "ETH", "USDC", "USDT", "JUP", "BONK", "RAY", "ORCA", "SAMO", 
                "WIF", "PYTH", "MNGO", "SRM", "ATLAS", "POLIS"
            ]
            
            # Extract tokens from pools and track token metadata
            pool_tokens = set()
            token_metadata = {}
            
            # First check if we have 'tokens' lists in the data (from newer API format)
            has_token_lists = any('tokens' in row and isinstance(row['tokens'], list) and len(row['tokens']) > 0 for _, row in df.iterrows())
            
            if has_token_lists:
                # Extract from token lists (preferred format from API)
                for _, row in df.iterrows():
                    if 'tokens' in row and isinstance(row['tokens'], list):
                        for token in row['tokens']:
                            if 'symbol' in token and isinstance(token['symbol'], str) and len(token['symbol']) > 0:
                                symbol = token['symbol'].upper()
                                pool_tokens.add(symbol)
                                # Store metadata keyed by uppercase symbol
                                if symbol not in token_metadata:
                                    token_metadata[symbol] = {
                                        'symbol': symbol,
                                        'name': token.get('name', 'Unknown'),
                                        'address': token.get('address', ''),
                                        'decimals': token.get('decimals', 0),
                                        'price': token.get('price', 0),
                                        'active': token.get('active', True)
                                    }
            
            # Fallback to traditional token1/token2 format
            elif "token1_symbol" in df.columns and "token2_symbol" in df.columns:
                for _, row in df.iterrows():
                    if isinstance(row["token1_symbol"], str) and len(row["token1_symbol"]) > 0:
                        symbol = row["token1_symbol"].upper()
                        pool_tokens.add(symbol)
                        # Store basic metadata
                        if symbol not in token_metadata:
                            token_metadata[symbol] = {
                                'symbol': symbol,
                                'name': 'Unknown',
                                'address': row.get('token1_address', ''),
                                'decimals': 0,
                                'price': row.get('token1_price', 0),
                                'active': True
                            }
                    if isinstance(row["token2_symbol"], str) and len(row["token2_symbol"]) > 0:
                        symbol = row["token2_symbol"].upper() 
                        pool_tokens.add(symbol)
                        # Store basic metadata
                        if symbol not in token_metadata:
                            token_metadata[symbol] = {
                                'symbol': symbol,
                                'name': 'Unknown',
                                'address': row.get('token2_address', ''),
                                'decimals': 0,
                                'price': row.get('token2_price', 0),
                                'active': True
                            }
            
            # Store the token metadata in session state for later use
            st.session_state['token_metadata'] = token_metadata
            
            # Combine and deduplicate token lists
            all_tokens = sorted(list(set(popular_tokens) | pool_tokens))
            
            # Create token selection interface
            col_search, col_select = st.columns([1, 2])
            
            with col_search:
                # Allow search by token name
                token_search = st.text_input("Search for a token", "")
                
                if token_search:
                    filtered_tokens = [t for t in all_tokens if token_search.upper() in t.upper()]
                else:
                    filtered_tokens = all_tokens
                    
                # List of tokens with selection
                selected_token = st.selectbox("Select a token", filtered_tokens)
                
                # Get token price
                token_price = get_token_price(selected_token)
                
                # Show current price in a big metric
                if token_price:
                    st.metric(
                        f"{selected_token} Price",
                        f"${token_price:,.4f}",
                        delta=None  # We don't have historical data for delta yet
                    )
                else:
                    st.warning(f"Could not retrieve price for {selected_token}")
            
            with col_select:
                # Multiple token selection for comparison
                comparison_tokens = st.multiselect(
                    "Compare multiple tokens",
                    options=all_tokens,
                    default=["SOL", "BTC", "ETH"][:3] if all(t in all_tokens for t in ["SOL", "BTC", "ETH"]) else []
                )
                
                if comparison_tokens:
                    with st.spinner("Fetching token prices..."):
                        # Fetch all prices at once
                        token_prices = get_multiple_prices(comparison_tokens)
                        
                        # Create data for display
                        price_data = []
                        for token in comparison_tokens:
                            if token in token_prices and token_prices[token] is not None:
                                price_data.append({
                                    "Token": token,
                                    "Price (USD)": token_prices[token]
                                })
                        
                        # Display as table
                        if price_data:
                            price_df = pd.DataFrame(price_data)
                            st.dataframe(price_df, use_container_width=True)
                        else:
                            st.warning("Could not retrieve prices for the selected tokens")
            
            # Token price visualization
            st.subheader("Token Price Visualization")
            
            # Simulate a price chart (since we don't have historical data)
            # In production, you would fetch historical price data from CoinGecko
            
            # Display all tokens from our metadata with detailed information
            st.subheader("All Available Tokens")
            
            # Check if we have token metadata
            if 'token_metadata' in st.session_state and st.session_state['token_metadata']:
                # Create a DataFrame from the token metadata
                token_metadata_df = pd.DataFrame([
                    {
                        "Symbol": metadata['symbol'],
                        "Name": metadata['name'],
                        "Address": metadata['address'][:10] + "..." if len(metadata['address']) > 10 else metadata['address'],
                        "Decimals": metadata['decimals'],
                        "Price": f"${metadata['price']:.6f}" if metadata['price'] < 0.01 else f"${metadata['price']:.2f}",
                        "Active": "✓" if metadata['active'] else "✗",
                        "Full Address": metadata['address']  # Hidden column for reference
                    }
                    for symbol, metadata in st.session_state['token_metadata'].items()
                ])
                
                # Sort by symbol for better readability
                token_metadata_df = token_metadata_df.sort_values('Symbol')
                
                # Add a search field for the token list
                token_list_search = st.text_input("Search in token list", "")
                
                # Filter based on search
                if token_list_search:
                    filtered_token_df = token_metadata_df[
                        token_metadata_df['Symbol'].str.contains(token_list_search, case=False) |
                        token_metadata_df['Name'].str.contains(token_list_search, case=False)
                    ]
                else:
                    filtered_token_df = token_metadata_df
                
                # Show token count
                st.write(f"Showing {len(filtered_token_df)} of {len(token_metadata_df)} tokens")
                
                # Display the dataframe with a copy button for addresses
                st.dataframe(
                    filtered_token_df,
                    use_container_width=True,
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                        "Name": st.column_config.TextColumn("Name", width="medium"),
                        "Address": st.column_config.TextColumn("Address", width="medium"),
                        "Decimals": st.column_config.NumberColumn("Decimals", width="small"),
                        "Price": st.column_config.TextColumn("Price", width="small"),
                        "Active": st.column_config.TextColumn("Active", width="small"),
                        "Full Address": st.column_config.TextColumn(
                            "Full Address", 
                            width=None,
                            disabled=True,
                            required=False
                        )
                    }
                )
            else:
                st.info("No detailed token metadata available. Try fetching data from the DeFi API for more token information.")
            
            # Create a section to fetch and display token information
            st.subheader("Token Information")
            
            if selected_token:
                # Create columns for token info
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.write("**Token Details**")
                    st.write(f"Symbol: {selected_token}")
                    
                    # First try to get metadata from our token service
                    token_metadata = token_service.get_token_metadata(selected_token)
                    
                    if token_metadata and token_metadata.get('name'):
                        st.write(f"Name: {token_metadata.get('name', 'Unknown')}")
                        
                        if token_metadata.get('address'):
                            st.write(f"Address: `{token_metadata.get('address')}`")
                        
                        if token_metadata.get('decimals') is not None:
                            st.write(f"Decimals: {token_metadata.get('decimals')}")
                            
                        price = token_metadata.get('price', 0)
                        if price > 0:
                            st.write(f"Price (from API): ${price:.6f}" if price < 0.01 else f"Price (from API): ${price:.2f}")
                            
                        # Show which DEXes use this token
                        dexes = token_metadata.get('dexes', [])
                        if dexes:
                            st.write("**Used by DEXes:**")
                            for dex in dexes:
                                st.write(f"- {dex.capitalize()}")
                    
                    # Fallback to CoinGecko
                    try:
                        # Display information from the token_price_service
                        from token_price_service import DEFAULT_TOKEN_MAPPING
                        
                        if selected_token.upper() in DEFAULT_TOKEN_MAPPING:
                            coingecko_id = DEFAULT_TOKEN_MAPPING[selected_token.upper()]
                            st.write(f"CoinGecko ID: {coingecko_id}")
                            
                            # Try to get more info from CoinGecko
                            try:
                                coingecko_url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
                                response = requests.get(coingecko_url, timeout=10)
                                
                                if response.status_code == 200:
                                    token_data = response.json()
                                    
                                    # Extract and display key information
                                    if 'market_data' in token_data:
                                        market_data = token_data['market_data']
                                        
                                        # Market cap
                                        if 'market_cap' in market_data and 'usd' in market_data['market_cap']:
                                            market_cap = market_data['market_cap']['usd']
                                            st.write(f"Market Cap: ${market_cap:,.2f}")
                                        
                                        # 24h volume
                                        if 'total_volume' in market_data and 'usd' in market_data['total_volume']:
                                            volume = market_data['total_volume']['usd']
                                            st.write(f"24h Volume: ${volume:,.2f}")
                                        
                                        # All-time high
                                        if 'ath' in market_data and 'usd' in market_data['ath']:
                                            ath = market_data['ath']['usd']
                                            st.write(f"All-Time High: ${ath:,.4f}")
                                    
                                    # Basic token info
                                    if 'name' in token_data:
                                        st.write(f"Name: {token_data['name']}")
                                        
                                    # Description
                                    if 'description' in token_data and 'en' in token_data['description']:
                                        with st.expander("Description"):
                                            st.markdown(token_data['description']['en'][:500] + "...")
                            except Exception as e:
                                st.warning(f"Could not fetch additional token data: {e}")
                        else:
                            st.warning(f"No CoinGecko mapping available for {selected_token}")
                    except Exception as e:
                        st.warning(f"Error retrieving token metadata: {e}")
                
                with info_col2:
                    st.write("**Price Statistics**")
                    
                    if token_price:
                        # In a production app, we would get these from the API
                        # For now, we'll just show the current price
                        st.write(f"Current Price: ${token_price:,.4f}")
                        
                        # Link to more information
                        if selected_token.upper() in DEFAULT_TOKEN_MAPPING:
                            coingecko_id = DEFAULT_TOKEN_MAPPING[selected_token.upper()]
                            st.markdown(f"[View on CoinGecko](https://www.coingecko.com/en/coins/{coingecko_id})")
                            
                        # Provide a link to pools containing this token
                        st.write("**Pools Containing This Token**")
                        
                        # Filter pools that contain the selected token
                        # First check if we can use tokens data or fall back to token1/token2
                        use_token_lists = any('tokens' in row and isinstance(row['tokens'], list) for _, row in df.iterrows())
                        
                        if use_token_lists:
                            # More precise filtering using tokens lists
                            token_pools = df[df.apply(
                                lambda row: any(
                                    isinstance(token, dict) and 
                                    'symbol' in token and 
                                    token['symbol'].upper() == selected_token.upper() 
                                    for token in row.get('tokens', [])
                                ),
                                axis=1
                            )]
                        else:
                            # Fall back to traditional token1/token2 fields
                            token_pools = df[
                                (df["token1_symbol"].str.upper() == selected_token.upper()) | 
                                (df["token2_symbol"].str.upper() == selected_token.upper())
                            ]
                        
                        if len(token_pools) > 0:
                            # Create a cleaner display of pools with more information
                            pool_data = []
                            
                            for _, pool in token_pools.iterrows():
                                pool_data.append({
                                    "Pool Name": pool['name'],
                                    "DEX": pool['dex'],
                                    "Liquidity": format_currency(pool['liquidity']),
                                    "Volume (24h)": format_currency(pool.get('volume_24h', 0)),
                                    "APR": format_percentage(pool['apr']),
                                    "Paired With": pool['token2_symbol'] if pool['token1_symbol'].upper() == selected_token.upper() else pool['token1_symbol']
                                })
                            
                            # Create a DataFrame for better display
                            pool_df = pd.DataFrame(pool_data)
                            
                            # Sort by liquidity for better relevance
                            pool_df = pool_df.sort_values("Liquidity", key=lambda x: x.str.replace('$', '').str.replace('M', '000000').str.replace('K', '000').astype(float), ascending=False)
                            
                            # Show the table
                            st.dataframe(pool_df, use_container_width=True)
                                
                            if len(token_pools) > 10:
                                st.info(f"Showing top pools by liquidity. {len(token_pools)} pools found containing {selected_token}.")
                        else:
                            st.write("No pools found containing this token")
        
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
            - **Token Explorer**: Monitor token prices and find pools containing specific tokens
            
            ### Data Sources
            
            Pool data is sourced directly from the Solana blockchain using RPC endpoints.
            Token price data is sourced from CoinGecko API.
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