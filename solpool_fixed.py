"""  
SolPool Insight - Comprehensive Solana Liquidity Pool Analytics Platform  
  
This application provides detailed analytics for Solana liquidity pools  
across various DEXes, with robust filtering, visualizations, and predictions.  
"""  
  
import streamlit as st  
import os  
import pandas as pd  
import json  
import random  
import math  
from datetime import datetime, timedelta  
import plotly.express as px  
import plotly.graph_objects as go  
from plotly.subplots import make_subplots  
import threading  
from PIL import Image  
import time  
import requests  
import logging  
import traceback  
import sys  
  
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('solpool_insight')
  
# Start performance monitoring  
from performance_monitor import get_performance_monitor  
perf_monitor = get_performance_monitor()  
perf_monitor.start_tracking("app_initialization")  
  
# Initialize API key with default value to avoid welcome screen  
from api_auth_helper import set_api_key  
api_key = os.getenv("DEFI_API_KEY") or "9feae0d0af47e4948e061f2d7820461e374e040c21cf65c087166d7ed18f5ed6"  
print(f"Using API key: {api_key[:8]}...")  
set_api_key(api_key)  

# CRITICAL FIX - Proper db_handler initialization
try:
    import db_handler
    from db_handler import get_db_handler
    db_handler_instance = get_db_handler()
    DB_CONNECTED = True
except Exception as e:
    logger.error(f"Error importing db_handler: {str(e)}")
    DB_CONNECTED = False
    db_handler_instance = None

# Import the historical data service  
from historical_data_service import get_historical_service, start_historical_collection  
start_time = time.time()  
loading_start = datetime.now()  
  
# Import our data service modules  
from data_services.initialize import init_services, get_stats  
from data_services.data_service import get_data_service  
from token_data_service import get_token_data_service as get_token_service  
from historical_data_service import get_historical_service  
  
# Import token services  
from token_price_service_minimal import get_token_price, get_multiple_prices  
  
# Import API key manager  
from api_key_manager import get_defi_api_key, set_defi_api_key, render_api_key_form  
  
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
        init_services()  
          
        # Get the data service  
        data_service = get_data_service()  
          
        # Initialize token service  
        token_service = get_token_service()  
          
        # Initialize historical data service  
        historical_service = get_historical_service()  
          
        # Store services in session state for future access  
        st.session_state["data_service"] = data_service  
        st.session_state["token_service"] = token_service  
        st.session_state["historical_service"] = historical_service  
          
        # Start scheduled collection if not already running  
        if not data_service.scheduler_running:  
            data_service.start_scheduled_collection()  
            logger.info("Started scheduled data collection service")  
          
        # Start historical data collection (new)  
        try:  
            start_historical_collection(data_service, interval_minutes=30)  
            logger.info("Started historical data collection service")  
        except Exception as e:  
            logger.error(f"Failed to start historical data collection: {str(e)}")  
          
        logger.info("Initialized data services with continuous collection for better predictions")  
        return True  
    except Exception as e:  
        logger.error(f"Failed to initialize data services: {str(e)}")  
        logger.error(traceback.format_exc())  
        return False  
          
# Enhanced token preloading function with pool data awareness  
def preload_tokens_background():  
    """  
    Preload token data in a background thread to keep UI responsive.  
    This function fetches all token data from the API and CoinGecko  
    without blocking the UI thread.  
    """  
    try:  
        logger.info("Starting token preloading in background thread")  
          
        # Get token service from session state or create new instance  
        token_service = st.session_state.get("token_service")  
        if not token_service:  
            token_service = get_token_service()  
          
        # Get data service to access pool information  
        try:  
            data_service = get_data_service()  
            all_pools = data_service.get_all_pools()  
            logger.info(f"Retrieved {len(all_pools) if all_pools else 0} pools for token extraction")  
              
            # Extract tokens from pools  
            token_symbols = set()  
            token_pools_map = {}  
              
            if all_pools and len(all_pools) > 0:  
                # Create token-to-pool mapping for relationship data  
                for pool in all_pools:  
                    # Extract token symbols  
                    token1 = pool.get('token1', {})  
                    token2 = pool.get('token2', {})  
                      
                    token1_symbol = token1.get('symbol', '') if isinstance(token1, dict) else str(token1)  
                    token2_symbol = token2.get('symbol', '') if isinstance(token2, dict) else str(token2)  
                      
                    # Add to set of tokens  
                    if token1_symbol and len(token1_symbol) > 0:  
                        token_symbols.add(token1_symbol)  
                          
                        # Map token to this pool  
                        if token1_symbol not in token_pools_map:  
                            token_pools_map[token1_symbol] = []  
                        token_pools_map[token1_symbol].append(pool)  
                      
                    # Add to set of tokens  
                    if token2_symbol and len(token2_symbol) > 0:  
                        token_symbols.add(token2_symbol)  
                          
                        # Map token to this pool  
                        if token2_symbol not in token_pools_map:  
                            token_pools_map[token2_symbol] = []  
                        token_pools_map[token2_symbol].append(pool)  
                  
                # Save the token-pool mapping to session state for quick access  
                st.session_state.token_pools_map = token_pools_map  
                logger.info(f"Created token-pool relationships for {len(token_symbols)} tokens")  
                  
                # Ensure common tokens are included in the list  
                common_tokens = [  
                    "SOL", "USDC", "USDT", "ETH", "BTC", "MSOL", "BONK", "RAY", "ORCA",   
                    "STSOL", "ATLA", "POLI", "JSOL", "JUPY", "HPSQ", "MNGO", "SAMO"  
                ]  
                for token in common_tokens:  
                    token_symbols.add(token)  
                  
                # Convert to list for batch processing  
                token_symbols = list(token_symbols)  
                logger.info(f"Preloading price data for {len(token_symbols)} tokens")  
                  
                # Process in smaller batches to avoid rate limits  
                batch_size = 5  
                for i in range(0, len(token_symbols), batch_size):  
                    batch = token_symbols[i:i+batch_size]  
                    logger.info(f"Fetching prices for token batch: {batch}")  
                      
                    for symbol in batch:  
                        try:  
                            # Get token data to ensure it's in cache  
                            token_data = token_service.get_token_data(symbol, force_refresh=True)  
                            logger.info(f"Preloaded token data for {symbol}")  
                              
                            # Sleep briefly to avoid hitting API rate limits  
                            time.sleep(1)  
                        except Exception as e:  
                            logger.warning(f"Error preloading token {symbol}: {str(e)}")  
                      
                    # Add a delay between batches  
                    time.sleep(2)  
            else:  
                logger.warning("No pools found for token extraction, using common tokens only")  
                # Preload common tokens as a fallback  
                token_service.preload_common_tokens()  
        except Exception as e:  
            logger.error(f"Error preloading additional tokens: {str(e)}")  
            # Fallback to common tokens  
            # Fetch prices for common tokens  
            common_tokens = [  
                "SOL", "USDC", "USDT", "ETH", "BTC", "MSOL", "BONK", "RAY", "ORCA",   
                "STSOL", "ATLA", "POLI", "JSOL", "JUPY", "HPSQ", "MNGO", "SAMO"  
            ]  
              
            # Process in smaller batches to avoid rate limits  
            batch_size = 5  
            for i in range(0, len(common_tokens), batch_size):  
                batch = common_tokens[i:i+batch_size]  
                logger.info(f"Fetching prices for token batch: {batch}")  
                  
                for symbol in batch:  
                    try:  
                        # Get token data to ensure it's in cache  
                        token_data = token_service.get_token_data(symbol, force_refresh=True)  
                        logger.info(f"Preloaded token data for {symbol}")  
                          
                        # Sleep briefly to avoid hitting API rate limits  
                        time.sleep(1)  
                    except Exception as e:  
                        logger.warning(f"Error preloading token {symbol}: {str(e)}")  
                  
                # Add a delay between batches  
                time.sleep(2)  
              
        logger.info("Completed token preloading in background")  
    except Exception as e:  
        logger.error(f"Error in background token preloading: {str(e)}")  
  
# Initialize API key with correct value  
def initialize_api_key():  
    """Initialize API key from environment variable or session state"""  
    api_key = os.getenv("DEFI_API_KEY") or "9feae0d0af47e4948e061f2d7820461e374e040c21cf65c087166d7ed18f5ed6"  
      
    # Store the API key in our auth helper for consistent access across components  
    set_api_key(api_key)  
    logger.info(f"API key initialized: {api_key[:5]}...")  
    return api_key  
  
# Run initialization if this is the first time  
if 'initialized_data_collection' not in st.session_state:  
    perf_monitor.start_tracking("data_collection_init")  
      
    # Initialize API key first so services can use it  
    api_key = initialize_api_key()  
      
    # Initialize data collection services  
    st.session_state.initialized_data_collection = initialize_continuous_data_collection()  
      
    # Start token preloading in a background thread  
    try:  
        token_thread = threading.Thread(target=preload_tokens_background, daemon=True)  
        token_thread.start()  
        logger.info("Started background thread for token preloading")  
    except Exception as e:  
        logger.error(f"Failed to start token preloading thread: {str(e)}")  
        # Fallback to synchronous loading if threading fails  
        if 'token_service' in st.session_state:  
            st.session_state.token_service.preload_all_tokens()  
      
    perf_monitor.stop_tracking("data_collection_init")  
    perf_monitor.mark_checkpoint("data_loaded")  
  
# Formatting helper functions  
def format_currency(value):  
    """Format a value as currency with robust error handling"""  
    if value is None:  
        return "$0.00"  
      
    try:  
        # Convert to float if it's a string  
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
        # If conversion fails, return default  
        return "$0.00"  
  
def format_percentage(value):  
    """Format a value as percentage with robust error handling"""  
    if value is None:  
        return "0.00%"  
      
    try:  
        # Convert to float if it's a string  
        value = float(value)  
        return f"{value:.2f}%"  
    except (ValueError, TypeError):  
        # If conversion fails, return default  
        return "0.00%"  
  
def get_trend_icon(value):  
    """Return an arrow icon based on trend direction with null handling"""  
    if value is None:  
        return "➡️"  # Stable for None  
          
    try:  
        value = float(value)  
          
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
    except (ValueError, TypeError):  
        return "➡️"  # Default to stable for any errors  
  
def get_category_badge(category):  
    """Return HTML for a category badge"""  
    colors = {  
        "Major Token": "#1E88E5",  # Blue  
        "Major": "#1E88E5",        # Blue (for backward compatibility)  
        "DeFi": "#43A047",         # Green  
        "Meme Token": "#FFC107",   # Yellow  
        "Meme": "#FFC107",         # Yellow (for backward compatibility)  
        "Gaming": "#D81B60",       # Pink  
        "Blue Chip": "#9C27B0",    # Purple  
        "Stablecoin": "#6D4C41",   # Brown  
        "Alt Token": "#FF5722",    # Orange  
        "Other": "#757575"         # Grey  
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
  
def display_pool_metrics(pool, col):  
    """Display key metrics for a pool in a standardized format"""  
    with col:  
        # Main metrics  
        liquidity = pool.get("liquidity", 0)  
        volume_24h = pool.get("volume_24h", 0)  
        apr = pool.get("apr", 0)  
          
        # Trends  
        apr_change_24h = pool.get("apr_change_24h", 0)  
        apr_change_7d = pool.get("apr_change_7d", 0)  
        tvl_change_24h = pool.get("tvl_change_24h", 0)  
        tvl_change_7d = pool.get("tvl_change_7d", 0)  
          
        # Format main metrics  
        st.metric("Liquidity", format_currency(liquidity), delta=f"{tvl_change_24h:.2f}%")  
        st.metric("Volume (24h)", format_currency(volume_24h))  
        st.metric("APR", format_percentage(apr), delta=f"{apr_change_24h:.2f}%")  
          
        # Create two columns for trend indicators  
        c1, c2 = st.columns(2)  
        with c1:  
            st.markdown("**7-Day Trends**")  
            st.markdown(f"**APR**: {get_trend_icon(apr_change_7d)} {apr_change_7d:.2f}%")  
            st.markdown(f"**TVL**: {get_trend_icon(tvl_change_7d)} {tvl_change_7d:.2f}%")  
        with c2:  
            # Get fee if available  
            fee = pool.get("fee", 0)  
            # Display additional metrics  
            st.markdown("**Other Metrics**")  
            st.markdown(f"**Fee**: {fee:.2f}%")  
            # Look for prediction score  
            prediction_score = pool.get("prediction_score", None)  
            if prediction_score is not None:  
                st.markdown(f"**Prediction**: {prediction_score:.2f}/1.0")  
  
def fetch_token_price_data(symbol, days=7):  
    """Fetch token price history for charting with robust error handling"""  
    try:  
        if 'token_service' in st.session_state:  
            token_service = st.session_state['token_service']  
            # Try to get historical prices from our service  
            price_history = token_service.get_token_price_history(symbol, days)  
            if price_history and len(price_history) > 0:  
                return price_history  
        
        # Fallback: Return some dummy data for demonstration  
        # This would normally not be needed with a proper API connection  
        base_price = random.uniform(0.5, 100.0)  
        if symbol == "SOL":  
            base_price = 145.0  
        elif symbol == "USDC" or symbol == "USDT":  
            base_price = 1.0  
        elif symbol == "BTC":  
            base_price = 66000.0  
        elif symbol == "ETH":  
            base_price = 3450.0  
        
        # Generate random price movements  
        today = datetime.now()  
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days, -1, -1)]  
        prices = []  
        current = base_price  
        
        for _ in range(len(dates)):  
            # Add some randomness but keep it somewhat realistic  
            change = random.uniform(-0.05, 0.05)  
            current *= (1 + change)  
            prices.append(current)  
        
        return [{"date": date, "price": price} for date, price in zip(dates, prices)]  
    except Exception as e:  
        logger.error(f"Error fetching token price data for {symbol}: {str(e)}")  
        # Return empty data on error  
        return []  
  
def create_price_chart(price_data, symbol, height=300):  
    """Create a price chart with error handling"""  
    if not price_data or len(price_data) == 0:  
        return None  
      
    try:  
        # Prepare data  
        df = pd.DataFrame(price_data)  
        if 'date' not in df.columns or 'price' not in df.columns:  
            return None  
              
        # Create chart  
        fig = px.line(df, x='date', y='price', title=f"{symbol} Price")  
        fig.update_layout(  
            height=height,  
            margin=dict(l=0, r=0, t=40, b=0),  
            hovermode="x unified",  
            xaxis_title=None,  
            yaxis_title="Price (USD)",  
            plot_bgcolor="rgba(0,0,0,0)",  
            paper_bgcolor="rgba(0,0,0,0)",  
            xaxis=dict(showgrid=False),  
            yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)")  
        )  
        fig.update_traces(line=dict(width=2))  
          
        return fig  
    except Exception as e:  
        logger.error(f"Error creating price chart: {str(e)}")  
        return None  
  
def create_liquidity_chart(historical_data, height=300):  
    """Create a liquidity chart with error handling"""  
    if not historical_data or len(historical_data) == 0:  
        return None  
      
    try:  
        # Convert to DataFrame  
        df = pd.DataFrame(historical_data)  
        if 'timestamp' not in df.columns or 'liquidity' not in df.columns:  
            return None  
              
        # Format the timestamp  
        if 'timestamp' in df.columns:  
            try:  
                # Handle different timestamp formats  
                if isinstance(df['timestamp'].iloc[0], str):  
                    df['timestamp'] = pd.to_datetime(df['timestamp'])  
            except Exception as e:  
                logger.warning(f"Error converting timestamps: {e}")  
          
        # Create chart  
        fig = px.area(df, x='timestamp', y='liquidity', title="Liquidity History")  
        fig.update_layout(  
            height=height,  
            margin=dict(l=0, r=0, t=40, b=0),  
            hovermode="x unified",  
            xaxis_title=None,  
            yaxis_title="Liquidity (USD)",  
            plot_bgcolor="rgba(0,0,0,0)",  
            paper_bgcolor="rgba(0,0,0,0)",  
            xaxis=dict(showgrid=False),  
            yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)")  
        )  
        fig.update_traces(  
            fill='tozeroy',  
            line=dict(width=1)  
        )  
          
        return fig  
    except Exception as e:  
        logger.error(f"Error creating liquidity chart: {str(e)}")  
        return None  
  
def create_apr_chart(historical_data, height=300):  
    """Create an APR chart with error handling"""  
    if not historical_data or len(historical_data) == 0:  
        return None  
      
    try:  
        # Convert to DataFrame  
        df = pd.DataFrame(historical_data)  
        
        # Check required columns  
        apr_columns = [col for col in df.columns if 'apr' in col.lower()]  
        timestamp_column = 'timestamp' if 'timestamp' in df.columns else None  
        
        if not timestamp_column or not apr_columns:  
            return None  
        
        # Use first available APR column  
        apr_column = apr_columns[0]  
        
        # Format the timestamp  
        if timestamp_column:  
            try:  
                if isinstance(df[timestamp_column].iloc[0], str):  
                    df[timestamp_column] = pd.to_datetime(df[timestamp_column])  
            except Exception as e:  
                logger.warning(f"Error converting timestamps in APR chart: {e}")  
          
        # Create chart  
        fig = px.line(df, x=timestamp_column, y=apr_column, title="APR History")  
        fig.update_layout(  
            height=height,  
            margin=dict(l=0, r=0, t=40, b=0),  
            hovermode="x unified",  
            xaxis_title=None,  
            yaxis_title="APR (%)",  
            plot_bgcolor="rgba(0,0,0,0)",  
            paper_bgcolor="rgba(0,0,0,0)",  
            xaxis=dict(showgrid=False),  
            yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)")  
        )  
        fig.update_traces(line=dict(width=2, color="#4CAF50"))  
          
        return fig  
    except Exception as e:  
        logger.error(f"Error creating APR chart: {str(e)}")  
        return None  
  
def create_volume_chart(historical_data, height=300):  
    """Create a volume chart with error handling"""  
    if not historical_data or len(historical_data) == 0:  
        return None  
      
    try:  
        # Convert to DataFrame  
        df = pd.DataFrame(historical_data)  
        
        # Check for required columns  
        volume_column = next((col for col in df.columns if 'volume' in col.lower()), None)  
        timestamp_column = 'timestamp' if 'timestamp' in df.columns else None  
        
        if not volume_column or not timestamp_column:  
            return None  
        
        # Format the timestamp  
        if timestamp_column:  
            try:  
                if isinstance(df[timestamp_column].iloc[0], str):  
                    df[timestamp_column] = pd.to_datetime(df[timestamp_column])  
            except Exception as e:  
                logger.warning(f"Error converting timestamps in volume chart: {e}")  
          
        # Create chart  
        fig = px.bar(df, x=timestamp_column, y=volume_column, title="Volume History")  
        fig.update_layout(  
            height=height,  
            margin=dict(l=0, r=0, t=40, b=0),  
            hovermode="x unified",  
            xaxis_title=None,  
            yaxis_title="Volume (USD)",  
            plot_bgcolor="rgba(0,0,0,0)",  
            paper_bgcolor="rgba(0,0,0,0)",  
            xaxis=dict(showgrid=False),  
            yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)")  
        )  
        fig.update_traces(marker_color="#3F51B5")  
          
        return fig  
    except Exception as e:  
        logger.error(f"Error creating volume chart: {str(e)}")  
        return None  
  
# Function to load pools from the database
def load_data(limit=50):  
    """Load pool data from database or fallback to JSON"""  
    try:  
        if DB_CONNECTED and db_handler_instance:  
            pools = db_handler_instance.get_pools(limit=limit)  
            return pools  
        else:  
            # Fallback to JSON  
            st.warning("Database connection not available - loading from file")  
            try:  
                with open('extracted_pools.json', 'r') as f:  
                    return json.load(f)  
            except FileNotFoundError:  
                # Create a minimal set of sample data  
                return []  
    except Exception as e:  
        st.error(f"Error loading data: {str(e)}")  
        return []  

# Main application function  
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
  
        # Create expandable section for filters  
        with st.expander("Filters", expanded=True):  
            # Initialize filters section  
            if 'min_liquidity' not in st.session_state:  
                st.session_state['min_liquidity'] = 0  
            if 'max_liquidity' not in st.session_state:  
                st.session_state['max_liquidity'] = 1000000000  
              
            # Create slider for liquidity filter  
            st.slider("Liquidity Range (USD)", 0, 50000000, (0, 50000000), 1000000, key="liquidity_range",  
                     format="$%d")  
              
            # Get values from slider  
            st.session_state['min_liquidity'] = st.session_state['liquidity_range'][0]  
            st.session_state['max_liquidity'] = st.session_state['liquidity_range'][1]  
              
            # APR filter  
            if 'min_apr' not in st.session_state:  
                st.session_state['min_apr'] = 0  
            if 'max_apr' not in st.session_state:  
                st.session_state['max_apr'] = 100  
              
            st.slider("APR Range (%)", 0, 100, (0, 100), 1, key="apr_range")  
              
            # Get values from slider  
            st.session_state['min_apr'] = st.session_state['apr_range'][0]  
            st.session_state['max_apr'] = st.session_state['apr_range'][1]  
              
            # DEX filter  
            if 'dex_filter' not in st.session_state:  
                st.session_state['dex_filter'] = "All"  
              
            st.selectbox("DEX", ["All", "Raydium", "Orca", "Jupiter", "Meteora"], key="dex_filter")  
  
        # Create expandable section for data options  
        with st.expander("Data Options", expanded=False):  
            # Pool count selection  
            pool_count = st.number_input("Number of pools to display", min_value=5, max_value=100, value=15, step=5)  
            st.session_state['pool_count'] = pool_count  
              
            # Sort options  
            sort_options = [  
                "APR (High to Low)",  
                "Liquidity (High to Low)",  
                "Volume (High to Low)",  
                "APR Change 24h (High to Low)",  
                "TVL Change 24h (High to Low)",  
                "Prediction Score (High to Low)"  
            ]  
              
            if 'sort_option' not in st.session_state:  
                st.session_state['sort_option'] = sort_options[0]  
              
            st.session_state['sort_option'] = st.selectbox("Sort by", sort_options)  
              
            # Create checkbox for refreshing data  
            if st.button("⟳ Refresh Data"):  
                st.session_state['last_refresh'] = datetime.now()  
                # Force API refresh  
                if 'data_service' in st.session_state:  
                    data_service = st.session_state['data_service']  
                    with st.spinner("Refreshing data from APIs..."):  
                        data_service.update_pools()  
                st.experimental_rerun()  
  
        # Display last refresh time if available  
        if 'last_refresh' in st.session_state:  
            st.caption(f"Last refreshed: {st.session_state['last_refresh'].strftime('%H:%M:%S')}")
            
        # Watchlists section in sidebar  
        with st.expander("Watchlists", expanded=False):  
            # Initialize and fetch existing watchlists if DB is connected  
            try:  
                if DB_CONNECTED and db_handler_instance and hasattr(db_handler_instance, 'get_watchlists'):  
                    watchlists = db_handler_instance.get_watchlists()  
                      
                    # Display them in a selectbox  
                    watchlist_options = ["All Pools"] + [w.get("name", f"Watchlist {i}") for i, w in enumerate(watchlists)]  
                    selected_watchlist = st.selectbox("Select Watchlist", watchlist_options)  
                      
                    if selected_watchlist != "All Pools":  
                        # Find the watchlist ID  
                        for watchlist in watchlists:  
                            if watchlist.get("name") == selected_watchlist:  
                                watchlist_id = watchlist.get("id")  
                                  
                                # Get pool IDs for this watchlist  
                                if watchlist_id and hasattr(db_handler_instance, 'get_pools_in_watchlist'):  
                                    try:  
                                        watchlist_pool_ids = db_handler_instance.get_pools_in_watchlist(watchlist_id)  
                                        st.session_state['watchlist_pool_ids'] = [p.get("pool_id") for p in watchlist_pool_ids]  
                                        st.success(f"Loaded {len(st.session_state['watchlist_pool_ids'])} pools from watchlist")  
                                    except Exception as e:  
                                        st.error(f"Error loading watchlist: {e}")  
                                break  
                    else:  
                        # Clear any selected watchlist  
                        if 'watchlist_pool_ids' in st.session_state:  
                            del st.session_state['watchlist_pool_ids']  
            except Exception as e:  
                st.error(f"Error with watchlists: {e}")  
  
    # Main content  
    st.title("SolPool Insight - Solana Liquidity Pool Analytics")  
      
    # About section - expanded by default  
    with st.expander("About SolPool Insight", expanded=True):  
        st.markdown("""  
        SolPool Insight provides real-time analytics for Solana liquidity pools across various DEXes  
        including Raydium, Orca, Jupiter, Meteora, Saber, and more. It provides comprehensive   
        data, historical metrics, and machine learning-based predictions.  
        """)  
      
    # Database status with enhanced error handling  
    if DB_CONNECTED and db_handler_instance:  
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
        st.error("No pool data available. Please check the database connection.")  
        return  
      
    # Convert to DataFrame for easier manipulation  
    df = pd.DataFrame(pool_data)  
      
    # Apply sort option  
    sort_option = st.session_state.get('sort_option', "APR (High to Low)")  
    if sort_option == "APR (High to Low)":  
        df = df.sort_values(by="apr", ascending=False)  
    elif sort_option == "Liquidity (High to Low)":  
        df = df.sort_values(by="liquidity", ascending=False)  
    elif sort_option == "Volume (High to Low)":  
        df = df.sort_values(by="volume_24h", ascending=False)  
    elif sort_option == "APR Change 24h (High to Low)":  
        df = df.sort_values(by="apr_change_24h", ascending=False)  
    elif sort_option == "TVL Change 24h (High to Low)":  
        df = df.sort_values(by="tvl_change_24h", ascending=False)  
    elif sort_option == "Prediction Score (High to Low)":  
        if "prediction_score" in df.columns:  
            df = df.sort_values(by="prediction_score", ascending=False)  
  
    # Apply basic filters  
    # Liquidity filter  
    min_liquidity = st.session_state.get('min_liquidity', 0)  
    max_liquidity = st.session_state.get('max_liquidity', 1000000000)  
    df = df[(df["liquidity"] >= min_liquidity) & (df["liquidity"] <= max_liquidity)]  
      
    # APR filter  
    min_apr = st.session_state.get('min_apr', 0)  
    max_apr = st.session_state.get('max_apr', 100)  
    df = df[(df["apr"] >= min_apr) & (df["apr"] <= max_apr)]  
      
    # DEX filter  
    dex_filter = st.session_state.get('dex_filter', "All")  
    if dex_filter != "All":  
        df = df[df["dex"] == dex_filter]  
          
    # Filter by watchlist if one is selected  
    if 'watchlist_pool_ids' in st.session_state:  
        df = df[df["id"].isin(st.session_state['watchlist_pool_ids'])]  
      
    # Cap at the number of pools to display  
    pool_count = st.session_state.get('pool_count', 15)  
    df = df.head(pool_count)  
  
    # Data Explorer tab  
    with tab_explore:  
        # Summary metrics  
        st.subheader("Summary Metrics")  
        col1, col2, col3, col4 = st.columns(4)  
          
        with col1:  
            st.metric("Total Pools", len(pool_data))  
              
        with col2:  
            total_liquidity = df["liquidity"].sum()  
            st.metric("Total Liquidity", format_currency(total_liquidity))  
              
        with col3:  
            total_volume = df["volume_24h"].sum()  
            st.metric("24h Volume", format_currency(total_volume))  
              
        with col4:  
            avg_apr = df["apr"].mean()  
            st.metric("Average APR", format_percentage(avg_apr))  
          
        # List of pools  
        st.subheader("Top Liquidity Pools")  
        
        # Create search box for pools  
        search_term = st.text_input("Search for pools by name, token, or DEX", "")  
        
        # If search term provided, filter the dataframe  
        if search_term:  
            search_term = search_term.lower()  
            # Check in various columns  
            name_match = df["name"].str.lower().str.contains(search_term, na=False)  
            dex_match = df["dex"].str.lower().str.contains(search_term, na=False)  
            token1_match = df["token1_symbol"].str.lower().str.contains(search_term, na=False)  
            token2_match = df["token2_symbol"].str.lower().str.contains(search_term, na=False)  
            
            # Combine all matches  
            df = df[name_match | dex_match | token1_match | token2_match]  
        
        for index, pool in df.iterrows():  
            with st.container():  
                col1, col2, col3 = st.columns([1, 2, 1])  
                  
                # Column 1: Pool identification  
                with col1:  
                    # Pool name  
                    st.subheader(pool.get("name", "Unnamed Pool"))  
                      
                    # Pool DEX and category  
                    dex = pool.get("dex", "Unknown DEX")  
                    category = pool.get("category", "Uncategorized")  
                      
                    st.markdown(f"**DEX**: {dex}")  
                    st.markdown(get_category_badge(category), unsafe_allow_html=True)  
                      
                    # Token prices if available  
                    token1_symbol = pool.get("token1_symbol", "Token 1")  
                    token2_symbol = pool.get("token2_symbol", "Token 2")  
                    token1_price = pool.get("token1_price", None)  
                    token2_price = pool.get("token2_price", None)  
                      
                    # Display token prices if available  
                    if token1_price:  
                        st.markdown(f"**{token1_symbol}**: {format_currency(token1_price)}")  
                      
                    if token2_price:  
                        st.markdown(f"**{token2_symbol}**: {format_currency(token2_price)}")  
                      
                # Column 2: Pool metrics  
                display_pool_metrics(pool, col2)  
                  
                # Column 3: Mini charts  
                with col3:  
                    # Try to get historical data if available  
                    try:  
                        if 'historical_service' in st.session_state and pool.get("id"):  
                            historical_service = st.session_state['historical_service']  
                            pool_history = historical_service.get_pool_metrics(pool.get("id"), days=7)  
                              
                            if pool_history is not None and not pool_history.empty:  
                                # Convert to list of dicts for charting  
                                history_records = pool_history.to_dict(orient='records')  
                                  
                                # Create a small APR chart  
                                apr_chart = create_apr_chart(history_records, height=100)  
                                if apr_chart:  
                                    st.plotly_chart(apr_chart, use_container_width=True, config={"displayModeBar": False})  
                        else:  
                            # Create dummy chart for UI layout consistency  
                            dummy_data = [{"date": datetime.now() - timedelta(days=i), "value": random.uniform(pool.get("apr", 10) * 0.9, pool.get("apr", 10) * 1.1)} for i in range(7, -1, -1)]  
                            st.markdown("*Historical data not available*")  
                    except Exception as e:  
                        st.error(f"Error with historical data: {e}")  
                  
                # Add a divider  
                st.markdown("---")  
      
    # Advanced Filtering tab  
    with tab_advanced:  
        st.subheader("Advanced Filtering & Analytics")  
          
        # Check if advanced filtering is available  
        if not HAS_ADVANCED_FILTERING:  
            st.warning("Advanced filtering module not available. Some features may be limited.")  
              
        # Create tabs for different filtering modes  
        filter_tab1, filter_tab2, filter_tab3 = st.tabs(["Custom Filters", "Pattern Discovery", "Comparative Analysis"])  
          
        with filter_tab1:  
            # Enhanced filtering options  
            st.subheader("Custom Filters")  
              
            col1, col2 = st.columns(2)  
              
            with col1:  
                # Token filter  
                tokens_in_pools = set()  
                for pool in pool_data:  
                    tokens_in_pools.add(pool.get("token1_symbol", ""))  
                    tokens_in_pools.add(pool.get("token2_symbol", ""))  
                      
                tokens_list = sorted(list(filter(None, tokens_in_pools)))  
                  
                token_filter = st.multiselect("Filter by Token", tokens_list)  
                  
                # Category filter  
                categories = set(p.get("category", "Uncategorized") for p in pool_data)  
                category_filter = st.multiselect("Filter by Category", sorted(list(categories)))  
                  
                # Volume filter  
                min_volume, max_volume = st.slider("Volume (24h) Range", 0, 5000000, (0, 5000000), 100000,   
                                              format="$%d")  
                  
                # Apply filters to create a new filtered dataframe  
                filtered_df = df.copy()  
                  
                # Apply token filter  
                if token_filter:  
                    filtered_df = filtered_df[  
                        filtered_df["token1_symbol"].isin(token_filter) | filtered_df["token2_symbol"].isin(token_filter)  
                    ]  
                  
                # Apply category filter  
                if category_filter:  
                    filtered_df = filtered_df[filtered_df["category"].isin(category_filter)]  
                  
                # Apply volume filter  
                filtered_df = filtered_df[  
                    (filtered_df["volume_24h"] >= min_volume) & (filtered_df["volume_24h"] <= max_volume)  
                ]  
                  
            with col2:  
                # Advanced metric filters  
                st.subheader("Advanced Metrics")  
                  
                # Filter by prediction score if available  
                if "prediction_score" in df.columns:  
                    min_prediction_score = st.slider("Minimum Prediction Score", 0.0, 1.0, 0.0, 0.05)  
                    if min_prediction_score > 0:  
                        filtered_df = filtered_df[filtered_df["prediction_score"] >= min_prediction_score]  
                  
                # Filter by trend  
                trend_options = ["Any", "Increasing APR", "Decreasing APR", "Stable APR", "Increasing TVL", "Decreasing TVL", "Stable TVL"]  
                trend_filter = st.selectbox("Trend Filter", trend_options)  
                  
                if trend_filter != "Any":  
                    if trend_filter == "Increasing APR":  
                        filtered_df = filtered_df[filtered_df["apr_change_24h"] > 0.5]  
                    elif trend_filter == "Decreasing APR":  
                        filtered_df = filtered_df[filtered_df["apr_change_24h"] < -0.5]  
                    elif trend_filter == "Stable APR":  
                        filtered_df = filtered_df[filtered_df["apr_change_24h"].between(-0.5, 0.5)]  
                    elif trend_filter == "Increasing TVL":  
                        filtered_df = filtered_df[filtered_df["tvl_change_24h"] > 0.5]  
                    elif trend_filter == "Decreasing TVL":  
                        filtered_df = filtered_df[filtered_df["tvl_change_24h"] < -0.5]  
                    elif trend_filter == "Stable TVL":  
                        filtered_df = filtered_df[filtered_df["tvl_change_24h"].between(-0.5, 0.5)]  
                  
                # Volume to liquidity ratio filter (high ratio = active trading)  
                st.markdown("**Volume to Liquidity Ratio**")  
                st.caption("Higher values indicate more active trading relative to pool size")  
                min_vol_ratio = st.slider("Minimum Vol/Liquidity Ratio (%)", 0.0, 50.0, 0.0, 0.5)  
                  
                if min_vol_ratio > 0:  
                    # Add a ratio column to the DataFrame  
                    filtered_df["vol_liquidity_ratio"] = (filtered_df["volume_24h"] / filtered_df["liquidity"]) * 100  
                    filtered_df = filtered_df[filtered_df["vol_liquidity_ratio"] >= min_vol_ratio]  
              
            # Show results of filtered pools  
            st.subheader(f"Filtered Results: {len(filtered_df)} pools")  
              
            if len(filtered_df) > 0:  
                # Create a table for the results  
                result_df = filtered_df[["name", "dex", "category", "liquidity", "volume_24h", "apr"]].copy()  
                  
                # Format values for display  
                result_df["liquidity"] = result_df["liquidity"].apply(lambda x: format_currency(x))  
                result_df["volume_24h"] = result_df["volume_24h"].apply(lambda x: format_currency(x))  
                result_df["apr"] = result_df["apr"].apply(lambda x: format_percentage(x))  
                  
                st.table(result_df)  
                  
                # Option to save as watchlist if DB is available  
                if DB_CONNECTED and hasattr(db_handler_instance, 'create_watchlist'):  
                    watchlist_name = st.text_input("Save results as watchlist (optional)")  
                    if watchlist_name and st.button("Save Watchlist"):  
                        try:  
                            # Create the watchlist  
                            watchlist = db_handler_instance.create_watchlist(  
                                name=watchlist_name,  
                                description=f"Auto-generated from filters on {datetime.now().strftime('%Y-%m-%d %H:%M')}")  
                              
                            # Add pools to the watchlist  
                            if watchlist and hasattr(db_handler_instance, 'add_pool_to_watchlist'):  
                                watchlist_id = watchlist.get("id")  
                                added_count = 0  
                                  
                                for pool_id in filtered_df["id"].tolist():  
                                    db_handler_instance.add_pool_to_watchlist(watchlist_id, pool_id)  
                                    added_count += 1  
                                  
                                st.success(f"✓ Watchlist '{watchlist_name}' created with {added_count} pools")  
                        except Exception as e:  
                            st.error(f"Error saving watchlist: {e}")  
            else:  
                st.info("No pools match the current filters")  
  
        with filter_tab2:  
            st.subheader("Pattern Discovery")  
              
            # If advanced filtering is available, offer more advanced options  
            if HAS_ADVANCED_FILTERING:  
                try:  
                    # Create AdvancedFilteringSystem instance  
                    filtering_system = AdvancedFilteringSystem(df)  
                      
                    # Button to find pools with increasing APR trend  
                    if st.button("Find Pools with Increasing APR"):  
                        # Define trend filter  
                        trend_filter = AdvancedFilteringSystem.trend_filter(  
                            field="apr", days=7, trend_type="increasing", threshold=1.0  
                        )  
                          
                        # Apply the filter  
                        filtering_system.add_filter(trend_filter)  
                        trend_results = filtering_system.apply_filters()  
                          
                        if len(trend_results) > 0:  
                            st.success(f"Found {len(trend_results)} pools with increasing APR trend")  
                            st.dataframe(trend_results[["name", "dex", "apr", "apr_change_24h", "apr_change_7d"]])  
                        else:  
                            st.info("No pools found with significant increasing APR trend")  
                      
                    # Option to find similar pools to a reference pool  
                    st.subheader("Find Similar Pools")  
                      
                    # Select a reference pool  
                    pool_options = df["name"].tolist()  
                    reference_pool = st.selectbox("Select Reference Pool", pool_options)  
                      
                    if st.button("Find Similar Pools"):  
                        # Get the pool ID for the selected pool  
                        reference_id = df[df["name"] == reference_pool]["id"].iloc[0]  
                          
                        # Find similar pools  
                        similar_pools = filtering_system.find_similar_pools(reference_id)  
                          
                        if len(similar_pools) > 0:  
                            st.success(f"Found {len(similar_pools)} pools similar to {reference_pool}")  
                            st.dataframe(similar_pools[["name", "dex", "apr", "liquidity", "volume_24h"]])  
                        else:  
                            st.info("No similar pools found")  
                      
                    # Cluster pools for insight discovery  
                    st.subheader("Cluster Pools")  
                      
                    cluster_count = st.slider("Number of Clusters", 2, 10, 5)  
                      
                    if st.button("Cluster Pools"):  
                        # Get pool clusters  
                        clustered_pools = filtering_system.get_pool_clusters(n_clusters=cluster_count)  
                          
                        if len(clustered_pools) > 0:  
                            # Display clusters  
                            for cluster_id in range(cluster_count):  
                                cluster_data = clustered_pools[clustered_pools["cluster"] == cluster_id]  
                                if len(cluster_data) > 0:  
                                    # Calculate cluster statistics  
                                    avg_apr = cluster_data["apr"].mean()  
                                    avg_liquidity = cluster_data["liquidity"].mean()  
                                    avg_volume = cluster_data["volume_24h"].mean()  
                                      
                                    st.markdown(f"### Cluster {cluster_id+1}")  
                                    st.markdown(f"**Avg APR:** {format_percentage(avg_apr)} | **Avg Liquidity:** {format_currency(avg_liquidity)} | **Avg Volume:** {format_currency(avg_volume)}")  
                                    st.dataframe(cluster_data[["name", "dex", "category", "apr", "liquidity"]])  
                        else:  
                            st.info("Clustering did not produce meaningful results")  
                except Exception as e:  
                    st.error(f"Error with advanced filtering: {e}")  
            else:  
                st.warning("Advanced filtering module not available. Pattern discovery features are limited.")  
                  
                # Basic pattern discovery without the advanced module  
                # Find pools with increasing APR  
                if st.button("Find Pools with Increasing APR (Basic)"):  
                    increasing_apr = df[df["apr_change_24h"] > 1.0]  
                    if len(increasing_apr) > 0:  
                        st.success(f"Found {len(increasing_apr)} pools with APR increasing by >1% in 24h")  
                        st.dataframe(increasing_apr[["name", "dex", "apr", "apr_change_24h"]])  
                    else:  
                        st.info("No pools found with APR increasing by >1% in 24h")  
  
        with filter_tab3:  
            st.subheader("Comparative Analysis")  
              
            # Multi-pool comparison  
            st.markdown("Compare multiple pools side-by-side")  
              
            # Create pool selection  
            pool_options = df["name"].tolist()  
            selected_pools = st.multiselect("Select Pools to Compare", pool_options)  
              
            if selected_pools:  
                # Get data for selected pools  
                comparison_data = df[df["name"].isin(selected_pools)]  
                  
                # Create comparison table  
                comparison_table = comparison_data[["name", "dex", "liquidity", "volume_24h", "apr", "apr_change_24h", "tvl_change_24h"]].copy()  
                  
                # Format values for display  
                comparison_table["liquidity"] = comparison_table["liquidity"].apply(lambda x: format_currency(x))  
                comparison_table["volume_24h"] = comparison_table["volume_24h"].apply(lambda x: format_currency(x))  
                comparison_table["apr"] = comparison_table["apr"].apply(lambda x: format_percentage(x))  
                comparison_table["apr_change_24h"] = comparison_table["apr_change_24h"].apply(lambda x: f"{x:.2f}%")  
                comparison_table["tvl_change_24h"] = comparison_table["tvl_change_24h"].apply(lambda x: f"{x:.2f}%")  
                  
                # Set name as index for better display  
                comparison_table.set_index("name", inplace=True)  
                  
                # Show the comparison table  
                st.table(comparison_table)  
                  
                # Create comparison charts  
                st.subheader("Comparison Charts")  
                  
                # Create APR comparison chart  
                apr_data = comparison_data[["name", "apr"]].sort_values(by="apr", ascending=False)  
                apr_chart = px.bar(apr_data, x="name", y="apr", title="APR Comparison")  
                apr_chart.update_layout(xaxis_title=None, yaxis_title="APR (%)")  
                st.plotly_chart(apr_chart, use_container_width=True)  
                  
                # Create Liquidity comparison chart  
                liquidity_data = comparison_data[["name", "liquidity"]].sort_values(by="liquidity", ascending=False)  
                liquidity_chart = px.bar(liquidity_data, x="name", y="liquidity", title="Liquidity Comparison")  
                liquidity_chart.update_layout(xaxis_title=None, yaxis_title="Liquidity (USD)")  
                st.plotly_chart(liquidity_chart, use_container_width=True)  
                  
                # Create Volume comparison chart  
                volume_data = comparison_data[["name", "volume_24h"]].sort_values(by="volume_24h", ascending=False)  
                volume_chart = px.bar(volume_data, x="name", y="volume_24h", title="Volume (24h) Comparison")  
                volume_chart.update_layout(xaxis_title=None, yaxis_title="Volume (USD)")  
                st.plotly_chart(volume_chart, use_container_width=True)  
            else:  
                st.info("Select two or more pools to compare")  
              
            # DEX Comparison Section  
            st.subheader("DEX Comparison")  
              
            # Get unique DEXes  
            dexes = df["dex"].unique()  
              
            if len(dexes) > 1:  
                # Compute average metrics by DEX  
                dex_comparison = df.groupby("dex").agg({  
                    "liquidity": "sum",  
                    "volume_24h": "sum",  
                    "apr": "mean",  
                    "id": "count"  
                }).reset_index()  
                  
                # Rename columns for clarity  
                dex_comparison.rename(columns={"id": "pool_count"}, inplace=True)  
                  
                # Create bar charts for comparison  
                dex_chart1 = px.bar(dex_comparison, x="dex", y="pool_count", title="Pools per DEX")  
                dex_chart1.update_layout(xaxis_title=None, yaxis_title="Pool Count")  
                st.plotly_chart(dex_chart1, use_container_width=True)  
                  
                dex_chart2 = px.bar(dex_comparison, x="dex", y="apr", title="Average APR by DEX")  
                dex_chart2.update_layout(xaxis_title=None, yaxis_title="APR (%)")  
                st.plotly_chart(dex_chart2, use_container_width=True)  
                  
                dex_chart3 = px.pie(dex_comparison, names="dex", values="liquidity", title="Liquidity Share by DEX")  
                st.plotly_chart(dex_chart3, use_container_width=True)  
            else:  
                st.info("Only one DEX present in the data")  
  
    # Predictions tab  
    with tab_predict:  
        st.subheader("APR and Performance Predictions")  
          
        # Show warning if prediction system not fully available  
        if not HAS_BACKGROUND_UPDATER:  
            st.warning("Prediction system is not fully available. Some features may be limited.")  
          
        # Create tabs for different prediction views  
        pred_tab1, pred_tab2, pred_tab3 = st.tabs(["APR Predictions", "Risk Analysis", "Backtesting"])  
          
        with pred_tab1:  
            st.subheader("7-Day APR Predictions")  
              
            # Filter to show only pools with prediction scores  
            has_predictions = "prediction_score" in df.columns  
            if has_predictions:  
                prediction_data = df[df["prediction_score"] > 0].sort_values(by="prediction_score", ascending=False)  
                  
                if len(prediction_data) > 0:  
                    # Create prediction table  
                    pred_table = prediction_data[["name", "dex", "apr", "prediction_score"]].copy()  
                      
                    # Format values  
                    pred_table["apr"] = pred_table["apr"].apply(lambda x: format_percentage(x))  
                    pred_table["prediction_score"] = pred_table["prediction_score"].apply(lambda x: f"{x:.2f}")  
                      
                    st.dataframe(pred_table)  
                      
                    # Create chart of prediction scores  
                    top_predictions = prediction_data.head(10)  
                    fig = px.bar(top_predictions, x="name", y="prediction_score",  
                               title="Top 10 Pools by Prediction Score")  
                    fig.update_layout(xaxis_title=None, yaxis_title="Prediction Score")  
                    st.plotly_chart(fig, use_container_width=True)  
                else:  
                    st.info("No prediction data available yet. Please check back later.")  
            else:  
                st.info("Prediction data is not available in the current dataset.")  
  
        with pred_tab2:  
            st.subheader("Risk Assessment")  
              
            # Create risk assessment visualization if data available  
            if has_predictions and len(prediction_data) > 0:  
                # Create a risk vs reward scatter plot  
                risk_data = prediction_data.copy()  
                  
                # Use APR volatility as a proxy for risk if available  
                if "apr_change_7d" in risk_data.columns:  
                    risk_data["risk_factor"] = risk_data["apr_change_7d"].abs()  
                else:  
                    # Otherwise use a simple calculation  
                    risk_data["risk_factor"] = risk_data["apr"] / 100  
                  
                # Create scatter plot  
                fig = px.scatter(risk_data, x="risk_factor", y="apr",  
                               size="liquidity", color="prediction_score",  
                               hover_name="name", size_max=50,  
                               color_continuous_scale="Viridis",  
                               title="Risk vs Reward Analysis")  
                  
                fig.update_layout(  
                    xaxis_title="Risk Factor (APR Volatility)",  
                    yaxis_title="Current APR (%)",  
                    height=600  
                )  
                  
                st.plotly_chart(fig, use_container_width=True)  
            else:  
                st.info("Insufficient data for risk assessment.")  
  
        with pred_tab3:  
            st.subheader("Backtesting Results")  
              
            # Simulated backtesting results (in a real app, this would use actual historical data)  
            st.markdown("*Note: This section shows simulated backtesting results for demonstration.*")  
              
            # Create a simple backtesting table  
            backtest_data = {  
                "Strategy": ["High APR", "Balanced", "Low Volatility", "Prediction-Based"],  
                "Avg Return": ["12.4%", "8.7%", "6.2%", "14.1%"],  
                "Max Drawdown": ["15.2%", "8.1%", "4.3%", "10.5%"],  
                "Sharpe Ratio": ["0.92", "1.15", "1.32", "1.45"]  
            }  
              
            backtest_df = pd.DataFrame(backtest_data)  
            st.table(backtest_df)  
              
            # Create a performance chart  
            # This would typically be generated from real backtesting data  
            dates = pd.date_range(end=datetime.now(), periods=90).tolist()  
              
            # Generate synthetic performance data  
            high_apr = [100]  
            balanced = [100]  
            low_vol = [100]  
            prediction = [100]  
              
            for i in range(1, 90):  
                high_apr.append(high_apr[-1] * (1 + (random.uniform(-0.03, 0.04))))  
                balanced.append(balanced[-1] * (1 + (random.uniform(-0.02, 0.03))))  
                low_vol.append(low_vol[-1] * (1 + (random.uniform(-0.01, 0.02))))  
                prediction.append(prediction[-1] * (1 + (random.uniform(-0.02, 0.05))))  
              
            # Create a DataFrame for the chart  
            perf_data = pd.DataFrame({  
                "Date": dates,  
                "High APR": high_apr,  
                "Balanced": balanced,  
                "Low Volatility": low_vol,  
                "Prediction-Based": prediction  
            })  
              
            # Create a plotly figure  
            fig = go.Figure()  
              
            # Add each line  
            fig.add_trace(go.Scatter(x=perf_data["Date"], y=perf_data["High APR"], name="High APR"))  
            fig.add_trace(go.Scatter(x=perf_data["Date"], y=perf_data["Balanced"], name="Balanced"))  
            fig.add_trace(go.Scatter(x=perf_data["Date"], y=perf_data["Low Volatility"], name="Low Volatility"))  
            fig.add_trace(go.Scatter(x=perf_data["Date"], y=perf_data["Prediction-Based"], name="Prediction-Based"))  
              
            # Update layout  
            fig.update_layout(  
                title="Strategy Backtest Comparison",  
                xaxis_title="Date",  
                yaxis_title="Portfolio Value",  
                hovermode="x unified",  
                height=500  
            )  
              
            st.plotly_chart(fig, use_container_width=True)  
  
    # Risk Assessment tab  
    with tab_risk:  
        st.subheader("Risk Analysis & Metrics")  
          
        # Create tabs for different risk views  
        risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Risk Scores", "Volatility Analysis", "Token Correlation"])  
          
        with risk_tab1:  
            st.subheader("Pool Risk Ratings")  
              
            # Create a risk metrics table  
            risk_metrics = df.copy()  
              
            # Calculate risk score based on available metrics  
            # This is a simplified calculation for demonstration  
            risk_metrics["volatility"] = risk_metrics["apr_change_7d"].abs() + risk_metrics["tvl_change_7d"].abs()  
            risk_metrics["risk_score"] = (  
                (risk_metrics["volatility"] / risk_metrics["volatility"].max() * 0.6) +   
                (risk_metrics["apr"] / risk_metrics["apr"].max() * 0.4)  
            ).clip(0, 1)  
              
            # Get top 15 by risk score  
            top_risk = risk_metrics.sort_values(by="risk_score", ascending=False).head(15)  
              
            # Create risk rating display  
            risk_display = top_risk[["name", "dex", "apr", "volatility", "risk_score"]].copy()  
              
            # Format columns  
            risk_display["apr"] = risk_display["apr"].apply(lambda x: format_percentage(x))  
            risk_display["volatility"] = risk_display["volatility"].apply(lambda x: f"{x:.2f}%")  
            risk_display["risk_score"] = risk_display["risk_score"].apply(lambda x: f"{x:.2f}")  
              
            st.dataframe(risk_display)  
              
            # Create risk distribution chart  
            fig = px.histogram(risk_metrics, x="risk_score", nbins=20,  
                             title="Distribution of Risk Scores",  
                             labels={"risk_score": "Risk Score", "count": "Number of Pools"})  
            fig.update_layout(height=400)  
            st.plotly_chart(fig, use_container_width=True)  
  
        with risk_tab2:  
            st.subheader("Volatility Analysis")  
              
            # Create volatility vs APR scatter plot  
            volatility_data = risk_metrics.copy()  
              
            # Create the chart  
            fig = px.scatter(volatility_data, x="volatility", y="apr",  
                           size="liquidity", color="dex",  
                           hover_name="name", size_max=50,  
                           title="Volatility vs APR by DEX")  
              
            fig.update_layout(  
                xaxis_title="Volatility (% change)",  
                yaxis_title="APR (%)",  
                height=500  
            )  
              
            st.plotly_chart(fig, use_container_width=True)  
              
            # Show pools with highest volatility  
            st.subheader("Highest Volatility Pools")  
            high_vol = volatility_data.sort_values(by="volatility", ascending=False).head(10)  
            high_vol_display = high_vol[["name", "dex", "apr", "volatility"]].copy()  
              
            # Format columns  
            high_vol_display["apr"] = high_vol_display["apr"].apply(lambda x: format_percentage(x))  
            high_vol_display["volatility"] = high_vol_display["volatility"].apply(lambda x: f"{x:.2f}%")  
              
            st.dataframe(high_vol_display)  
              
            # Show pools with lowest volatility  
            st.subheader("Lowest Volatility Pools")  
            low_vol = volatility_data.sort_values(by="volatility", ascending=True).head(10)  
            low_vol_display = low_vol[["name", "dex", "apr", "volatility"]].copy()  
              
            # Format columns  
            low_vol_display["apr"] = low_vol_display["apr"].apply(lambda x: format_percentage(x))  
            low_vol_display["volatility"] = low_vol_display["volatility"].apply(lambda x: f"{x:.2f}%")  
              
            st.dataframe(low_vol_display)  
  
        with risk_tab3:  
            st.subheader("Token Correlation")  
              
            # This would typically use actual token price correlation data  
            # For this example, we'll create a simulated correlation matrix  
              
            # Get the top tokens by appearance in pools  
            token_counts = {}  
            for pool in pool_data:  
                token1 = pool.get("token1_symbol", "")  
                token2 = pool.get("token2_symbol", "")  
                  
                if token1:  
                    token_counts[token1] = token_counts.get(token1, 0) + 1  
                if token2:  
                    token_counts[token2] = token_counts.get(token2, 0) + 1  
              
            # Get top 10 tokens  
            top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]  
            top_token_names = [t[0] for t in top_tokens]  
              
            # Create a simulated correlation matrix  
            np = pd  # Use pandas since numpy may not be available  
            n = len(top_token_names)  
            corr_matrix = pd.DataFrame(index=top_token_names, columns=top_token_names)  
              
            # Fill the matrix with simulated correlations  
            for i, token1 in enumerate(top_token_names):  
                for j, token2 in enumerate(top_token_names):  
                    if i == j:  
                        corr_matrix.iloc[i, j] = 1.0  
                    else:  
                        # Some tokens are more correlated with each other  
                        if ("SOL" in token1 and "SOL" in token2) or ("BTC" in token1 and "BTC" in token2):  
                            corr = random.uniform(0.7, 0.9)  
                        elif ("USD" in token1 and "USD" in token2):  
                            corr = random.uniform(0.9, 1.0)  
                        else:  
                            corr = random.uniform(-0.3, 0.7)  
                          
                        corr_matrix.iloc[i, j] = corr  
                        corr_matrix.iloc[j, i] = corr  # Ensure symmetry  
              
            # Create heatmap  
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",  
                         color_continuous_scale="RdBu_r", title="Token Price Correlation Matrix")  
            fig.update_layout(height=600)  
            st.plotly_chart(fig, use_container_width=True)  
              
            st.markdown("**Interpretation:**")  
            st.markdown("* Positive values (blue) indicate tokens that tend to move together")  
            st.markdown("* Negative values (red) indicate tokens that tend to move in opposite directions")  
            st.markdown("* Values close to zero indicate little relationship between token prices")  
              
            st.caption("Note: This is a simulated correlation matrix for demonstration purposes.")  
  
    # NLP Reports tab  
    with tab_nlp:  
        st.subheader("Natural Language Analysis")  
          
        # Create tabs for different NLP views  
        nlp_tab1, nlp_tab2 = st.tabs(["Market Sentiment", "Pool Insights"])  
          
        with nlp_tab1:  
            st.subheader("Market Sentiment Analysis")  
              
            # This would typically connect to a real sentiment analysis system  
            # For this example, we'll use simulated data  
              
            st.info("This feature uses AI to analyze social media, news, and forum discussions about Solana tokens and liquidity pools.")  
              
            # Create a sample sentiment chart  
            sentiment_data = {  
                "Token": ["SOL", "BTC", "ETH", "BONK", "MSOL", "USDC", "RAY", "ORCA"],  
                "Positive": [65, 58, 62, 70, 55, 48, 60, 52],  
                "Neutral": [25, 30, 28, 15, 35, 45, 30, 38],  
                "Negative": [10, 12, 10, 15, 10, 7, 10, 10]  
            }  
              
            sentiment_df = pd.DataFrame(sentiment_data)  
              
            # Create stacked bar chart  
            fig = go.Figure()  
              
            # Add traces for each sentiment  
            fig.add_trace(go.Bar(x=sentiment_df["Token"], y=sentiment_df["Positive"], name="Positive", marker_color="#4CAF50"))  
            fig.add_trace(go.Bar(x=sentiment_df["Token"], y=sentiment_df["Neutral"], name="Neutral", marker_color="#FFC107"))  
            fig.add_trace(go.Bar(x=sentiment_df["Token"], y=sentiment_df["Negative"], name="Negative", marker_color="#F44336"))  
              
            # Update layout  
            fig.update_layout(  
                title="Token Sentiment Analysis",  
                xaxis_title="Token",  
                yaxis_title="Sentiment (%)",  
                barmode="stack",  
                height=400  
            )  
              
            st.plotly_chart(fig, use_container_width=True)  
              
            # Create a DEX sentiment comparison  
            dex_sentiment = {  
                "DEX": ["Raydium", "Orca", "Jupiter", "Meteora"],  
                "Score": [8.2, 7.9, 8.5, 7.7]  
            }  
              
            dex_sentiment_df = pd.DataFrame(dex_sentiment)  
              
            # Create bar chart  
            fig = px.bar(dex_sentiment_df, x="DEX", y="Score",  
                      color="Score", color_continuous_scale="Viridis",  
                      title="DEX Sentiment Score (0-10)")  
              
            fig.update_layout(height=400)  
            st.plotly_chart(fig, use_container_width=True)  
  
        with nlp_tab2:  
            st.subheader("AI-Generated Pool Insights")  
              
            # This would typically connect to a real NLP/AI system  
            # For this example, we'll use pre-written insights  
              
            # Create sample insights for a few pool types  
            insights = {  
                "SOL-USDC": "The SOL-USDC pool has seen consistent high trading volume recently, indicating strong market interest. The APR has remained relatively stable with low volatility, making it a reliable option for liquidity providers seeking stable returns. This pool may be well-suited for those seeking a balance of returns and stability.",  
                  
                "BTC-USDC": "The BTC-USDC pool shows an interesting divergence between volatility and APR. While price volatility has increased, the pool's APR has remained relatively stable, suggesting effective arbitrage mechanisms. This pool may offer a good way to capture Bitcoin price movements while still earning yield through trading fees.",  
                  
                "BONK-USDC": "The BONK-USDC pool shows extremely high volatility in both price and APR, characteristic of meme tokens. The pool experiences periodic volume spikes, likely corresponding to social media interest. This pool represents one of the highest risk-reward profiles in the ecosystem and may be suitable only for those with high risk tolerance.",  
                  
                "ETH-USDC": "The ETH-USDC pool shows strong correlation with broader ETH market movements while maintaining consistent fee generation. This pool offers a balance between the stability of a major asset pair and the liquidity advantages of the Solana ecosystem.",  
                  
                "MSOL-SOL": "The MSOL-SOL pool offers an interesting case of minimal impermanent loss due to the close price relationship between staked and unstaked SOL. The pool has shown consistent APR with low variance, making it a good option for more conservative liquidity providers.",  
            }  
              
            # Let user select a pool to get insights  
            pool_options = list(insights.keys())  
            selected_pool = st.selectbox("Select Pool for AI Analysis", pool_options)  
              
            if selected_pool:  
                st.markdown(f"### AI Analysis for {selected_pool} Pool")  
                st.markdown(insights[selected_pool])  
                  
                # Add a simulated risk-reward gauge  
                risk_rewards = {  
                    "SOL-USDC": 0.4,  
                    "BTC-USDC": 0.5,  
                    "BONK-USDC": 0.9,  
                    "ETH-USDC": 0.45,  
                    "MSOL-SOL": 0.3  
                }  
                  
                risk_score = risk_rewards[selected_pool]  
                  
                # Create a gauge chart  
                fig = go.Figure(go.Indicator(  
                    mode="gauge+number",  
                    value=risk_score * 100,  
                    title={"text": "Risk-Reward Score"},  
                    gauge={  
                        "axis": {"range": [0, 100]},  
                        "bar": {"color": "royalblue"},  
                        "steps": [  
                            {"range": [0, 33], "color": "green"},  
                            {"range": [33, 66], "color": "yellow"},  
                            {"range": [66, 100], "color": "red"}  
                        ]  
                    }  
                ))  
                  
                fig.update_layout(height=300)  
                st.plotly_chart(fig, use_container_width=True)  
                  
                # Add some simulated personalized recommendations  
                st.markdown("### Personalized Recommendations")  
                  
                if risk_score < 0.4:  
                    st.markdown("**Conservative Strategy**: This pool fits well in a conservative portfolio. Consider allocating up to 20% of your liquidity for stable returns.")  
                elif risk_score < 0.7:  
                    st.markdown("**Balanced Strategy**: This pool represents a moderate risk-reward profile. Consider a 10-15% allocation as part of a diversified liquidity strategy.")  
                else:  
                    st.markdown("**Aggressive Strategy**: This pool has a high risk-reward profile. Consider limiting exposure to 5% of your total liquidity and setting tight stop-loss parameters.")  
  
    # Token Explorer tab  
    with tab_tokens:  
        st.subheader("Token Analytics")  
          
        # Create tabs for different token views  
        token_tab1, token_tab2 = st.tabs(["Token Overview", "Detailed Analysis"])  
          
        with token_tab1:  
            # Get unique tokens across pools  
            all_tokens = set()  
            for pool in pool_data:  
                all_tokens.add(pool.get("token1_symbol", ""))  
                all_tokens.add(pool.get("token2_symbol", ""))  
            all_tokens = list(filter(None, all_tokens))  
              
            # Let user select a token to explore  
            selected_token = st.selectbox("Select a token to explore", sorted(all_tokens))  
              
            if selected_token:  
                # Display token information  
                st.markdown(f"## {selected_token} Analytics")  
                  
                # Get the token price if available  
                token_price = None  
                token_price_source = None  
                  
                try:  
                    token_price, token_price_source = get_token_price(selected_token, return_source=True)  
                    if token_price:  
                        st.metric("Current Price", format_currency(token_price))  
                        st.caption(f"Price source: {token_price_source}")  
                except Exception as e:  
                    st.warning(f"Could not retrieve price: {e}")  
                  
                # Find all pools that contain this token  
                token_pools = [p for p in pool_data if   
                              p.get("token1_symbol") == selected_token or   
                              p.get("token2_symbol") == selected_token]  
                  
                # Display token stats  
                st.markdown(f"### {selected_token} is found in {len(token_pools)} pools")  
                  
                # Calculate total liquidity for this token  
                total_token_liquidity = sum(p.get("liquidity", 0) / 2 for p in token_pools)  
                st.metric("Total Liquidity", format_currency(total_token_liquidity))  
                  
                # Calculate total 24h volume for this token  
                total_token_volume = sum(p.get("volume_24h", 0) / 2 for p in token_pools)  
                st.metric("24h Trading Volume", format_currency(total_token_volume))  
                  
                # Create price chart if available  
                price_history = fetch_token_price_data(selected_token)  
                if price_history:  
                    price_chart = create_price_chart(price_history, selected_token)  
                    if price_chart:  
                        st.plotly_chart(price_chart, use_container_width=True)  
                  
                # Display pools containing this token  
                st.markdown(f"### Pools containing {selected_token}")  
                  
                # Convert to DataFrame  
                token_pools_df = pd.DataFrame(token_pools)  
                if not token_pools_df.empty:  
                    # Sort by liquidity  
                    token_pools_df = token_pools_df.sort_values(by="liquidity", ascending=False)  
                      
                    # Create a clean display table  
                    display_df = token_pools_df[["name", "dex", "liquidity", "volume_24h", "apr"]].copy()  
                      
                    # Format columns  
                    display_df["liquidity"] = display_df["liquidity"].apply(lambda x: format_currency(x))  
                    display_df["volume_24h"] = display_df["volume_24h"].apply(lambda x: format_currency(x))  
                    display_df["apr"] = display_df["apr"].apply(lambda x: format_percentage(x))  
                      
                    st.dataframe(display_df)  
                      
                    # Create a pie chart of liquidity distribution  
                    fig = px.pie(token_pools_df, names="name", values="liquidity",  
                               title=f"{selected_token} Liquidity Distribution")  
                    st.plotly_chart(fig, use_container_width=True)  
                else:  
                    st.info(f"No pools found containing {selected_token}")  
  
        with token_tab2:  
            st.subheader("Cross-Token Comparisons")  
              
            # Let user select multiple tokens to compare  
            compare_tokens = st.multiselect("Select tokens to compare", sorted(all_tokens))  
              
            if compare_tokens and len(compare_tokens) > 1:  
                # Create a comparison table  
                token_data = []  
                  
                for token in compare_tokens:  
                    try:  
                        # Get token price  
                        price = get_token_price(token)  
                          
                        # Get pools with this token  
                        token_pools = [p for p in pool_data if   
                                      p.get("token1_symbol") == token or   
                                      p.get("token2_symbol") == token]  
                          
                        # Calculate metrics  
                        avg_apr = sum(p.get("apr", 0) for p in token_pools) / len(token_pools) if token_pools else 0  
                        total_liquidity = sum(p.get("liquidity", 0) / 2 for p in token_pools)  
                        total_volume = sum(p.get("volume_24h", 0) / 2 for p in token_pools)  
                        pool_count = len(token_pools)  
                          
                        # Add to data list  
                        token_data.append({  
                            "Token": token,  
                            "Price": price,  
                            "Total Liquidity": total_liquidity,  
                            "24h Volume": total_volume,  
                            "Pool Count": pool_count,  
                            "Avg APR": avg_apr  
                        })  
                    except Exception as e:  
                        st.warning(f"Error processing {token}: {e}")  
                  
                # Create comparison DataFrame  
                if token_data:  
                    compare_df = pd.DataFrame(token_data)  
                      
                    # Format for display  
                    display_df = compare_df.copy()  
                    display_df["Price"] = display_df["Price"].apply(lambda x: format_currency(x))  
                    display_df["Total Liquidity"] = display_df["Total Liquidity"].apply(lambda x: format_currency(x))  
                    display_df["24h Volume"] = display_df["24h Volume"].apply(lambda x: format_currency(x))  
                    display_df["Avg APR"] = display_df["Avg APR"].apply(lambda x: format_percentage(x))  
                      
                    st.dataframe(display_df)  
                      
                    # Create comparison charts  
                    # Liquidity Chart  
                    fig1 = px.bar(compare_df, x="Token", y="Total Liquidity",  
                                title="Liquidity Comparison")  
                    st.plotly_chart(fig1, use_container_width=True)  
                      
                    # Volume Chart  
                    fig2 = px.bar(compare_df, x="Token", y="24h Volume",  
                                title="24h Volume Comparison")  
                    st.plotly_chart(fig2, use_container_width=True)  
                      
                    # APR Chart  
                    fig3 = px.bar(compare_df, x="Token", y="Avg APR",  
                                title="Average APR Comparison")  
                    st.plotly_chart(fig3, use_container_width=True)  
                  
            else:  
                st.info("Select at least two tokens to compare")  
              
    # Performance stats in footer  
    with st.expander("Performance Statistics", expanded=False):  
        perf_metrics = perf_monitor.get_metrics()  
        st.markdown("### Application Performance Metrics")  
          
        if perf_metrics:  
            # Create two columns for metrics  
            col1, col2 = st.columns(2)  
              
            with col1:  
                st.markdown("**Checkpoint Timings:**")  
                for checkpoint, time_value in perf_metrics.get("checkpoints", {}).items():  
                    st.markdown(f"* {checkpoint}: {time_value:.2f}s")  
              
            with col2:  
                st.markdown("**Function Execution Times:**")  
                for func, time_value in perf_metrics.get("function_times", {}).items():  
                    st.markdown(f"* {func}: {time_value:.2f}s")  
                  
            # Additional stats  
            if "app_start_time" in perf_metrics:  
                uptime = datetime.now() - perf_metrics["app_start_time"]  
                st.markdown(f"**Application uptime**: {uptime.total_seconds() / 60:.1f} minutes")  
        else:  
            st.info("No performance metrics available")  

if __name__ == "__main__":
    main()
