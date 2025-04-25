import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Import custom modules
# If running as a standalone app, add the parent directory to the path
if os.path.basename(os.getcwd()) != "workspace":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pool_retrieval_system import PoolRetriever, PoolData
    from advanced_filtering import AdvancedFilteringSystem
except ImportError:
    st.error("Could not import required modules. Make sure pool_retrieval_system.py and advanced_filtering.py are in the same directory.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Solana Liquidity Pool Explorer",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better mobile responsiveness
st.markdown("""
<style>
    /* Mobile-friendly styling */
    .reportview-container .main .block-container {
        max-width: 1000px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Ensure tables are responsive */
    .dataframe {
        width: 100% !important;
        overflow-x: auto;
    }
    
    /* Make sidebar more usable on mobile */
    @media (max-width: 768px) {
        .sidebar .sidebar-content {
            padding-top: 1rem;
        }
    }
    
    /* Make buttons more touch-friendly */
    button {
        min-height: 44px;
    }
    
    /* Improve readability on mobile */
    p, li {
        font-size: 16px !important;
    }
    
    /* Custom metric formatting */
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 14px;
        color: #888;
    }
    
    /* Custom card styling */
    .card {
        padding: 16px;
        border-radius: 8px;
        background-color: #f9f9f9;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 16px;
    }
    
    /* Custom highlighting */
    .highlight {
        background-color: #ffc;
        padding: 2px 4px;
        border-radius: 3px;
    }
    
    /* Custom tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #888;
    }
    
    /* Custom badges for categories */
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        color: white;
    }
    
    .badge-meme {
        background-color: #ff6b6b;
    }
    
    .badge-defi {
        background-color: #4c6ef5;
    }
    
    .badge-major {
        background-color: #37b24d;
    }
    
    .badge-gaming {
        background-color: #f76707;
    }
    
    .badge-stablecoin {
        background-color: #7950f2;
    }
    
    .badge-other {
        background-color: #868e96;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def format_currency(value):
    """Format a value as currency"""
    if pd.isna(value) or value is None:
        return "N/A"
    
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value):
    """Format a value as percentage"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.2f}%"

def get_trend_icon(value):
    """Return an arrow icon based on trend direction"""
    if pd.isna(value) or value is None:
        return "➖"
    return "📈" if value >= 0 else "📉"

def get_trend_color(value):
    """Return a color based on trend direction"""
    if pd.isna(value) or value is None:
        return "gray"
    return "green" if value >= 0 else "red"

def get_prediction_color(score):
    """Return a color based on prediction score"""
    if pd.isna(score) or score is None:
        return "gray"
    
    if score >= 85:
        return "green"
    elif score >= 70:
        return "orange"
    else:
        return "red"

def get_category_badge(category):
    """Return HTML for a category badge"""
    if pd.isna(category) or category is None:
        category = "Other"
    
    category_lower = category.lower()
    
    badge_class = f"badge badge-{category_lower}"
    return f'<span class="{badge_class}">{category}</span>'

def load_sample_data():
    """Load sample data for demonstration"""
    # Check if we have sample data saved
    if os.path.exists("sample_pool_data.json"):
        try:
            with open("sample_pool_data.json", "r") as f:
                data = json.load(f)
                return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
    
    # Generate sample data if we can't load from file
    # This simulates what we would get from the PoolRetriever
    num_pools = 100
    
    # DEXes and their weights
    dexes = ["Raydium", "Orca", "Jupiter", "Meteora", "Saber"]
    dex_weights = [0.4, 0.3, 0.15, 0.1, 0.05]
    
    # Categories and their weights
    categories = ["Major", "Meme", "DeFi", "Gaming", "Stablecoin"]
    category_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    
    # Token symbols for first position
    token1_options = {
        "Major": ["SOL", "BTC", "ETH"],
        "Meme": ["BONK", "SAMO", "DOGWIFHAT", "POPCAT", "BIGMONKEY", "WIF"],
        "DeFi": ["RAY", "JUP", "MNGO", "MARI", "ORCA", "TULIP"],
        "Gaming": ["AURORY", "STAR", "ATLAS", "POLIS", "GARI"],
        "Stablecoin": ["USDC", "USDT", "DAI"]
    }
    
    # Token symbols for second position
    token2_options = {
        "Major": ["USDC", "USDT", "SOL"],
        "Meme": ["USDC", "SOL"],
        "DeFi": ["USDC", "SOL"],
        "Gaming": ["USDC", "SOL"],
        "Stablecoin": ["USDC", "USDT", "DAI"]
    }
    
    # Generate random pool data
    np.random.seed(42)  # For reproducibility
    
    data = []
    for i in range(num_pools):
        # Random dex based on weights
        dex = np.random.choice(dexes, p=dex_weights)
        
        # Random category based on weights
        category = np.random.choice(categories, p=category_weights)
        
        # Random tokens based on category
        token1 = np.random.choice(token1_options[category])
        token2 = np.random.choice(token2_options[category])
        
        # Base metrics based on category
        if category == "Major":
            base_liquidity = np.random.uniform(10_000_000, 50_000_000)
            base_volume = np.random.uniform(1_000_000, 10_000_000)
            base_apr = np.random.uniform(5, 15)
            volatility = np.random.uniform(0.02, 0.06)
        elif category == "Meme":
            base_liquidity = np.random.uniform(1_000_000, 10_000_000)
            base_volume = np.random.uniform(500_000, 5_000_000)
            base_apr = np.random.uniform(15, 40)
            volatility = np.random.uniform(0.1, 0.3)
        elif category == "DeFi":
            base_liquidity = np.random.uniform(2_000_000, 20_000_000)
            base_volume = np.random.uniform(500_000, 3_000_000)
            base_apr = np.random.uniform(10, 25)
            volatility = np.random.uniform(0.05, 0.15)
        elif category == "Gaming":
            base_liquidity = np.random.uniform(1_000_000, 5_000_000)
            base_volume = np.random.uniform(200_000, 1_000_000)
            base_apr = np.random.uniform(12, 30)
            volatility = np.random.uniform(0.08, 0.2)
        else:  # Stablecoin
            base_liquidity = np.random.uniform(20_000_000, 80_000_000)
            base_volume = np.random.uniform(5_000_000, 15_000_000)
            base_apr = np.random.uniform(3, 8)
            volatility = np.random.uniform(0.01, 0.03)
        
        # Generate historical APR changes
        apr_24h_change = np.random.uniform(-0.5, 0.5) * base_apr * 0.05
        apr_7d_change = np.random.uniform(-1, 1) * base_apr * 0.1
        apr_30d_change = np.random.uniform(-2, 2) * base_apr * 0.2
        
        # Generate TVL changes
        tvl_24h_change = np.random.uniform(-0.5, 0.5) * 0.03
        tvl_7d_change = np.random.uniform(-1, 1) * 0.08
        tvl_30d_change = np.random.uniform(-2, 2) * 0.15
        
        # Generate prediction score based on trends and volatility
        # Higher APR, lower volatility, and positive trends lead to higher scores
        score_components = [
            np.random.uniform(0, 30),  # Random base
            base_apr / 40 * 20,  # APR component (max 20 points)
            (1 - volatility / 0.3) * 15,  # Volatility component (max 15 points)
            (tvl_7d_change > 0) * 10,  # Positive 7d TVL trend (10 points)
            (apr_7d_change > 0) * 10,  # Positive 7d APR trend (10 points)
            (1 - abs(tvl_24h_change) / 0.03) * 5,  # Stability component (max 5 points)
            (dex == "Raydium") * 5 + (dex == "Orca") * 3  # DEX reputation (max 5 points)
        ]
        prediction_score = min(100, max(0, sum(score_components)))
        
        # Generate pool data
        pool = {
            "id": f"pool_{i}",
            "name": f"{token1}/{token2}",
            "dex": dex,
            "category": category,
            "token1_symbol": token1,
            "token2_symbol": token2,
            "token1_address": f"{token1}_address_{i}",
            "token2_address": f"{token2}_address_{i}",
            "liquidity": base_liquidity,
            "volume_24h": base_volume,
            "apr": base_apr,
            "apr_24h_ago": base_apr - apr_24h_change,
            "apr_7d_ago": base_apr - apr_7d_change,
            "apr_30d_ago": base_apr - apr_30d_change,
            "apr_change_24h": apr_24h_change,
            "apr_change_7d": apr_7d_change,
            "apr_change_30d": apr_30d_change,
            "tvl_change_24h": tvl_24h_change * 100,  # Convert to percentage
            "tvl_change_7d": tvl_7d_change * 100,
            "tvl_change_30d": tvl_30d_change * 100,
            "volatility": volatility,
            "prediction_score": prediction_score,
            "fee": 0.003 if dex != "Orca" else 0.004,
            "version": "v4" if dex == "Raydium" else "Whirlpool" if dex == "Orca" else "v1"
        }
        
        data.append(pool)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to file for future use
    try:
        with open("sample_pool_data.json", "w") as f:
            json.dump(data, f)
    except Exception as e:
        st.warning(f"Error saving sample data: {e}")
    
    return df

def fetch_live_data():
    """Fetch live data from APIs and blockchain"""
    with st.spinner("Fetching pool data from APIs and blockchain..."):
        # Initialize with API keys if available
        api_keys = {
            "raydium": os.getenv("RAYDIUM_API_KEY", ""),
            "jupiter": os.getenv("JUPITER_API_KEY", "")
        }
        
        rpc_endpoint = os.getenv("SOLANA_RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
        
        # In a production app, we'd use the retriever for real data
        retriever = PoolRetriever(api_keys, rpc_endpoint)
        
        try:
            # Get all pools
            pools = retriever.get_all_pools()
            
            # Convert to DataFrame
            pool_dicts = [p.to_dict() for p in pools]
            df = pd.DataFrame(pool_dicts)
            
            return df
        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            st.error("Falling back to sample data")
            return load_sample_data()

def load_data(use_live_data=False):
    """Load pool data from API or sample data"""
    if use_live_data:
        return fetch_live_data()
    else:
        return load_sample_data()

def main():
    """Main Streamlit app"""
    # Header Section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🌊 Solana Liquidity Pool Explorer")
        st.markdown("""
        Discover, analyze, and track thousands of liquidity pools across Solana's DeFi ecosystem, 
        with advanced filtering and ML-powered predictions.
        """)
    
    with col2:
        st.image("https://solana.com/_next/static/media/logotype.e4df684f.svg", width=200)
    
    # Show options for data source
    use_live_data = st.checkbox("Use live data (API/blockchain connection required)", value=False)
    data_source = "Live APIs & Blockchain" if use_live_data else "Sample Data"
    
    st.info(f"📊 Data Source: {data_source} - We can retrieve data for 3000+ pools across all major Solana DEXes")
    
    if st.button("Load Pool Data"):
        # Load data
        pool_data = load_data(use_live_data)
        
        # Store in session state
        st.session_state['pool_data'] = pool_data
        st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if we have data loaded
    if 'pool_data' not in st.session_state:
        st.warning("Please click 'Load Pool Data' to start exploring pools")
        
        # Show instructions
        st.subheader("How to use this explorer")
        st.markdown("""
        1. Click the **Load Pool Data** button to fetch information about Solana liquidity pools
        2. Use the **filters** in the sidebar to narrow down pools based on your criteria
        3. View detailed analysis of specific pools
        4. Explore trends and predictions for future performance
        
        This system can retrieve data for 3000+ pools across all major Solana DEXes, including:
        - **Raydium** pools (v3 and v4)
        - **Orca** Whirlpools
        - **Jupiter** aggregated pools
        - **Meteora** concentrated liquidity pools
        - **Saber** stable pools
        - And many more!
        
        The advanced filtering system allows you to filter by:
        - Liquidity (TVL) range
        - APR range
        - Volume requirements
        - Specific DEXes
        - Pool categories (Meme, DeFi, Major, etc.)
        - Token inclusion
        - Trend direction (increasing/decreasing metrics)
        - Prediction score thresholds
        """)
        
        # Create tabs for explanation content
        tab1, tab2, tab3 = st.tabs(["Data Sources", "Prediction Technology", "API Documentation"])
        
        with tab1:
            st.subheader("Data Sources")
            st.markdown("""
            Our system combines multiple data sources to provide comprehensive pool information:
            
            1. **Direct DEX APIs**
               - Raydium official API
               - Orca Whirlpools API
               - Jupiter aggregator API
               
            2. **Solana Blockchain**
               - Direct RPC node queries
               - Program account scanning
               - Transaction history analysis
               
            3. **Pool Account Parsing**
               - Custom binary data parsers for each DEX format
               - Specialized account structure handling
               
            This multi-source approach allows us to discover pools that aren't listed in official interfaces and
            provide more comprehensive data than any single source.
            """)
        
        with tab2:
            st.subheader("Prediction Technology")
            st.markdown("""
            Our prediction system combines multiple machine learning approaches:
            
            1. **Gradient Boosting Models (XGBoost/LightGBM)**
               - Non-linear relationship modeling
               - Feature importance ranking
               - Robust to outliers
            
            2. **LSTM Neural Networks**
               - Time-series prediction for APR and TVL changes
               - Pattern recognition in historical data
               - Sequence modeling for trend prediction
            
            3. **Reinforcement Learning**
               - Adaptive strategy optimization
               - Changing market condition handling
               - Continuous learning from new data
            
            4. **Statistical Methods**
               - ARIMA/GARCH for volatility forecasting
               - Bayesian inference for probability distributions
            
            The system combines these approaches through ensemble methods, with dynamic weight adjustment
            based on market conditions and model performance.
            """)
        
        with tab3:
            st.subheader("API Documentation")
            st.markdown("""
            ### API Endpoints
            
            Our system provides RESTful API endpoints for accessing pool data:
            
            #### GET /api/v1/pools
            
            Retrieve a list of pools with optional filtering.
            
            **Query Parameters:**
            - `dex` (string): Filter by DEX name
            - `category` (string): Filter by pool category
            - `min_tvl` (number): Minimum TVL threshold
            - `min_apr` (number): Minimum APR threshold
            - `token` (string): Filter pools containing this token
            - `limit` (integer): Maximum number of results (default: 100)
            - `sort_by` (string): Field to sort by (default: 'liquidity')
            - `sort_dir` (string): 'asc' or 'desc' (default: 'desc')
            
            **Example:** `GET /api/v1/pools?dex=Raydium&min_tvl=1000000&min_apr=10`
            
            #### GET /api/v1/pools/{pool_id}
            
            Retrieve detailed information about a specific pool.
            
            **Path Parameters:**
            - `pool_id` (string): The unique identifier of the pool
            
            **Example:** `GET /api/v1/pools/58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2`
            
            #### GET /api/v1/pools/{pool_id}/history
            
            Retrieve historical data for a specific pool.
            
            **Path Parameters:**
            - `pool_id` (string): The unique identifier of the pool
            
            **Query Parameters:**
            - `days` (integer): Number of days of history to retrieve (default: 30)
            - `interval` (string): Time interval ('hour', 'day', 'week') (default: 'day')
            
            **Example:** `GET /api/v1/pools/58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2/history?days=60&interval=day`
            
            #### GET /api/v1/predictions
            
            Retrieve ML-based predictions for pools.
            
            **Query Parameters:**
            - `min_score` (number): Minimum prediction score (0-100)
            - `category` (string): Filter by pool category
            - `limit` (integer): Maximum number of results (default: 20)
            
            **Example:** `GET /api/v1/predictions?min_score=80&category=Meme`
            
            ### Authentication
            
            All API requests require an API key passed in the `X-API-Key` header.
            
            ### Rate Limiting
            
            Requests are limited to 100 per minute per API key.
            
            ### Error Handling
            
            The API returns standard HTTP status codes:
            - 200: Success
            - 400: Bad request (invalid parameters)
            - 401: Unauthorized (invalid API key)
            - 404: Resource not found
            - 429: Rate limit exceeded
            - 500: Server error
            
            Error responses include a JSON body with error details:
            ```json
            {
              "error": "Error message",
              "code": "ERROR_CODE",
              "details": {}
            }
            ```
            """)
        
        st.stop()
    
    # If we have data, proceed with the analysis UI
    pool_data = st.session_state['pool_data']
    last_update = st.session_state.get('last_update', 'N/A')
    
    # Main Interface
    # Create tabs for different views
    tabs = st.tabs([
        "📊 Overview", 
        "🔍 Pool Explorer", 
        "📈 Trends & Predictions", 
        "📱 Mobile View"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        st.subheader("Market Overview")
        st.write(f"Last updated: {last_update}")
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_pools = len(pool_data)
        total_tvl = pool_data['liquidity'].sum()
        total_volume = pool_data['volume_24h'].sum()
        avg_apr = pool_data['apr'].mean()
        
        with col1:
            st.metric("Total Pools", f"{total_pools}")
            
        with col2:
            st.metric("Total TVL", format_currency(total_tvl))
            
        with col3:
            st.metric("24h Volume", format_currency(total_volume))
            
        with col4:
            st.metric("Avg APR", format_percentage(avg_apr))
            
        with col5:
            # Count of pools with high prediction scores
            high_potential = len(pool_data[pool_data['prediction_score'] >= 80])
            st.metric("High Potential Pools", f"{high_potential}")
        
        # DEX Breakdown
        st.subheader("DEX Breakdown")
        
        dex_counts = pool_data['dex'].value_counts().reset_index()
        dex_counts.columns = ['DEX', 'Pool Count']
        
        dex_liquidity = pool_data.groupby('dex')['liquidity'].sum().reset_index()
        dex_liquidity.columns = ['DEX', 'Total Liquidity']
        
        dex_volume = pool_data.groupby('dex')['volume_24h'].sum().reset_index()
        dex_volume.columns = ['DEX', 'Total Volume']
        
        dex_apr = pool_data.groupby('dex')['apr'].mean().reset_index()
        dex_apr.columns = ['DEX', 'Average APR']
        
        # Merge all DEX metrics
        dex_metrics = dex_counts.merge(dex_liquidity, on='DEX')
        dex_metrics = dex_metrics.merge(dex_volume, on='DEX')
        dex_metrics = dex_metrics.merge(dex_apr, on='DEX')
        
        # Sort by pool count
        dex_metrics = dex_metrics.sort_values('Pool Count', ascending=False)
        
        # Format metrics
        dex_metrics['Total Liquidity'] = dex_metrics['Total Liquidity'].apply(format_currency)
        dex_metrics['Total Volume'] = dex_metrics['Total Volume'].apply(format_currency)
        dex_metrics['Average APR'] = dex_metrics['Average APR'].apply(format_percentage)
        
        # Display as table
        st.dataframe(dex_metrics, use_container_width=True)
        
        # Category Breakdown
        st.subheader("Pool Categories")
        
        category_counts = pool_data['category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Pool Count']
        
        # Plot as bar chart
        fig = px.bar(
            category_counts, 
            x='Category', 
            y='Pool Count',
            color='Category',
            title="Pool Count by Category",
            text='Pool Count'
        )
        fig.update_layout(showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # APR Distribution
        st.subheader("APR Distribution")
        
        fig = px.histogram(
            pool_data,
            x='apr',
            nbins=20,
            title="APR Distribution",
            labels={'apr': 'APR (%)'},
            opacity=0.7
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top Pools Tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Pools by TVL")
            top_tvl = pool_data.sort_values('liquidity', ascending=False).head(5)
            top_tvl_display = pd.DataFrame({
                'Pool': top_tvl['name'],
                'DEX': top_tvl['dex'],
                'TVL': top_tvl['liquidity'].apply(format_currency),
                'APR': top_tvl['apr'].apply(format_percentage)
            })
            st.table(top_tvl_display)
        
        with col2:
            st.subheader("Top Pools by APR")
            # Filter out pools with very low liquidity for APR ranking
            min_liquidity = 1_000_000  # $1M minimum
            high_apr = pool_data[pool_data['liquidity'] >= min_liquidity].sort_values('apr', ascending=False).head(5)
            high_apr_display = pd.DataFrame({
                'Pool': high_apr['name'],
                'DEX': high_apr['dex'],
                'APR': high_apr['apr'].apply(format_percentage),
                'TVL': high_apr['liquidity'].apply(format_currency)
            })
            st.table(high_apr_display)
        
        # Category Performance Comparison
        st.subheader("Category Performance")
        
        # Calculate metrics by category
        category_metrics = pool_data.groupby('category').agg({
            'liquidity': 'sum',
            'volume_24h': 'sum',
            'apr': 'mean',
            'prediction_score': 'mean'
        }).reset_index()
        
        # Create a radar chart for category comparison
        categories = category_metrics['category'].tolist()
        
        # Normalize metrics for radar chart
        normalized_metrics = category_metrics.copy()
        for col in ['liquidity', 'volume_24h', 'apr', 'prediction_score']:
            max_val = normalized_metrics[col].max()
            if max_val > 0:
                normalized_metrics[col] = normalized_metrics[col] / max_val
        
        # Create radar chart
        fig = go.Figure()
        
        for i, category in enumerate(categories):
            row = normalized_metrics[normalized_metrics['category'] == category].iloc[0]
            
            fig.add_trace(go.Scatterpolar(
                r=[row['liquidity'], row['volume_24h'], row['apr'], row['prediction_score']],
                theta=['TVL', 'Volume', 'APR', 'Prediction Score'],
                fill='toself',
                name=category
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Category Performance (Normalized)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw metrics for reference
        with st.expander("View Raw Category Metrics"):
            display_metrics = category_metrics.copy()
            display_metrics['liquidity'] = display_metrics['liquidity'].apply(format_currency)
            display_metrics['volume_24h'] = display_metrics['volume_24h'].apply(format_currency)
            display_metrics['apr'] = display_metrics['apr'].apply(format_percentage)
            display_metrics['prediction_score'] = display_metrics['prediction_score'].apply(lambda x: f"{x:.1f}/100")
            
            st.dataframe(display_metrics, use_container_width=True)
    
    # Tab 2: Pool Explorer with Advanced Filtering
    with tabs[1]:
        st.subheader("Pool Explorer")
        st.write("Use the filters in the sidebar to find pools matching your criteria.")
        
        # Sidebar filters
        st.sidebar.title("Advanced Filters")
        
        # DEX filter
        dexes = sorted(pool_data['dex'].unique().tolist())
        selected_dexes = st.sidebar.multiselect("DEX", dexes, default=dexes)
        
        # Category filter
        categories = sorted(pool_data['category'].unique().tolist())
        selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
        
        # Token filter
        all_tokens = set()
        for token in pool_data['token1_symbol'].unique():
            all_tokens.add(token)
        for token in pool_data['token2_symbol'].unique():
            all_tokens.add(token)
        all_tokens = sorted(list(all_tokens))
        
        selected_token = st.sidebar.selectbox("Contains Token", ["All"] + all_tokens)
        
        # TVL range filter
        min_tvl = float(pool_data['liquidity'].min())
        max_tvl = float(pool_data['liquidity'].max())
        
        tvl_range = st.sidebar.slider(
            "TVL Range",
            min_value=min_tvl,
            max_value=max_tvl,
            value=(min_tvl, max_tvl),
            format="%e"
        )
        
        # APR range filter
        min_apr = float(pool_data['apr'].min())
        max_apr = float(pool_data['apr'].max())
        
        apr_range = st.sidebar.slider(
            "APR Range (%)",
            min_value=min_apr,
            max_value=max_apr,
            value=(min_apr, max_apr),
            format="%f"
        )
        
        # Volume filter
        min_volume = st.sidebar.number_input(
            "Minimum 24h Volume ($)",
            min_value=0.0,
            value=0.0,
            step=100000.0
        )
        
        # Prediction score threshold
        min_prediction = st.sidebar.slider(
            "Minimum Prediction Score",
            min_value=0,
            max_value=100,
            value=0
        )
        
        # Trend filters
        trend_options = st.sidebar.expander("Trend Filters")
        
        with trend_options:
            # APR trend
            apr_trend = st.radio(
                "APR Trend (7d)",
                ["All", "Increasing", "Decreasing", "Stable"]
            )
            
            apr_threshold = st.slider(
                "APR Change Threshold (%)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )
            
            # TVL trend
            tvl_trend = st.radio(
                "TVL Trend (7d)",
                ["All", "Increasing", "Decreasing", "Stable"]
            )
            
            tvl_threshold = st.slider(
                "TVL Change Threshold (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1
            )
        
        # Apply filters
        filtered_data = pool_data.copy()
        
        # Basic filters
        if selected_dexes:
            filtered_data = filtered_data[filtered_data['dex'].isin(selected_dexes)]
            
        if selected_categories:
            filtered_data = filtered_data[filtered_data['category'].isin(selected_categories)]
            
        if selected_token != "All":
            token_mask = (
                (filtered_data['token1_symbol'] == selected_token) | 
                (filtered_data['token2_symbol'] == selected_token)
            )
            filtered_data = filtered_data[token_mask]
            
        # Range filters
        filtered_data = filtered_data[
            (filtered_data['liquidity'] >= tvl_range[0]) & 
            (filtered_data['liquidity'] <= tvl_range[1])
        ]
        
        filtered_data = filtered_data[
            (filtered_data['apr'] >= apr_range[0]) & 
            (filtered_data['apr'] <= apr_range[1])
        ]
        
        if min_volume > 0:
            filtered_data = filtered_data[filtered_data['volume_24h'] >= min_volume]
            
        if min_prediction > 0:
            filtered_data = filtered_data[filtered_data['prediction_score'] >= min_prediction]
            
        # Trend filters
        if apr_trend != "All" and 'apr_change_7d' in filtered_data.columns:
            if apr_trend == "Increasing":
                filtered_data = filtered_data[filtered_data['apr_change_7d'] >= apr_threshold]
            elif apr_trend == "Decreasing":
                filtered_data = filtered_data[filtered_data['apr_change_7d'] <= -apr_threshold]
            else:  # Stable
                filtered_data = filtered_data[
                    (filtered_data['apr_change_7d'] > -apr_threshold) & 
                    (filtered_data['apr_change_7d'] < apr_threshold)
                ]
                
        if tvl_trend != "All" and 'tvl_change_7d' in filtered_data.columns:
            if tvl_trend == "Increasing":
                filtered_data = filtered_data[filtered_data['tvl_change_7d'] >= tvl_threshold]
            elif tvl_trend == "Decreasing":
                filtered_data = filtered_data[filtered_data['tvl_change_7d'] <= -tvl_threshold]
            else:  # Stable
                filtered_data = filtered_data[
                    (filtered_data['tvl_change_7d'] > -tvl_threshold) & 
                    (filtered_data['tvl_change_7d'] < tvl_threshold)
                ]
        
        # Sort options
        sort_options = {
            "TVL (High to Low)": "liquidity",
            "TVL (Low to High)": "liquidity_asc",
            "APR (High to Low)": "apr",
            "APR (Low to High)": "apr_asc",
            "Volume (High to Low)": "volume_24h",
            "Volume (Low to High)": "volume_24h_asc",
            "Prediction Score (High to Low)": "prediction_score",
            "Prediction Score (Low to High)": "prediction_score_asc"
        }
        
        sort_by = st.selectbox("Sort by", list(sort_options.keys()))
        sort_key = sort_options[sort_by]
        
        # Apply sorting
        if sort_key.endswith("_asc"):
            base_key = sort_key.replace("_asc", "")
            filtered_data = filtered_data.sort_values(base_key, ascending=True)
        else:
            filtered_data = filtered_data.sort_values(sort_key, ascending=False)
        
        # Display filter summary
        st.write(f"Showing {len(filtered_data)} pools matching your criteria")
        
        # Display pools
        if not filtered_data.empty:
            # Prepare display data
            display_data = filtered_data.copy()
            
            # Create a clean display table
            display_cols = {
                'name': 'Pool',
                'dex': 'DEX',
                'category': 'Category',
                'liquidity': 'TVL',
                'volume_24h': '24h Volume',
                'apr': 'APR',
                'prediction_score': 'ML Score'
            }
            
            # Add trend columns if available
            if 'apr_change_7d' in display_data.columns:
                display_cols['apr_change_7d'] = '7d APR Change'
                
            if 'tvl_change_7d' in display_data.columns:
                display_cols['tvl_change_7d'] = '7d TVL Change'
            
            # Create display table
            table_data = display_data[list(display_cols.keys())].copy()
            table_data.columns = list(display_cols.values())
            
            # Format columns
            table_data['TVL'] = table_data['TVL'].apply(format_currency)
            table_data['24h Volume'] = table_data['24h Volume'].apply(format_currency)
            table_data['APR'] = table_data['APR'].apply(format_percentage)
            table_data['ML Score'] = table_data['ML Score'].apply(lambda x: f"{x:.1f}/100")
            
            if '7d APR Change' in table_data.columns:
                table_data['7d APR Change'] = table_data['7d APR Change'].apply(
                    lambda x: f"{get_trend_icon(x)} {format_percentage(abs(x))}"
                )
                
            if '7d TVL Change' in table_data.columns:
                table_data['7d TVL Change'] = table_data['7d TVL Change'].apply(
                    lambda x: f"{get_trend_icon(x)} {format_percentage(abs(x))}"
                )
            
            # Display the table
            st.dataframe(table_data, use_container_width=True)
            
            # Pool details section
            st.subheader("Pool Details")
            
            # Create a selection box with pool names
            pool_options = {}
            for i, row in display_data.iterrows():
                display_name = f"{row['name']} ({row['dex']})"
                pool_options[display_name] = row['id']
                
            selected_pool_name = st.selectbox("Select Pool", list(pool_options.keys()))
            selected_pool_id = pool_options[selected_pool_name]
            
            # Get the selected pool data
            pool = display_data[display_data['id'] == selected_pool_id].iloc[0]
            
            # Display detailed pool information
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("### Basic Information")
                st.write(f"**Pool ID:** {pool['id']}")
                st.write(f"**Name:** {pool['name']}")
                st.write(f"**DEX:** {pool['dex']}")
                st.write(f"**Category:** {pool['category']}")
                st.write(f"**Token 1:** {pool['token1_symbol']}")
                st.write(f"**Token 2:** {pool['token2_symbol']}")
                
                if 'version' in pool:
                    st.write(f"**Version:** {pool['version']}")
                    
                if 'fee' in pool:
                    st.write(f"**Fee:** {format_percentage(pool['fee'] * 100)}")
            
            with col2:
                st.write("### Performance Metrics")
                st.write(f"**TVL:** {format_currency(pool['liquidity'])}")
                st.write(f"**24h Volume:** {format_currency(pool['volume_24h'])}")
                st.write(f"**APR:** {format_percentage(pool['apr'])}")
                
                if 'volatility' in pool:
                    st.write(f"**Volatility:** {format_percentage(pool['volatility'] * 100)}")
                
                # Display 7d changes if available
                if 'apr_change_7d' in pool:
                    apr_change = pool['apr_change_7d']
                    change_str = f"{get_trend_icon(apr_change)} {format_percentage(abs(apr_change))}"
                    st.write(f"**7d APR Change:** {change_str}")
                
                if 'tvl_change_7d' in pool:
                    tvl_change = pool['tvl_change_7d']
                    change_str = f"{get_trend_icon(tvl_change)} {format_percentage(abs(tvl_change))}"
                    st.write(f"**7d TVL Change:** {change_str}")
                
                # Display prediction score
                score = pool['prediction_score']
                st.write(f"**ML Prediction Score:** {score:.1f}/100")
                
                # Add a visual indicator for prediction score
                if score >= 85:
                    st.success("High growth potential")
                elif score >= 70:
                    st.info("Moderate growth potential")
                else:
                    st.warning("Limited growth potential")
            
            # Historical data visualization (if available)
            if ('apr_change_7d' in pool and 'apr_change_24h' in pool and 
                'apr_30d_ago' in pool and 'apr_7d_ago' in pool and 'apr_24h_ago' in pool):
                
                st.write("### Historical APR")
                
                # Create historical APR data
                apr_data = {
                    'Period': ['Current', '24h Ago', '7d Ago', '30d Ago'],
                    'APR': [
                        pool['apr'],
                        pool['apr_24h_ago'] if 'apr_24h_ago' in pool else pool['apr'] - pool['apr_change_24h'],
                        pool['apr_7d_ago'] if 'apr_7d_ago' in pool else pool['apr'] - pool['apr_change_7d'],
                        pool['apr_30d_ago'] if 'apr_30d_ago' in pool else pool['apr'] - pool['apr_change_30d']
                    ]
                }
                
                # Create bar chart
                fig = px.bar(
                    apr_data,
                    x='Period',
                    y='APR',
                    title="Historical APR (%)",
                    labels={'APR': 'APR (%)'},
                    text_auto='.2f'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction analysis
            st.write("### Prediction Analysis")
            
            # Create a mock prediction chart showing the predicted APR range
            current_apr = pool['apr']
            prediction_score = pool['prediction_score']
            
            # Create a range based on the prediction score
            # Higher scores = narrower confidence intervals
            confidence_factor = (100 - prediction_score) / 100 * 0.5 + 0.1  # Range from 0.1 to 0.6
            
            lower_bound = current_apr * (1 - confidence_factor)
            upper_bound = current_apr * (1 + confidence_factor)
            
            # Create prediction data
            today = datetime.now()
            dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
            
            # Create a projected trend based on recent changes and prediction score
            trend_factor = 0
            
            if 'apr_change_7d' in pool:
                # Use recent trend, weighted by prediction score
                recent_change = pool['apr_change_7d'] / 7  # Daily change
                trend_factor = recent_change * (prediction_score / 100)
            
            # Generate future APR projections
            projected_apr = [current_apr]
            lower_bound_apr = [lower_bound]
            upper_bound_apr = [upper_bound]
            
            for i in range(1, 30):
                next_apr = projected_apr[-1] + trend_factor
                projected_apr.append(next_apr)
                
                # Widen confidence interval over time
                day_factor = (i / 30) * 0.5 + 1  # Ranges from 1 to 1.5
                lower_bound_apr.append(next_apr * (1 - confidence_factor * day_factor))
                upper_bound_apr.append(next_apr * (1 + confidence_factor * day_factor))
            
            # Create prediction dataframe
            prediction_df = pd.DataFrame({
                'Date': dates,
                'Projected APR': projected_apr,
                'Lower Bound': lower_bound_apr,
                'Upper Bound': upper_bound_apr
            })
            
            # Plot prediction
            fig = go.Figure()
            
            # Add projected APR line
            fig.add_trace(go.Scatter(
                x=prediction_df['Date'],
                y=prediction_df['Projected APR'],
                name='Projected APR',
                line=dict(color='blue', width=2)
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=prediction_df['Date'],
                y=prediction_df['Upper Bound'],
                fill=None,
                mode='lines',
                line=dict(color='rgba(0,0,255,0)'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=prediction_df['Date'],
                y=prediction_df['Lower Bound'],
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(0,0,255,0)'),
                fillcolor='rgba(0,0,255,0.2)',
                name='Confidence Interval'
            ))
            
            # Add current APR marker
            fig.add_trace(go.Scatter(
                x=[dates[0]],
                y=[current_apr],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Current APR'
            ))
            
            # Update layout
            fig.update_layout(
                title="30-Day APR Projection",
                xaxis_title="Date",
                yaxis_title="APR (%)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add prediction commentary
            st.write("### Analysis")
            
            if prediction_score >= 85:
                st.success(f"""
                **High Growth Potential (Score: {prediction_score:.1f}/100)**
                
                This pool shows strong indicators for APR and/or TVL growth in the coming weeks.
                Key factors contributing to this prediction:
                
                - {"Strong positive APR trend" if 'apr_change_7d' in pool and pool['apr_change_7d'] > 0 else "Stable APR with growth potential"}
                - {"Increasing liquidity" if 'tvl_change_7d' in pool and pool['tvl_change_7d'] > 0 else "Substantial liquidity base"}
                - {"High trading volume relative to TVL" if pool['volume_24h'] / pool['liquidity'] > 0.1 else "Consistent trading activity"}
                - {f"Popular {pool['category']} category with current market momentum" if pool['category'] in ['Meme', 'DeFi'] else "Strong fundamentals in established category"}
                
                Consider this pool for potential liquidity provision opportunities.
                """)
            elif prediction_score >= 70:
                st.info(f"""
                **Moderate Growth Potential (Score: {prediction_score:.1f}/100)**
                
                This pool shows moderate potential for stable or increasing APR in the near term.
                Key factors to consider:
                
                - {"Slightly positive APR trend" if 'apr_change_7d' in pool and pool['apr_change_7d'] > 0 else "Relatively stable APR"}
                - {"Modest liquidity growth" if 'tvl_change_7d' in pool and pool['tvl_change_7d'] > 0 else "Consistent liquidity level"}
                - {"Reasonable trading volume" if pool['volume_24h'] / pool['liquidity'] > 0.05 else "Adequate market activity"}
                - {"Established position in its category" if pool['liquidity'] > 5_000_000 else "Growing presence in the market"}
                
                Monitor this pool for potential opportunities as market conditions evolve.
                """)
            else:
                st.warning(f"""
                **Limited Growth Potential (Score: {prediction_score:.1f}/100)**
                
                This pool shows limited indicators for significant APR or TVL growth in the near term.
                Key considerations:
                
                - {"Declining APR trend" if 'apr_change_7d' in pool and pool['apr_change_7d'] < 0 else "Inconsistent APR performance"}
                - {"Decreasing liquidity" if 'tvl_change_7d' in pool and pool['tvl_change_7d'] < 0 else "Stagnant liquidity levels"}
                - {"Low trading volume relative to TVL" if pool['volume_24h'] / pool['liquidity'] < 0.03 else "Volatile trading patterns"}
                - {"Challenging market conditions for this category" if pool['category'] not in ['Major', 'Stablecoin'] else "Limited growth drivers despite stable foundation"}
                
                Consider other pools with higher growth potential for liquidity provision.
                """)
            
            # Display similar pools
            st.write("### Similar Pools")
            
            # Find pools in the same category and with similar characteristics
            same_category = display_data[
                (display_data['category'] == pool['category']) & 
                (display_data['id'] != pool['id'])
            ]
            
            if len(same_category) >= 3:
                # Calculate a simple similarity score based on APR and TVL
                similarities = []
                
                for _, similar_pool in same_category.iterrows():
                    # Calculate TVL difference (normalized)
                    tvl_diff = abs(similar_pool['liquidity'] - pool['liquidity']) / max(pool['liquidity'], 1)
                    
                    # Calculate APR difference (normalized)
                    apr_diff = abs(similar_pool['apr'] - pool['apr']) / max(pool['apr'], 1)
                    
                    # Weighted similarity score (lower is more similar)
                    similarity = 0.7 * tvl_diff + 0.3 * apr_diff
                    
                    similarities.append((similar_pool, similarity))
                
                # Sort by similarity and take top 3
                similarities.sort(key=lambda x: x[1])
                top_similar = [item[0] for item in similarities[:3]]
                
                # Display similar pools
                similar_data = []
                
                for similar in top_similar:
                    similar_data.append({
                        'Pool': similar['name'],
                        'DEX': similar['dex'],
                        'TVL': format_currency(similar['liquidity']),
                        'APR': format_percentage(similar['apr']),
                        'Score': f"{similar['prediction_score']:.1f}/100"
                    })
                
                similar_df = pd.DataFrame(similar_data)
                st.table(similar_df)
            else:
                st.write("No similar pools found in this category.")
    
    # Tab 3: Trends & Predictions
    with tabs[2]:
        st.subheader("Trends & Prediction Analysis")
        st.write("Explore pool performance trends and ML-based predictions.")
        
        # Category-based trend analysis
        st.write("### Category Trends")
        
        # Check if we have trend data
        has_trend_data = 'apr_change_7d' in pool_data.columns and 'tvl_change_7d' in pool_data.columns
        
        if has_trend_data:
            # Calculate average changes by category
            category_trends = pool_data.groupby('category').agg({
                'apr_change_7d': 'mean',
                'tvl_change_7d': 'mean',
                'apr_change_30d': 'mean',
                'tvl_change_30d': 'mean',
                'prediction_score': 'mean',
                'id': 'count'
            }).reset_index()
            
            category_trends.columns = [
                'Category', 'APR 7d Change', 'TVL 7d Change', 
                'APR 30d Change', 'TVL 30d Change', 
                'Avg Prediction', 'Pool Count'
            ]
            
            # Sort by prediction score
            category_trends = category_trends.sort_values('Avg Prediction', ascending=False)
            
            # Format for display
            display_trends = category_trends.copy()
            display_trends['APR 7d Change'] = display_trends['APR 7d Change'].apply(
                lambda x: f"{get_trend_icon(x)} {format_percentage(abs(x))}"
            )
            display_trends['TVL 7d Change'] = display_trends['TVL 7d Change'].apply(
                lambda x: f"{get_trend_icon(x)} {format_percentage(abs(x))}"
            )
            display_trends['APR 30d Change'] = display_trends['APR 30d Change'].apply(
                lambda x: f"{get_trend_icon(x)} {format_percentage(abs(x))}"
            )
            display_trends['TVL 30d Change'] = display_trends['TVL 30d Change'].apply(
                lambda x: f"{get_trend_icon(x)} {format_percentage(abs(x))}"
            )
            display_trends['Avg Prediction'] = display_trends['Avg Prediction'].apply(
                lambda x: f"{x:.1f}/100"
            )
            
            st.table(display_trends)
            
            # Visualize trends
            st.write("### 7-Day APR Change by Category")
            
            fig = px.bar(
                category_trends,
                x='Category',
                y='APR 7d Change',
                color='APR 7d Change',
                color_continuous_scale='RdYlGn',
                title="7-Day APR Change by Category (%)",
                text_auto='.2f'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("### 7-Day TVL Change by Category")
            
            fig = px.bar(
                category_trends,
                x='Category',
                y='TVL 7d Change',
                color='TVL 7d Change',
                color_continuous_scale='RdYlGn',
                title="7-Day TVL Change by Category (%)",
                text_auto='.2f'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Trend data not available in the sample dataset.")
        
        # Top prediction scores
        st.write("### Top Prediction Scores")
        
        # Get pools with high prediction scores
        high_potential = pool_data.sort_values('prediction_score', ascending=False).head(10)
        
        # Create display table
        potential_display = pd.DataFrame({
            'Pool': high_potential['name'],
            'DEX': high_potential['dex'],
            'Category': high_potential['category'],
            'TVL': high_potential['liquidity'].apply(format_currency),
            'APR': high_potential['apr'].apply(format_percentage),
            'Prediction Score': high_potential['prediction_score'].apply(lambda x: f"{x:.1f}/100")
        })
        
        st.table(potential_display)
        
        # Prediction score distribution
        st.write("### Prediction Score Distribution")
        
        fig = px.histogram(
            pool_data,
            x='prediction_score',
            nbins=20,
            title="Distribution of Prediction Scores",
            labels={'prediction_score': 'Prediction Score'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # DEX performance trend
        st.write("### DEX Performance Trends")
        
        if has_trend_data:
            # Calculate average changes by DEX
            dex_trends = pool_data.groupby('dex').agg({
                'apr_change_7d': 'mean',
                'tvl_change_7d': 'mean',
                'prediction_score': 'mean',
                'id': 'count'
            }).reset_index()
            
            dex_trends.columns = [
                'DEX', 'APR 7d Change', 'TVL 7d Change', 
                'Avg Prediction', 'Pool Count'
            ]
            
            # Create a bubble chart of DEX trends
            fig = px.scatter(
                dex_trends,
                x='APR 7d Change',
                y='TVL 7d Change',
                size='Pool Count',
                color='Avg Prediction',
                hover_name='DEX',
                size_max=60,
                color_continuous_scale='viridis',
                title="DEX Performance Trends"
            )
            
            # Add horizontal and vertical lines at 0
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            # Update layout
            fig.update_layout(
                xaxis_title="7-Day APR Change (%)",
                yaxis_title="7-Day TVL Change (%)",
                coloraxis_colorbar_title="Avg Prediction"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Trend data not available in the sample dataset.")
        
        # ML prediction methodology explanation
        with st.expander("Prediction Methodology"):
            st.write("""
            ### Machine Learning Prediction Methodology
            
            Our prediction system uses a combination of multiple machine learning approaches:
            
            #### Data Features
            
            The models analyze dozens of features, including:
            
            - **Historical metrics:** APR, TVL, and volume trends over various time periods
            - **Volatility measures:** How stable the pool's metrics have been
            - **Token correlation:** How token prices move in relation to each other
            - **On-chain activity:** Transaction patterns and liquidity provider behavior
            - **Market sentiment:** Derived from social media and developer activity
            - **DEX-specific factors:** Protocol upgrades, incentive programs, etc.
            
            #### Model Architecture
            
            We use a multi-model approach:
            
            1. **Gradient Boosting Models (XGBoost/LightGBM)**
               - Specialized for non-linear relationship modeling
               - Feature importance ranking to understand key drivers
            
            2. **LSTM Neural Networks**
               - Time-series prediction for APR and TVL changes
               - Pattern recognition in historical data
            
            3. **Reinforcement Learning**
               - Adaptive strategy optimization
               - Continuous learning from new data
            
            4. **Statistical Methods**
               - ARIMA/GARCH for volatility forecasting
               - Bayesian inference for probability distributions
            
            #### Ensemble Method
            
            The final prediction score combines outputs from all models using a weighted ensemble approach.
            Weights are dynamically adjusted based on recent model performance.
            
            #### Performance Validation
            
            Our models are continuously validated using:
            
            - Out-of-sample testing
            - Backtesting against historical data
            - Regular retraining with the latest data
            
            Prediction scores range from 0 to 100, with higher scores indicating greater confidence
            in positive APR and/or TVL growth over the next 30 days.
            """)
    
    # Tab 4: Mobile View (Simplified)
    with tabs[3]:
        st.subheader("Mobile View")
        st.write("A simplified view optimized for mobile devices")
        
        # Simple filters
        st.write("### Quick Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            quick_dex = st.selectbox("DEX", ["All"] + dexes)
            
        with col2:
            quick_category = st.selectbox("Category", ["All"] + categories)
        
        # Apply simple filters
        mobile_filtered = pool_data.copy()
        
        if quick_dex != "All":
            mobile_filtered = mobile_filtered[mobile_filtered['dex'] == quick_dex]
            
        if quick_category != "All":
            mobile_filtered = mobile_filtered[mobile_filtered['category'] == quick_category]
        
        # Sort options
        mobile_sort = st.selectbox(
            "Sort by",
            ["TVL (High to Low)", "APR (High to Low)", "Prediction Score (High to Low)"]
        )
        
        # Apply sort
        if mobile_sort == "TVL (High to Low)":
            mobile_filtered = mobile_filtered.sort_values('liquidity', ascending=False)
        elif mobile_sort == "APR (High to Low)":
            mobile_filtered = mobile_filtered.sort_values('apr', ascending=False)
        else:  # Prediction Score
            mobile_filtered = mobile_filtered.sort_values('prediction_score', ascending=False)
        
        # Take top 20 for mobile view
        mobile_top = mobile_filtered.head(20)
        
        # Display in card format
        st.write(f"### Top Pools ({len(mobile_top)})")
        
        for _, pool in mobile_top.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <h3>{pool['name']} ({pool['dex']})</h3>
                    <p><span class="metric-label">TVL:</span> <span class="metric-value">{format_currency(pool['liquidity'])}</span></p>
                    <p><span class="metric-label">APR:</span> <span class="metric-value">{format_percentage(pool['apr'])}</span></p>
                    <p><span class="metric-label">Category:</span> {get_category_badge(pool['category'])}</p>
                    <p><span class="metric-label">Prediction:</span> <span class="metric-value" style="color:{get_prediction_color(pool['prediction_score'])}">{pool['prediction_score']:.1f}/100</span></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Quick tips
        st.write("### Quick Tips")
        
        st.info("""
        **Finding High-Potential Pools:**
        - Look for pools with prediction scores above 80
        - Check for positive 7-day APR and TVL trends
        - Consider the pool's category and market conditions
        - Higher APR often comes with higher volatility
        
        **Using This App:**
        - Use the desktop view for advanced filtering
        - Check the "Trends & Predictions" tab for market insights
        - Select a pool to see detailed performance data
        - View similar pools for alternative investment options
        """)

# Run the app
if __name__ == "__main__":
    main()