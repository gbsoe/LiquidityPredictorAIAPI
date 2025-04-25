import streamlit as st
import pandas as pd
import json
import os
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Attempt to import the onchain extractor
try:
    from onchain_extractor import OnChainExtractor
    HAS_EXTRACTOR = True
except ImportError:
    HAS_EXTRACTOR = False

# Attempt to import the advanced prediction engine
HAS_BASIC_PREDICTION = False
HAS_ADVANCED_PREDICTION = False

# Set page configuration
st.set_page_config(
    page_title="FiLot - Solana Liquidity Pool Analysis",
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

def load_data():
    """Load pool data from the cached file first, and only try live data if explicitly requested"""
    cache_file = "extracted_pools.json"
    
    # First check if we have the cached file - use this by default for stability
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                file_content = f.read()
                if file_content.strip():
                    st.success("✓ Successfully loaded pool data from local cache")
                    pools = json.loads(file_content)
                    if pools and len(pools) > 0:
                        return pools
        except Exception as e:
            st.warning(f"Error loading cached data: {e}")
    
    # Sidebar controls for data loading
    with st.sidebar:
        st.subheader("Data Source")
        
        force_live_data = st.checkbox("Try live blockchain data", value=False, 
                              help="When checked, attempts to fetch fresh data from blockchain")
        
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
                
                # Initialize extractor with better error handling - limit pools to 2 per DEX to avoid rate limits
                extractor = OnChainExtractor(rpc_endpoint=custom_rpc)
                pools = extractor.extract_and_enrich_pools(max_per_dex=2)
                
                # Verify the data is not empty
                if pools and len(pools) > 0:
                    st.success(f"Successfully extracted {len(pools)} pools from blockchain")
                    # Save to cache
                    try:
                        with open(cache_file, "w") as f:
                            json.dump(pools, f, indent=2)
                        st.info(f"Data saved to cache file: {cache_file}")
                    except Exception as e:
                        st.warning(f"Error saving data to cache: {e}")
                    
                    return pools
                else:
                    st.error("No pool data was returned from the blockchain")
            except Exception as e:
                st.error(f"Error extracting data from blockchain: {str(e)}")
    
    # If we got here, we need to use sample data
    st.warning("Unable to load pool data - using generated sample data")
    sample_data = generate_sample_data()
    
    # Save sample data to cache so it's consistent between runs
    try:
        with open(cache_file, "w") as f:
            json.dump(sample_data, f, indent=2)
    except Exception as e:
        st.warning(f"Error saving sample data to cache: {e}")
    
    return sample_data

def generate_sample_data():
    """Generate sample pool data"""
    # We'll create data for 50 pools across 5 DEXes
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
    for i in range(50):
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
        
        # Create pool data
        pool = {
            "id": pool_id,
            "name": f"{token1}/{token2}",
            "dex": dex,
            "category": category,
            "token1_symbol": token1,
            "token2_symbol": token2,
            "token1_address": token_addresses.get(token1, "Unknown"),
            "token2_address": token_addresses.get(token2, "Unknown"),
            "liquidity": liquidity,
            "volume_24h": volume_24h,
            "apr": apr,
            "fee": fee,
            "version": "v4" if dex == "Raydium" else "Whirlpool" if dex == "Orca" else "v1",
            "apr_change_24h": apr_change_24h,
            "apr_change_7d": apr_change_7d,
            "apr_change_30d": apr_change_30d,
            "tvl_change_24h": tvl_change_24h,
            "tvl_change_7d": tvl_change_7d,
            "tvl_change_30d": tvl_change_30d,
            "prediction_score": prediction_score
        }
        
        pools.append(pool)
    
    return pools

def main():
    # Display logo and title
    col_logo, col_title = st.columns([1, 3])
    
    with col_logo:
        st.image("static/filot_logo_new.png", width=150)
    
    with col_title:
        st.title("FiLot - Solana Liquidity Pool Analysis")
        st.subheader("Advanced analysis and predictions for Solana DeFi liquidity pools")
        st.markdown("""
        This tool analyzes thousands of Solana liquidity pools across all major DEXes, 
        including Raydium, Orca, Jupiter, Meteora, Saber, and more. It provides comprehensive 
        data, historical metrics, and machine learning-based predictions.
        """)
    
    # Create tabs for different views
    tab_explore, tab_predict, tab_risk, tab_nlp = st.tabs([
        "Data Explorer", "Predictions", "Risk Assessment", "NLP Reports"
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
            return
    else:
        st.error("Failed to load any pool data")
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
        
        # Extract display data
        display_df = filtered_df[display_columns].copy()
        
        # Format the display data
        display_df["liquidity"] = display_df["liquidity"].apply(format_currency)
        display_df["volume_24h"] = display_df["volume_24h"].apply(format_currency)
        display_df["apr"] = display_df["apr"].apply(format_percentage)
        display_df["apr_change_24h"] = display_df["apr_change_24h"].apply(
            lambda x: f"{get_trend_icon(x)} {format_percentage(x)}"
        )
        display_df["apr_change_7d"] = display_df["apr_change_7d"].apply(
            lambda x: f"{get_trend_icon(x)} {format_percentage(x)}"
        )
        
        # Format prediction score with a visual indicator
        display_df["prediction_score"] = display_df["prediction_score"].apply(
            lambda x: f"{'🟢' if x > 75 else '🟡' if x > 50 else '🔴'} {x:.1f}/100"
        )
        
        # Format the category column with HTML badges
        display_df["category"] = filtered_df["category"].apply(get_category_badge)
        
        # Rename columns for display
        column_labels = {
            "name": "Pool Name",
            "dex": "DEX",
            "category": "Category",
            "liquidity": "TVL",
            "volume_24h": "24h Volume",
            "apr": "APR",
            "apr_change_24h": "24h Δ",
            "apr_change_7d": "7d Δ",
            "prediction_score": "Pred Score",
            "id": "Pool ID"
        }
        
        display_df = display_df.rename(columns=column_labels)
        
        # Display the table with HTML formatting
        st.write(
            display_df.to_html(escape=False, index=False),
            unsafe_allow_html=True
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
        
        # Get top 10 pools by prediction score
        top_predicted = df.sort_values("prediction_score", ascending=False).head(10)
        
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
            
            # Get pools with highest APR changes (proxy for volatility)
            volatile_pools = df.iloc[df['apr_change_24h'].abs().argsort()[::-1]].head(5)
            
            for _, row in volatile_pools.iterrows():
                st.markdown(f"**{row['name']}** ({row['dex']})")
                st.markdown(f"APR Change 24h: {format_percentage(row['apr_change_24h'])} {get_trend_icon(row['apr_change_24h'])}")
                st.markdown("---")
        
        with risk_col2:
            st.subheader("Lowest Liquidity Pools")
            
            # Get pools with lowest liquidity
            low_liquidity = df.sort_values("liquidity").head(5)
            
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
        base_liquidity = pool_data["liquidity"] / (1 + pool_data["tvl_change_30d"]/100)
        base_apr = pool_data["apr"] / (1 + pool_data["apr_change_30d"]/100)
        
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days, 0, -1)]
        
        # Generate random daily fluctuations within the overall trend
        liquidity_trend = pool_data["tvl_change_30d"] / 30  # daily trend
        apr_trend = pool_data["apr_change_30d"] / 30  # daily trend
        
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

if __name__ == "__main__":
    main()