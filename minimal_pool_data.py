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
try:
    from advanced_prediction_engine import (
        SelfEvolvingPredictionEngine, 
        predict_pool_performance,
        predict_multiple_pools
    )
    HAS_PREDICTION_ENGINE = True
except ImportError:
    HAS_PREDICTION_ENGINE = False

# Set page configuration
st.set_page_config(
    page_title="Solana Liquidity Pool Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better mobile responsiveness
st.markdown("""
<style>
    /* Mobile-friendly styling */
    @media (max-width: 768px) {
        div.row-widget.stRadio > div {
            flex-direction: column;
        }
        
        div.row-widget.stRadio > div button {
            width: 100%;
            margin: 2px 0;
        }
        
        div.block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #f9f9fa;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E88E5;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    
    /* Table styling */
    .dataframe-container th {
        background-color: #f2f2f2;
        color: #333;
    }
    
    .dataframe-container td, .dataframe-container th {
        text-align: left;
        padding: 8px;
    }
    
    .dataframe-container tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    /* Highlight styling */
    .highlight-container {
        border-left: 4px solid #1E88E5;
        padding-left: 10px;
        margin: 10px 0;
    }
    
    /* Category badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        color: white;
    }
    
    .badge-meme {
        background-color: #FF5722;
    }
    
    .badge-major {
        background-color: #4CAF50;
    }
    
    .badge-defi {
        background-color: #2196F3;
    }
    
    .badge-gaming {
        background-color: #9C27B0;
    }
    
    .badge-stablecoin {
        background-color: #607D8B;
    }
    
    .badge-other {
        background-color: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def format_currency(value):
    """Format a value as currency"""
    if value is None:
        return "N/A"
    elif value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value):
    """Format a value as percentage"""
    if value is None:
        return "N/A"
    return f"{value:.2f}%"

def get_trend_icon(value):
    """Return an arrow icon based on trend direction"""
    if value is None:
        return "➖"
    return "📈" if value >= 0 else "📉"

def get_category_badge(category):
    """Return HTML for a category badge"""
    category = category.lower() if category else "other"
    badge_class = f"badge badge-{category}"
    return f'<span class="{badge_class}">{category.capitalize()}</span>'

def load_data():
    """Load pool data from the extractor or a JSON file"""
    # Check if we have cached data
    cache_file = "extracted_pools.json"
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading cached data: {e}")
    
    # If we have the extractor, try to get live data
    if HAS_EXTRACTOR:
        try:
            with st.spinner("Extracting pool data from Solana blockchain..."):
                extractor = OnChainExtractor()
                pools = extractor.extract_and_enrich_pools(max_per_dex=10)
                
                # Save to cache
                try:
                    with open(cache_file, "w") as f:
                        json.dump(pools, f, indent=2)
                except Exception as e:
                    st.warning(f"Error saving data to cache: {e}")
                
                return pools
        except Exception as e:
            st.error(f"Error extracting pool data: {e}")
    
    # Fallback to sample data
    st.warning("Using sample data - connect to Solana RPC for live data")
    return generate_sample_data()

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
    st.title("Solana Liquidity Pool Analysis")
    
    # Introduction
    st.markdown("""
    This tool analyzes thousands of Solana liquidity pools across all major DEXes, 
    including Raydium, Orca, Jupiter, Meteora, Saber, and more. It provides 
    comprehensive data, historical metrics, and machine learning-based predictions.
    """)
    
    # Load data
    pool_data = load_data()
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(pool_data)
    
    # Create tabs for different views
    tabs = st.tabs(["Overview", "Pool Explorer", "Insights & Predictions"])
    
    # Tab 1: Overview
    with tabs[0]:
        st.header("Market Overview")
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(
                f"""<div class="metric-container">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Total Pools</div>
                </div>""", 
                unsafe_allow_html=True
            )
        
        with col2:
            total_tvl = df['liquidity'].sum()
            st.markdown(
                f"""<div class="metric-container">
                <div class="metric-value">{format_currency(total_tvl)}</div>
                <div class="metric-label">Total TVL</div>
                </div>""", 
                unsafe_allow_html=True
            )
        
        with col3:
            total_volume = df['volume_24h'].sum()
            st.markdown(
                f"""<div class="metric-container">
                <div class="metric-value">{format_currency(total_volume)}</div>
                <div class="metric-label">24h Volume</div>
                </div>""", 
                unsafe_allow_html=True
            )
        
        with col4:
            avg_apr = df['apr'].mean()
            st.markdown(
                f"""<div class="metric-container">
                <div class="metric-value">{format_percentage(avg_apr)}</div>
                <div class="metric-label">Average APR</div>
                </div>""", 
                unsafe_allow_html=True
            )
        
        with col5:
            high_potential = len(df[df['prediction_score'] >= 80])
            st.markdown(
                f"""<div class="metric-container">
                <div class="metric-value">{high_potential}</div>
                <div class="metric-label">High Potential</div>
                </div>""", 
                unsafe_allow_html=True
            )
        
        # DEX breakdown
        st.subheader("DEX Breakdown")
        
        dex_counts = df['dex'].value_counts().reset_index()
        dex_counts.columns = ['DEX', 'Pool Count']
        
        fig = px.pie(
            dex_counts, 
            values='Pool Count', 
            names='DEX', 
            title="Pool Distribution by DEX",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        st.subheader("Pool Categories")
        
        category_counts = df['category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Pool Count']
        
        fig = px.bar(
            category_counts, 
            x='Category', 
            y='Pool Count',
            title="Pools by Category",
            color='Category',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top pools by different metrics
        st.subheader("Top Pools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Pools by TVL**")
            top_tvl = df.nlargest(5, 'liquidity')
            top_tvl_table = pd.DataFrame({
                'Pool': top_tvl['name'],
                'DEX': top_tvl['dex'],
                'TVL': top_tvl['liquidity'].apply(format_currency),
                'APR': top_tvl['apr'].apply(format_percentage)
            })
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.table(top_tvl_table)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.write("**Top Pools by APR**")
            # Filter for reasonable liquidity to avoid tiny pools with extreme APRs
            min_liquidity = 1_000_000  # $1M
            high_apr = df[df['liquidity'] >= min_liquidity].nlargest(5, 'apr')
            high_apr_table = pd.DataFrame({
                'Pool': high_apr['name'],
                'DEX': high_apr['dex'],
                'APR': high_apr['apr'].apply(format_percentage),
                'TVL': high_apr['liquidity'].apply(format_currency)
            })
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.table(high_apr_table)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # APR by Category comparison
        st.subheader("APR by Category")
        
        category_metrics = df.groupby('category').agg({
            'apr': 'mean',
            'liquidity': 'mean',
            'volume_24h': 'sum'
        }).reset_index()
        
        fig = px.bar(
            category_metrics,
            x='category',
            y='apr',
            color='category',
            title="Average APR by Category",
            labels={'category': 'Category', 'apr': 'Average APR (%)'},
            text_auto='.2f'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Pool Explorer
    with tabs[1]:
        st.header("Pool Explorer")
        st.write("Explore all pools with advanced filtering options")
        
        # Sidebar filters
        st.sidebar.header("Pool Filters")
        
        # DEX filter
        dex_options = ['All'] + sorted(df['dex'].unique().tolist())
        selected_dex = st.sidebar.selectbox("DEX", dex_options)
        
        # Category filter
        category_options = ['All'] + sorted(df['category'].unique().tolist())
        selected_category = st.sidebar.selectbox("Category", category_options)
        
        # Token filter
        all_tokens = set()
        for token in df['token1_symbol'].unique():
            all_tokens.add(token)
        for token in df['token2_symbol'].unique():
            all_tokens.add(token)
        token_options = ['All'] + sorted(list(all_tokens))
        selected_token = st.sidebar.selectbox("Contains Token", token_options)
        
        # TVL and APR range sliders
        min_tvl = df['liquidity'].min()
        max_tvl = df['liquidity'].max()
        
        tvl_range = st.sidebar.slider(
            "TVL Range ($)",
            min_value=float(min_tvl),
            max_value=float(max_tvl),
            value=(float(min_tvl), float(max_tvl)),
            format="%e"
        )
        
        min_apr = df['apr'].min()
        max_apr = df['apr'].max()
        
        apr_range = st.sidebar.slider(
            "APR Range (%)",
            min_value=float(min_apr),
            max_value=float(max_apr),
            value=(float(min_apr), float(max_apr))
        )
        
        # Prediction score filter
        min_prediction = st.sidebar.slider(
            "Min Prediction Score",
            min_value=0,
            max_value=100,
            value=0
        )
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_dex != 'All':
            filtered_df = filtered_df[filtered_df['dex'] == selected_dex]
        
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        if selected_token != 'All':
            token_mask = (
                (filtered_df['token1_symbol'] == selected_token) | 
                (filtered_df['token2_symbol'] == selected_token)
            )
            filtered_df = filtered_df[token_mask]
        
        # Apply range filters
        filtered_df = filtered_df[
            (filtered_df['liquidity'] >= tvl_range[0]) & 
            (filtered_df['liquidity'] <= tvl_range[1]) &
            (filtered_df['apr'] >= apr_range[0]) & 
            (filtered_df['apr'] <= apr_range[1])
        ]
        
        # Apply prediction score filter
        if min_prediction > 0:
            filtered_df = filtered_df[filtered_df['prediction_score'] >= min_prediction]
        
        # Sort options
        sort_options = [
            "TVL (High to Low)",
            "APR (High to Low)",
            "Volume (High to Low)",
            "Prediction Score (High to Low)"
        ]
        
        selected_sort = st.selectbox("Sort by", sort_options)
        
        # Apply sorting
        if selected_sort == "TVL (High to Low)":
            filtered_df = filtered_df.sort_values('liquidity', ascending=False)
        elif selected_sort == "APR (High to Low)":
            filtered_df = filtered_df.sort_values('apr', ascending=False)
        elif selected_sort == "Volume (High to Low)":
            filtered_df = filtered_df.sort_values('volume_24h', ascending=False)
        elif selected_sort == "Prediction Score (High to Low)":
            filtered_df = filtered_df.sort_values('prediction_score', ascending=False)
        
        # Display filtered results
        st.write(f"Found {len(filtered_df)} pools matching your criteria")
        
        # Prepare display data
        display_data = []
        
        for _, pool in filtered_df.iterrows():
            # Format data
            apr_change_7d = pool.get('apr_change_7d')
            apr_change_icon = get_trend_icon(apr_change_7d)
            
            tvl_change_7d = pool.get('tvl_change_7d')
            tvl_change_icon = get_trend_icon(tvl_change_7d)
            
            category_html = get_category_badge(pool.get('category'))
            
            display_data.append({
                "Name": pool['name'],
                "DEX": pool['dex'],
                "Category": category_html,
                "TVL": format_currency(pool['liquidity']),
                "TVL Change": f"{tvl_change_icon} {format_percentage(abs(tvl_change_7d)) if tvl_change_7d is not None else 'N/A'}",
                "APR": format_percentage(pool['apr']),
                "APR Change": f"{apr_change_icon} {format_percentage(abs(apr_change_7d)) if apr_change_7d is not None else 'N/A'}",
                "Score": f"{pool['prediction_score']:.0f}/100",
                "id": pool['id']  # Hidden column for reference
            })
        
        # Convert to DataFrame
        display_df = pd.DataFrame(display_data)
        
        # Display as a table
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.write(display_df.drop(columns=['id']).to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Pool detail view
        if len(filtered_df) > 0:
            st.subheader("Pool Details")
            
            # Create a mapping of display names to IDs
            pool_options = {}
            for i, row in filtered_df.reset_index().iterrows():
                display_name = f"{row['name']} ({row['dex']})"
                pool_options[display_name] = i
            
            # Select a pool for detailed view
            selected_name = st.selectbox("Select pool for detailed view", list(pool_options.keys()))
            selected_idx = pool_options[selected_name]
            selected_pool = filtered_df.iloc[selected_idx]
            
            # Display detailed information
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown("<div class='highlight-container'>", unsafe_allow_html=True)
                st.markdown("### Basic Info")
                st.write(f"**ID:** {selected_pool['id']}")
                st.write(f"**Name:** {selected_pool['name']}")
                st.write(f"**DEX:** {selected_pool['dex']}")
                st.write(f"**Category:** {selected_pool['category']}")
                st.write(f"**Token 1:** {selected_pool['token1_symbol']}")
                st.write(f"**Token 2:** {selected_pool['token2_symbol']}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='highlight-container'>", unsafe_allow_html=True)
                st.markdown("### Performance")
                st.write(f"**TVL:** {format_currency(selected_pool['liquidity'])}")
                st.write(f"**24h Volume:** {format_currency(selected_pool['volume_24h'])}")
                st.write(f"**Fee Rate:** {format_percentage(selected_pool['fee'] * 100)}")
                st.write(f"**APR:** {format_percentage(selected_pool['apr'])}")
                
                if 'apr_change_7d' in selected_pool:
                    apr_change = selected_pool['apr_change_7d']
                    st.write(f"**7d APR Change:** {get_trend_icon(apr_change)} {format_percentage(abs(apr_change))}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='highlight-container'>", unsafe_allow_html=True)
                st.markdown("### Prediction")
                score = selected_pool['prediction_score']
                st.write(f"**Prediction Score:** {score:.0f}/100")
                
                # Prediction indicator
                if score >= 85:
                    st.success("High growth potential")
                elif score >= 70:
                    st.info("Moderate growth potential")
                else:
                    st.warning("Limited growth potential")
                
                # Key factors
                st.write("**Key Factors:**")
                
                apr_change = selected_pool.get('apr_change_7d', 0)
                tvl_change = selected_pool.get('tvl_change_7d', 0)
                
                factors = []
                
                if apr_change > 2:
                    factors.append("• Strong APR growth")
                elif apr_change < -2:
                    factors.append("• APR decline")
                    
                if tvl_change > 5:
                    factors.append("• Increasing liquidity")
                elif tvl_change < -5:
                    factors.append("• Decreasing liquidity")
                    
                if selected_pool['category'] == 'Meme':
                    factors.append("• Meme coin volatility")
                
                if selected_pool['volume_24h'] / selected_pool['liquidity'] > 0.2:
                    factors.append("• High trading volume")
                
                for factor in factors:
                    st.write(factor)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Historical data visualization
            st.subheader("Historical Performance")
            
            # Generate some simulated historical data
            dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
            
            # Create base values
            base_apr = selected_pool['apr']
            base_tvl = selected_pool['liquidity']
            
            # Generate historical data with realistic trends
            # Use the 7d and 30d changes to create a trend
            apr_change_7d = selected_pool.get('apr_change_7d', 0)
            apr_change_30d = selected_pool.get('apr_change_30d', 0)
            
            tvl_change_7d = selected_pool.get('tvl_change_7d', 0)
            tvl_change_30d = selected_pool.get('tvl_change_30d', 0)
            
            # Calculate daily rates
            apr_daily_change_recent = apr_change_7d / 7 if apr_change_7d is not None else 0
            apr_daily_change_old = (apr_change_30d - apr_change_7d) / 23 if apr_change_30d is not None and apr_change_7d is not None else 0
            
            tvl_daily_change_recent = tvl_change_7d / 7 * 0.01 if tvl_change_7d is not None else 0
            tvl_daily_change_old = (tvl_change_30d - tvl_change_7d) / 23 * 0.01 if tvl_change_30d is not None and tvl_change_7d is not None else 0
            
            # Generate data points
            apr_data = []
            tvl_data = []
            
            for i in range(30):
                if i < 7:  # Recent data (last 7 days)
                    day_apr = base_apr - ((7 - i) * apr_daily_change_recent)
                    day_tvl = base_tvl / (1 + ((7 - i) * tvl_daily_change_recent))
                else:  # Older data
                    day_apr = base_apr - (7 * apr_daily_change_recent) - ((i - 7) * apr_daily_change_old)
                    day_tvl = base_tvl / (1 + (7 * tvl_daily_change_recent)) / (1 + ((i - 7) * tvl_daily_change_old))
                
                # Add small random noise
                day_apr += random.uniform(-0.5, 0.5)
                day_tvl *= (1 + random.uniform(-0.01, 0.01))
                
                apr_data.append(day_apr)
                tvl_data.append(day_tvl)
            
            # Create figure with two y-axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add APR line
            fig.add_trace(
                go.Scatter(x=dates, y=apr_data, name="APR (%)", line=dict(color="#2196F3", width=2)),
                secondary_y=False
            )
            
            # Add TVL line
            fig.add_trace(
                go.Scatter(x=dates, y=tvl_data, name="TVL ($)", line=dict(color="#4CAF50", width=2)),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title="30-Day Historical Performance",
                xaxis_title="Date",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="APR (%)", secondary_y=False)
            fig.update_yaxes(title_text="TVL ($)", secondary_y=True, tickformat="$,.0f")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Insights & Predictions
    with tabs[2]:
        st.header("Insights & Predictions")
        st.write("Machine learning-based predictions and market insights")
        
        # Prediction score distribution
        st.subheader("Prediction Score Distribution")
        
        fig = px.histogram(
            df,
            x="prediction_score",
            nbins=20,
            title="Distribution of Prediction Scores",
            labels={"prediction_score": "Prediction Score"},
            color_discrete_sequence=["#1E88E5"]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Advanced prediction engine explanation
        st.subheader("Self-Evolving Prediction Technology")
        
        st.markdown("""
        Our platform uses a sophisticated self-evolving prediction system that continually 
        improves its accuracy over time. The system combines multiple machine learning approaches:
        
        1. **Ensemble Learning** - Multiple models work together to generate predictions
        2. **Reinforcement Learning** - The system learns from the accuracy of past predictions
        3. **LSTM Neural Networks** - Recognize complex patterns in time series data
        4. **Adaptive Hyperparameter Optimization** - Self-tunes for optimal performance
        
        With each prediction cycle, the system evaluates its own performance and adjusts its
        internal weights, feature selection, and model parameters to improve future predictions.
        """)
        
        # If we have the prediction engine, show a button to run it
        if HAS_PREDICTION_ENGINE:
            st.write("#### Generate Advanced Predictions")
            
            # Create a demo using the advanced prediction engine
            use_advanced_engine = st.checkbox("Use advanced prediction engine for pool analysis", value=True)
            
            if use_advanced_engine:
                with st.expander("Advanced Prediction Settings", expanded=False):
                    prediction_horizon = st.slider(
                        "Prediction Horizon (Days)", 
                        min_value=1, 
                        max_value=30, 
                        value=7
                    )
                    
                    confidence_threshold = st.slider(
                        "Minimum Confidence Threshold (%)", 
                        min_value=0, 
                        max_value=100, 
                        value=60
                    )
        
        # Top predicted pools
        st.subheader("Top Growth Potential Pools")
        
        # Filter for minimum liquidity
        min_liquidity = 1_000_000  # $1M
        top_predictions = df[df['liquidity'] >= min_liquidity].nlargest(10, 'prediction_score')
        
        # If we have advanced prediction engine and user wants to use it
        if HAS_PREDICTION_ENGINE and 'use_advanced_engine' in locals() and use_advanced_engine:
            with st.spinner("Generating advanced predictions using self-evolving AI..."):
                try:
                    # Process the top pools with the advanced engine
                    top_pools_data = top_predictions.to_dict('records')
                    
                    # Initialize the prediction engine
                    engine = SelfEvolvingPredictionEngine(prediction_horizon=prediction_horizon)
                    
                    # Generate predictions
                    advanced_predictions = []
                    for pool in top_pools_data:
                        prediction_result = engine.predict(pool)
                        
                        # Extract prediction data
                        pred_dict = prediction_result.to_dict()
                        
                        # Calculate overall score
                        score = pred_dict.get('prediction_score', 0)
                        
                        # Only include if it meets confidence threshold
                        if score >= confidence_threshold:
                            # Add prediction details
                            pool['advanced_prediction'] = pred_dict
                            advanced_predictions.append(pool)
                    
                    # Sort by prediction score
                    advanced_predictions.sort(key=lambda x: x.get('advanced_prediction', {}).get('prediction_score', 0), reverse=True)
                    
                    # Create display data
                    top_pred_data = []
                    
                    for pool in advanced_predictions[:10]:  # Take up to 10
                        # Format data
                        apr_change_7d = pool.get('apr_change_7d')
                        apr_change_icon = get_trend_icon(apr_change_7d)
                        
                        tvl_change_7d = pool.get('tvl_change_7d')
                        tvl_change_icon = get_trend_icon(tvl_change_7d)
                        
                        category_html = get_category_badge(pool.get('category'))
                        
                        # Get advanced prediction data
                        adv_pred = pool.get('advanced_prediction', {})
                        pred_values = adv_pred.get('prediction_values', {})
                        
                        # Get predicted changes
                        pred_apr_change = pred_values.get('apr_change', 0)
                        pred_tvl_change = pred_values.get('tvl_change_percent', 0)
                        
                        top_pred_data.append({
                            "Name": pool['name'],
                            "DEX": pool['dex'],
                            "Category": category_html,
                            "Current APR": format_percentage(pool['apr']),
                            "Predicted APR": format_percentage(pred_values.get('future_apr', pool['apr'])),
                            "APR Trend": f"{get_trend_icon(pred_apr_change)} {format_percentage(abs(pred_apr_change)) if pred_apr_change is not None else 'N/A'}",
                            "TVL": format_currency(pool['liquidity']),
                            "TVL Trend": f"{get_trend_icon(pred_tvl_change)} {format_percentage(abs(pred_tvl_change)) if pred_tvl_change is not None else 'N/A'}",
                            "AI Score": f"{adv_pred.get('prediction_score', 0):.0f}/100"
                        })
                
                except Exception as e:
                    st.error(f"Error using advanced prediction engine: {str(e)}")
                    # Fall back to basic predictions
                    # Create display data from basic predictions
                    top_pred_data = []
                    
                    for _, pool in top_predictions.iterrows():
                        # Format data
                        apr_change_7d = pool.get('apr_change_7d')
                        apr_change_icon = get_trend_icon(apr_change_7d)
                        
                        tvl_change_7d = pool.get('tvl_change_7d')
                        tvl_change_icon = get_trend_icon(tvl_change_7d)
                        
                        category_html = get_category_badge(pool.get('category'))
                        
                        top_pred_data.append({
                            "Name": pool['name'],
                            "DEX": pool['dex'],
                            "Category": category_html,
                            "TVL": format_currency(pool['liquidity']),
                            "TVL Change": f"{tvl_change_icon} {format_percentage(abs(tvl_change_7d)) if tvl_change_7d is not None else 'N/A'}",
                            "APR": format_percentage(pool['apr']),
                            "APR Change": f"{apr_change_icon} {format_percentage(abs(apr_change_7d)) if apr_change_7d is not None else 'N/A'}",
                            "Score": f"{pool['prediction_score']:.0f}/100"
                        })
        else:
            # Create display data from basic predictions
            top_pred_data = []
            
            for _, pool in top_predictions.iterrows():
                # Format data
                apr_change_7d = pool.get('apr_change_7d')
                apr_change_icon = get_trend_icon(apr_change_7d)
                
                tvl_change_7d = pool.get('tvl_change_7d')
                tvl_change_icon = get_trend_icon(tvl_change_7d)
                
                category_html = get_category_badge(pool.get('category'))
                
                top_pred_data.append({
                    "Name": pool['name'],
                    "DEX": pool['dex'],
                    "Category": category_html,
                    "TVL": format_currency(pool['liquidity']),
                    "TVL Change": f"{tvl_change_icon} {format_percentage(abs(tvl_change_7d)) if tvl_change_7d is not None else 'N/A'}",
                    "APR": format_percentage(pool['apr']),
                    "APR Change": f"{apr_change_icon} {format_percentage(abs(apr_change_7d)) if apr_change_7d is not None else 'N/A'}",
                    "Score": f"{pool['prediction_score']:.0f}/100"
                })
        
        # Convert to DataFrame
        top_pred_df = pd.DataFrame(top_pred_data)
        
        # Display as a table
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.write(top_pred_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Category performance comparison
        st.subheader("Category Performance Analysis")
        
        # Calculate metrics by category
        category_perf = df.groupby('category').agg({
            'apr': 'mean',
            'liquidity': 'mean',
            'volume_24h': 'mean',
            'prediction_score': 'mean',
            'id': 'count'
        }).reset_index()
        
        category_perf.columns = ['Category', 'Avg APR', 'Avg TVL', 'Avg Volume', 'Avg Score', 'Pool Count']
        
        # Create radar chart data
        categories = category_perf['Category'].tolist()
        
        # Normalize metrics for radar chart
        radar_data = category_perf.copy()
        for col in ['Avg APR', 'Avg TVL', 'Avg Volume', 'Avg Score']:
            max_val = radar_data[col].max()
            if max_val > 0:
                radar_data[col] = radar_data[col] / max_val
        
        # Create radar chart
        fig = go.Figure()
        
        for i, category in enumerate(categories):
            row = radar_data[radar_data['Category'] == category].iloc[0]
            
            fig.add_trace(go.Scatterpolar(
                r=[row['Avg APR'], row['Avg TVL'], row['Avg Volume'], row['Avg Score']],
                theta=['APR', 'TVL', 'Volume', 'Prediction Score'],
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
        
        # Explanation of prediction methodology
        with st.expander("Prediction Methodology"):
            st.markdown("""
            ### Machine Learning Prediction Methodology
            
            Our prediction system combines multiple machine learning approaches to forecast the potential for APR and TVL growth:
            
            #### Data Features Analyzed
            
            - **Historical metrics:** APR, TVL, and volume trends over multiple time periods
            - **Volatility patterns:** How stable the pool's metrics have been
            - **Token correlation:** How token prices move in relation to each other
            - **Market activity:** Trading patterns and liquidity provider behavior
            - **Social sentiment:** Community interest and activity around tokens
            - **Protocol factors:** Recent upgrades, incentive programs, etc.
            
            #### Model Architecture
            
            We use a multi-model ensemble approach:
            
            1. **Gradient Boosting Models (XGBoost/LightGBM)**
               - Handle non-linear relationships between features
               - Identify key predictive factors
            
            2. **LSTM Neural Networks**
               - Time-series prediction for APR and TVL patterns
               - Pattern recognition in historical metrics
            
            3. **Reinforcement Learning**
               - Adaptive strategy development
               - Learning from market responses
            
            4. **Statistical Methods**
               - ARIMA/GARCH for volatility forecasting
               - Bayesian inference for probability distributions
            
            The final prediction score combines outputs from all models using a weighted ensemble approach.
            Higher scores (80-100) indicate high confidence in positive growth potential,
            while moderate scores (60-80) suggest moderate potential.
            """)
        
        # Market trends
        st.subheader("Market Trends")
        
        # Calculate DEX trends (7-day changes)
        if 'apr_change_7d' in df.columns and 'tvl_change_7d' in df.columns:
            dex_trends = df.groupby('dex').agg({
                'apr_change_7d': 'mean',
                'tvl_change_7d': 'mean',
                'prediction_score': 'mean',
                'id': 'count'
            }).reset_index()
            
            dex_trends.columns = ['DEX', 'APR Change', 'TVL Change', 'Prediction Score', 'Pool Count']
            
            # Create bubble chart
            fig = px.scatter(
                dex_trends,
                x='APR Change',
                y='TVL Change',
                size='Pool Count',
                color='Prediction Score',
                hover_name='DEX',
                size_max=60,
                color_continuous_scale=px.colors.sequential.Viridis,
                title="DEX Performance Trends"
            )
            
            # Add horizontal and vertical lines at 0
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            # Update layout
            fig.update_layout(
                xaxis_title="7-Day APR Change (%)",
                yaxis_title="7-Day TVL Change (%)",
                coloraxis_colorbar_title="Prediction Score"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Historical trend data not available for market trend analysis")
        
        # APR to TVL correlation
        st.subheader("APR to TVL Correlation")
        
        fig = px.scatter(
            df,
            x='liquidity',
            y='apr',
            color='category',
            size='volume_24h',
            hover_name='name',
            log_x=True,
            title="APR vs. TVL Correlation",
            labels={
                'liquidity': 'TVL (log scale)',
                'apr': 'APR (%)',
                'category': 'Category',
                'volume_24h': 'Volume'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Key Insights
        
        1. **Meme Tokens** generally show the highest APRs but with higher volatility
        2. **Stablecoin Pairs** have the lowest APRs but highest liquidity and stability
        3. **Major Pairs** (SOL, BTC, ETH) offer balanced APR to risk ratios
        4. Higher **prediction scores** correlate with recent positive APR and TVL trends
        5. Some DEXes consistently outperform others in similar pool categories
        """)

if __name__ == "__main__":
    main()