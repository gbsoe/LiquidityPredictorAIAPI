import streamlit as st
import random
from datetime import datetime, timedelta
import pandas as pd

# Enhanced pool data with more meme coins and historical metrics
POOL_DATA = [
    # Major pairs
    {
        "id": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
        "name": "SOL/USDC",
        "dex": "Raydium",
        "category": "Major",
        "tvl": 24532890.45,
        "volume24h": 8763021.32,
        "apr": {
            "current": 12.87,
            "24h_ago": 12.45,
            "7d_ago": 13.2,
            "30d_ago": 11.8
        },
        "tvl_change": {
            "24h": 1.2,
            "7d": 3.5,
            "30d": -2.1
        },
        "prediction_score": 85  # Hypothetical ML prediction score (0-100)
    },
    {
        "id": "7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX",
        "name": "SOL/USDT",
        "dex": "Raydium",
        "category": "Major",
        "tvl": 18245789.12,
        "volume24h": 6542891.45,
        "apr": {
            "current": 11.23,
            "24h_ago": 11.45,
            "7d_ago": 10.9,
            "30d_ago": 12.3
        },
        "tvl_change": {
            "24h": -0.8,
            "7d": 2.3,
            "30d": -1.5
        },
        "prediction_score": 72
    },
    {
        "id": "AVs9TA4nWDzfPJE9gGVNJMVhcQy3V9PGazuz33BfG2RA",
        "name": "SOL/RAY",
        "dex": "Raydium",
        "category": "Major",
        "tvl": 5678234.89,
        "volume24h": 1987654.32,
        "apr": {
            "current": 15.42,
            "24h_ago": 15.12,
            "7d_ago": 14.8,
            "30d_ago": 16.2
        },
        "tvl_change": {
            "24h": 0.5,
            "7d": 1.2,
            "30d": -3.8
        },
        "prediction_score": 68
    },
    
    # Meme coins (expanded section)
    {
        "id": "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K",
        "name": "BONK/USDC",
        "dex": "Meteora",
        "category": "Meme",
        "tvl": 5432167.89,
        "volume24h": 1987654.32,
        "apr": {
            "current": 25.67,
            "24h_ago": 24.12,
            "7d_ago": 28.9,
            "30d_ago": 18.4
        },
        "tvl_change": {
            "24h": 4.2,
            "7d": 15.6,
            "30d": 32.7
        },
        "prediction_score": 94
    },
    {
        "id": "Dooar9JkhdZ7J3LHN3A7YCuoGRUggXhQaG4kijfLGU2j",
        "name": "SAMO/USDC",
        "dex": "Raydium",
        "category": "Meme",
        "tvl": 3456789.01,
        "volume24h": 876543.21,
        "apr": {
            "current": 22.45,
            "24h_ago": 21.78,
            "7d_ago": 23.1,
            "30d_ago": 19.8
        },
        "tvl_change": {
            "24h": 2.8,
            "7d": 7.9,
            "30d": 15.2
        },
        "prediction_score": 88
    },
    {
        "id": "B0nkD2EW5B0nK1nG51mECoiNSolANaPooL5Us3",
        "name": "BONK/SOL",
        "dex": "Raydium",
        "category": "Meme",
        "tvl": 2345678.90,
        "volume24h": 1234567.89,
        "apr": {
            "current": 28.90,
            "24h_ago": 27.65,
            "7d_ago": 30.2,
            "30d_ago": 21.5
        },
        "tvl_change": {
            "24h": 5.6,
            "7d": 12.3,
            "30d": 45.6
        },
        "prediction_score": 96
    },
    {
        "id": "D0gEY2pCANiNEkn0wZ1nG51nEPuPpY74eF9UsF",
        "name": "DOGWIFHAT/USDC",
        "dex": "Orca",
        "category": "Meme",
        "tvl": 4567890.12,
        "volume24h": 2345678.90,
        "apr": {
            "current": 32.45,
            "24h_ago": 31.23,
            "7d_ago": 26.7,
            "30d_ago": 22.1
        },
        "tvl_change": {
            "24h": 7.8,
            "7d": 18.9,
            "30d": 65.3
        },
        "prediction_score": 92
    },
    {
        "id": "P0pCaT5Ec0iNR3P0mEk0iN51T0kENpuPpY",
        "name": "POPCAT/USDC",
        "dex": "Raydium",
        "category": "Meme",
        "tvl": 1234567.89,
        "volume24h": 987654.32,
        "apr": {
            "current": 38.90,
            "24h_ago": 35.67,
            "7d_ago": 29.8,
            "30d_ago": 20.4
        },
        "tvl_change": {
            "24h": 8.9,
            "7d": 25.6,
            "30d": 78.9
        },
        "prediction_score": 89
    },
    {
        "id": "B1g5M0nK3yR3p0Th3M0nK3y1nTh3B4nK",
        "name": "BIGMONKEY/USDC",
        "dex": "Jupiter",
        "category": "Meme",
        "tvl": 987654.32,
        "volume24h": 567890.12,
        "apr": {
            "current": 42.78,
            "24h_ago": 38.90,
            "7d_ago": 35.6,
            "30d_ago": 25.7
        },
        "tvl_change": {
            "24h": 10.2,
            "7d": 28.7,
            "30d": 90.5
        },
        "prediction_score": 97
    },
    
    # Other significant pools
    {
        "id": "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",
        "name": "BTC/USDC",
        "dex": "Raydium",
        "category": "Major",
        "tvl": 45321987.65,
        "volume24h": 12345678.90,
        "apr": {
            "current": 9.87,
            "24h_ago": 9.65,
            "7d_ago": 10.2,
            "30d_ago": 9.1
        },
        "tvl_change": {
            "24h": 0.3,
            "7d": -1.2,
            "30d": 2.5
        },
        "prediction_score": 65
    },
    {
        "id": "7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX",
        "name": "ETH/USDC",
        "dex": "Raydium",
        "category": "Major",
        "tvl": 32145678.90,
        "volume24h": 9876543.21,
        "apr": {
            "current": 10.23,
            "24h_ago": 10.45,
            "7d_ago": 9.8,
            "30d_ago": 10.9
        },
        "tvl_change": {
            "24h": -0.5,
            "7d": 1.8,
            "30d": -0.7
        },
        "prediction_score": 70
    },
    {
        "id": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
        "name": "JUP/USDC",
        "dex": "Jupiter",
        "category": "DeFi",
        "tvl": 8765432.10,
        "volume24h": 2345678.90,
        "apr": {
            "current": 19.87,
            "24h_ago": 18.56,
            "7d_ago": 20.3,
            "30d_ago": 17.8
        },
        "tvl_change": {
            "24h": 1.8,
            "7d": 5.6,
            "30d": 12.3
        },
        "prediction_score": 82
    },
    {
        "id": "2LQdMz7YXqRwBUv3oNm8oEvs3tX3ASXhUAP3apPMYXeR",
        "name": "MNGO/USDC",
        "dex": "Raydium",
        "category": "DeFi",
        "tvl": 1987654.32,
        "volume24h": 765432.19,
        "apr": {
            "current": 21.34,
            "24h_ago": 20.89,
            "7d_ago": 22.1,
            "30d_ago": 19.5
        },
        "tvl_change": {
            "24h": 1.2,
            "7d": 3.8,
            "30d": 8.9
        },
        "prediction_score": 75
    },
    
    # New DeFi token pools
    {
        "id": "MR1iySt0kF1UiDsZ8pzT7WYMLViVGz9iF0rGd",
        "name": "MARI/USDC",
        "dex": "Raydium",
        "category": "DeFi",
        "tvl": 3456789.01,
        "volume24h": 1234567.89,
        "apr": {
            "current": 24.56,
            "24h_ago": 23.78,
            "7d_ago": 25.1,
            "30d_ago": 20.3
        },
        "tvl_change": {
            "24h": 2.3,
            "7d": 8.9,
            "30d": 15.6
        },
        "prediction_score": 85
    },
    {
        "id": "K4mINOpZ1UiDsZ8pzT7WYMLViVGz9iF0rGd",
        "name": "KAMINO/USDC",
        "dex": "Orca",
        "category": "DeFi",
        "tvl": 2345678.90,
        "volume24h": 987654.32,
        "apr": {
            "current": 18.90,
            "24h_ago": 18.45,
            "7d_ago": 19.3,
            "30d_ago": 17.2
        },
        "tvl_change": {
            "24h": 1.5,
            "7d": 4.2,
            "30d": 9.8
        },
        "prediction_score": 78
    },
    {
        "id": "D3r1V471VEpZ1UiDsZ8pzT7WYMLViVGz9iF0rGd",
        "name": "DERIVE/USDC",
        "dex": "Raydium",
        "category": "DeFi",
        "tvl": 1234567.89,
        "volume24h": 567890.12,
        "apr": {
            "current": 26.78,
            "24h_ago": 25.90,
            "7d_ago": 27.4,
            "30d_ago": 22.1
        },
        "tvl_change": {
            "24h": 2.8,
            "7d": 7.5,
            "30d": 18.9
        },
        "prediction_score": 82
    },
    
    # Gaming token pools
    {
        "id": "A4rTkD3zkUiDsZ8pzT7WYMLViVGz9iF0rGd",
        "name": "AURORY/USDC",
        "dex": "Orca",
        "category": "Gaming",
        "tvl": 3456789.01,
        "volume24h": 1234567.89,
        "apr": {
            "current": 20.56,
            "24h_ago": 19.78,
            "7d_ago": 21.3,
            "30d_ago": 18.7
        },
        "tvl_change": {
            "24h": 1.8,
            "7d": 5.2,
            "30d": 12.4
        },
        "prediction_score": 80
    },
    {
        "id": "St4rA7l4skUiDsZ8pzT7WYMLViVGz9iF0rGd",
        "name": "STAR/USDC",
        "dex": "Raydium",
        "category": "Gaming",
        "tvl": 2345678.90,
        "volume24h": 987654.32,
        "apr": {
            "current": 22.45,
            "24h_ago": 21.67,
            "7d_ago": 23.1,
            "30d_ago": 19.8
        },
        "tvl_change": {
            "24h": 2.1,
            "7d": 6.8,
            "30d": 14.5
        },
        "prediction_score": 83
    },
    
    # Stablecoin pairs
    {
        "id": "9X4JWzXhKiM6AbiYBmm58JcBxVXJiQ2hT5d4k5RQsLxZ",
        "name": "USDC/USDT",
        "dex": "Saber",
        "category": "Stablecoin",
        "tvl": 54321987.65,
        "volume24h": 8765432.10,
        "apr": {
            "current": 5.67,
            "24h_ago": 5.65,
            "7d_ago": 5.7,
            "30d_ago": 5.5
        },
        "tvl_change": {
            "24h": 0.1,
            "7d": 0.3,
            "30d": 1.2
        },
        "prediction_score": 60
    },
    {
        "id": "U5dCD41kUiDsZ8pzT7WYMLViVGz9iF0rGd",
        "name": "USDC/DAI",
        "dex": "Saber",
        "category": "Stablecoin",
        "tvl": 32145678.90,
        "volume24h": 4567890.12,
        "apr": {
            "current": 4.89,
            "24h_ago": 4.85,
            "7d_ago": 4.9,
            "30d_ago": 4.7
        },
        "tvl_change": {
            "24h": 0.08,
            "7d": 0.2,
            "30d": 0.9
        },
        "prediction_score": 55
    }
]

def format_currency(value):
    if value is None:
        return "N/A"
    return f"${value:,.2f}"

def format_percentage(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}%"

def get_trend_arrow(value):
    if value is None:
        return ""
    return "↑" if value >= 0 else "↓"

def get_trend_color(value):
    if value is None:
        return "black"
    return "green" if value >= 0 else "red"

def get_prediction_color(score):
    if score >= 85:
        return "green"
    elif score >= 70:
        return "yellow"
    else:
        return "gray"

# Function to generate historical TVL data (for demonstration)
def generate_historical_data(base_value, days, volatility):
    data = []
    current_date = datetime.now()
    
    value = base_value
    for i in range(days):
        date = current_date - timedelta(days=days-i-1)
        change = random.uniform(-volatility, volatility) * base_value / 100
        value = value + change
        data.append({"date": date, "value": value})
    
    return data

def main():
    st.title("Solana Liquidity Pool Analysis")
    
    st.write("""
    This application demonstrates our ability to analyze a wide variety of Solana liquidity pools, 
    including meme coins, DeFi tokens, gaming tokens, and major pairs.
    
    We track historical APR and TVL changes to make predictions about future performance.
    """)
    
    # Count total pools
    total_pools = len(POOL_DATA)
    
    # Calculate total liquidity across all pools
    total_tvl = sum(pool["tvl"] for pool in POOL_DATA)
    
    # Calculate total 24h volume
    total_volume = sum(pool["volume24h"] for pool in POOL_DATA)
    
    # Calculate average APR
    avg_apr = sum(pool["apr"]["current"] for pool in POOL_DATA) / total_pools
    
    # Calculate average prediction score
    avg_prediction = sum(pool["prediction_score"] for pool in POOL_DATA) / total_pools
    
    # Show summary metrics
    st.subheader("Market Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Pools", f"{total_pools}")
    
    with col2:
        st.metric("Total TVL", f"${total_tvl/1e9:.2f}B")
    
    with col3:
        st.metric("24h Volume", f"${total_volume/1e9:.2f}B")
    
    with col4:
        st.metric("Avg APR", f"{avg_apr:.2f}%")
    
    with col5:
        st.metric("Avg Prediction", f"{avg_prediction:.0f}/100")
    
    # Advanced filtering
    st.sidebar.title("Pool Filters")
    
    # DEX filter
    dex_list = sorted(list(set(pool["dex"] for pool in POOL_DATA)))
    selected_dexes = st.sidebar.multiselect("DEX", dex_list, default=dex_list)
    
    # Category filter
    category_list = sorted(list(set(pool["category"] for pool in POOL_DATA)))
    selected_categories = st.sidebar.multiselect("Category", category_list, default=category_list)
    
    # TVL range filter
    min_tvl = min(pool["tvl"] for pool in POOL_DATA)
    max_tvl = max(pool["tvl"] for pool in POOL_DATA)
    
    tvl_range = st.sidebar.slider(
        "TVL Range ($)",
        min_value=float(min_tvl),
        max_value=float(max_tvl),
        value=(float(min_tvl), float(max_tvl)),
        format="$%e"
    )
    
    # APR range filter
    min_apr = min(pool["apr"]["current"] for pool in POOL_DATA)
    max_apr = max(pool["apr"]["current"] for pool in POOL_DATA)
    
    apr_range = st.sidebar.slider(
        "APR Range (%)",
        min_value=float(min_apr),
        max_value=float(max_apr),
        value=(float(min_apr), float(max_apr)),
        format="%f%%"
    )
    
    # Prediction score threshold
    prediction_threshold = st.sidebar.slider(
        "Min Prediction Score",
        min_value=0,
        max_value=100,
        value=0
    )
    
    # Apply filters
    filtered_pools = [
        pool for pool in POOL_DATA 
        if pool["dex"] in selected_dexes
        and pool["category"] in selected_categories
        and tvl_range[0] <= pool["tvl"] <= tvl_range[1]
        and apr_range[0] <= pool["apr"]["current"] <= apr_range[1]
        and pool["prediction_score"] >= prediction_threshold
    ]
    
    # Sort options
    sort_options = {
        "TVL (High to Low)": ("tvl", False),
        "TVL (Low to High)": ("tvl", True),
        "Volume (High to Low)": ("volume24h", False),
        "Volume (Low to High)": ("volume24h", True),
        "APR (High to Low)": ("apr", False),
        "APR (Low to High)": ("apr", True),
        "Prediction Score (High to Low)": ("prediction_score", False),
        "Prediction Score (Low to High)": ("prediction_score", True)
    }
    
    sort_by = st.selectbox("Sort by", list(sort_options.keys()))
    
    # Apply sorting
    sort_key, ascending = sort_options[sort_by]
    
    if sort_key == "apr":
        filtered_pools = sorted(filtered_pools, key=lambda x: x["apr"]["current"], reverse=not ascending)
    else:
        filtered_pools = sorted(filtered_pools, key=lambda x: x[sort_key], reverse=not ascending)
    
    # Show filtered pools
    st.subheader(f"Pools ({len(filtered_pools)})")
    
    # Create a table for pool data
    table_data = []
    for pool in filtered_pools:
        table_data.append({
            "Name": pool["name"],
            "DEX": pool["dex"],
            "Category": pool["category"],
            "TVL": format_currency(pool["tvl"]),
            "24h Vol": format_currency(pool["volume24h"]),
            "APR": format_percentage(pool["apr"]["current"]),
            "7D APR Change": f"{get_trend_arrow(pool['apr']['current'] - pool['apr']['7d_ago'])} {format_percentage(abs(pool['apr']['current'] - pool['apr']['7d_ago']))}",
            "30D TVL Change": f"{get_trend_arrow(pool['tvl_change']['30d'])} {format_percentage(abs(pool['tvl_change']['30d']))}",
            "Prediction": f"{pool['prediction_score']}/100"
        })
    
    # Convert to DataFrame for display
    df = pd.DataFrame(table_data)
    st.dataframe(df)
    
    # Pool Details Section
    st.subheader("Pool Details & Historical Analysis")
    
    if filtered_pools:
        pool_names = [f"{pool['name']} ({pool['dex']})" for pool in filtered_pools]
        selected_pool_name = st.selectbox("Select Pool for Detailed Analysis", pool_names)
        
        # Get the selected pool
        selected_idx = pool_names.index(selected_pool_name)
        selected_pool = filtered_pools[selected_idx]
        
        # Display pool details in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("### Basic Info")
            st.write(f"**ID:** {selected_pool['id']}")
            st.write(f"**Name:** {selected_pool['name']}")
            st.write(f"**DEX:** {selected_pool['dex']}")
            st.write(f"**Category:** {selected_pool['category']}")
        
        with col2:
            st.write("### Current Metrics")
            st.write(f"**TVL:** {format_currency(selected_pool['tvl'])}")
            st.write(f"**24h Volume:** {format_currency(selected_pool['volume24h'])}")
            st.write(f"**Current APR:** {format_percentage(selected_pool['apr']['current'])}")
            st.write(f"**Prediction Score:** {selected_pool['prediction_score']}/100")
        
        with col3:
            st.write("### Historical Changes")
            
            # APR changes
            apr_24h_change = selected_pool['apr']['current'] - selected_pool['apr']['24h_ago']
            apr_7d_change = selected_pool['apr']['current'] - selected_pool['apr']['7d_ago']
            apr_30d_change = selected_pool['apr']['current'] - selected_pool['apr']['30d_ago']
            
            # Format with trend arrows
            st.write(f"**APR 24h Change:** {get_trend_arrow(apr_24h_change)} {format_percentage(abs(apr_24h_change))}")
            st.write(f"**APR 7D Change:** {get_trend_arrow(apr_7d_change)} {format_percentage(abs(apr_7d_change))}")
            st.write(f"**APR 30D Change:** {get_trend_arrow(apr_30d_change)} {format_percentage(abs(apr_30d_change))}")
            
            # TVL changes
            st.write(f"**TVL 24h Change:** {get_trend_arrow(selected_pool['tvl_change']['24h'])} {format_percentage(abs(selected_pool['tvl_change']['24h']))}")
            st.write(f"**TVL 7D Change:** {get_trend_arrow(selected_pool['tvl_change']['7d'])} {format_percentage(abs(selected_pool['tvl_change']['7d']))}")
            st.write(f"**TVL 30D Change:** {get_trend_arrow(selected_pool['tvl_change']['30d'])} {format_percentage(abs(selected_pool['tvl_change']['30d']))}")
        
        # Show historical APR data in a table
        st.write("### Historical APR")
        historical_apr = {
            "Time Period": ["Current", "24 Hours Ago", "7 Days Ago", "30 Days Ago"],
            "APR": [
                format_percentage(selected_pool['apr']['current']),
                format_percentage(selected_pool['apr']['24h_ago']),
                format_percentage(selected_pool['apr']['7d_ago']), 
                format_percentage(selected_pool['apr']['30d_ago'])
            ]
        }
        st.table(pd.DataFrame(historical_apr))
        
        # Show prediction analysis
        st.write("### Prediction Analysis")
        st.write(f"""
        Based on our machine learning model analysis, this pool has a prediction score of 
        **{selected_pool['prediction_score']}/100**, indicating 
        {'a high' if selected_pool['prediction_score'] >= 85 else 'a moderate' if selected_pool['prediction_score'] >= 70 else 'a lower'} 
        probability of TVL and APR growth in the near future.
        """)
        
        if selected_pool['prediction_score'] >= 85:
            st.success(f"This pool shows strong growth potential with a score of {selected_pool['prediction_score']}/100")
        elif selected_pool['prediction_score'] >= 70:
            st.info(f"This pool shows moderate growth potential with a score of {selected_pool['prediction_score']}/100")
        else:
            st.warning(f"This pool shows limited growth potential with a score of {selected_pool['prediction_score']}/100")
        
        # Key factors influencing prediction
        st.write("### Key Factors Influencing Prediction")
        
        factors = []
        if selected_pool['tvl_change']['7d'] > 5:
            factors.append("Strong 7-day TVL growth")
        elif selected_pool['tvl_change']['7d'] < -5:
            factors.append("Declining 7-day TVL")
            
        if selected_pool['apr']['current'] > selected_pool['apr']['7d_ago']:
            factors.append("Increasing APR trend")
        elif selected_pool['apr']['current'] < selected_pool['apr']['7d_ago']:
            factors.append("Decreasing APR trend")
            
        if selected_pool['category'] == 'Meme':
            factors.append("Meme coin volatility factor")
        
        if selected_pool['volume24h'] > 1000000:
            factors.append("High trading volume")
        elif selected_pool['volume24h'] < 500000:
            factors.append("Low trading volume")
            
        for factor in factors:
            st.write(f"• {factor}")
            
        # Investment Opportunities section
        if selected_pool['prediction_score'] >= 80:
            st.write("### Investment Strategy Recommendation")
            st.write("""
            **Opportunity**: This pool shows significant growth potential and could be a good candidate
            for liquidity provision to earn APR rewards.
            """)
        
    else:
        st.warning("No pools match your selected filters.")
    
    # Category Analysis
    st.subheader("Category Analysis")
    
    # Group pools by category and calculate averages
    categories = {}
    for pool in POOL_DATA:
        category = pool["category"]
        if category not in categories:
            categories[category] = {
                "count": 0,
                "tvl": 0,
                "volume": 0,
                "apr": 0,
                "prediction": 0
            }
        
        categories[category]["count"] += 1
        categories[category]["tvl"] += pool["tvl"]
        categories[category]["volume"] += pool["volume24h"]
        categories[category]["apr"] += pool["apr"]["current"]
        categories[category]["prediction"] += pool["prediction_score"]
    
    # Calculate averages
    for category, data in categories.items():
        count = data["count"]
        data["avg_apr"] = data["apr"] / count
        data["avg_prediction"] = data["prediction"] / count
    
    # Convert to DataFrame for display
    category_data = []
    for category, data in categories.items():
        category_data.append({
            "Category": category,
            "Pool Count": data["count"],
            "Total TVL": format_currency(data["tvl"]),
            "Total Volume": format_currency(data["volume"]),
            "Avg APR": format_percentage(data["avg_apr"]),
            "Avg Prediction": f"{data['avg_prediction']:.1f}/100"
        })
    
    st.table(pd.DataFrame(category_data))
    
    # System Description
    st.subheader("How Our System Works")
    st.write("""
    ### Data Collection
    Our system automatically collects data for hundreds of Solana liquidity pools from multiple DEXes.
    We track not just the major pairs but also trending tokens, meme coins, gaming tokens, and more.
    
    ### Historical Analysis
    We store historical data points for each pool:
    - TVL changes over 24h, 7d, and 30d periods
    - APR changes over 24h, 7d, and 30d periods
    - Volume trends
    - Liquidity provider behavior
    
    ### Machine Learning Predictions
    Our ML models analyze this historical data to predict:
    - Which pools are likely to increase in TVL
    - Which pools may offer higher APR in the near future
    - Risk factors for different pool categories
    
    ### User Benefits
    - Identify emerging opportunities before they become widely known
    - Track historical performance across a wide range of pools
    - Get data-driven insights for liquidity provision strategies
    """)

if __name__ == "__main__":
    main()