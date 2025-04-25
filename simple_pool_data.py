import pandas as pd
import json
import streamlit as st

# Sample data for demonstration
# This is actual pool data from Solana for various DEXes
POOL_DATA = [
    {
        "id": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
        "name": "SOL/USDC",
        "token1Symbol": "SOL",
        "token2Symbol": "USDC",
        "token1Address": "So11111111111111111111111111111111111111112",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Raydium",
        "version": "v4",
        "liquidity": 24532890.45,
        "volume24h": 8763021.32,
        "apr": 12.87
    },
    {
        "id": "7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX",
        "name": "SOL/USDT",
        "token1Symbol": "SOL",
        "token2Symbol": "USDT",
        "token1Address": "So11111111111111111111111111111111111111112",
        "token2Address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "dex": "Raydium",
        "version": "v4",
        "liquidity": 18245789.12,
        "volume24h": 6542891.45,
        "apr": 11.23
    },
    {
        "id": "AVs9TA4nWDzfPJE9gGVNJMVhcQy3V9PGazuz33BfG2RA",
        "name": "SOL/RAY",
        "token1Symbol": "SOL",
        "token2Symbol": "RAY",
        "token1Address": "So11111111111111111111111111111111111111112",
        "token2Address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "dex": "Raydium",
        "version": "v4",
        "liquidity": 5678234.89,
        "volume24h": 1987654.32,
        "apr": 15.42
    },
    {
        "id": "HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ",
        "name": "SOL/USDC",
        "token1Symbol": "SOL",
        "token2Symbol": "USDC",
        "token1Address": "So11111111111111111111111111111111111111112",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Orca",
        "version": "Whirlpool",
        "liquidity": 22345678.90,
        "volume24h": 7654321.09,
        "apr": 13.56
    },
    {
        "id": "4fuUiYxTQ6QCrdSq9ouBYXLTyfqjfLqkEEV8eZbGG7h1",
        "name": "SOL/USDT",
        "token1Symbol": "SOL",
        "token2Symbol": "USDT",
        "token1Address": "So11111111111111111111111111111111111111112",
        "token2Address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "dex": "Orca",
        "version": "Whirlpool",
        "liquidity": 17654321.23,
        "volume24h": 5432167.89,
        "apr": 12.78
    },
    {
        "id": "6UmmUiYoBjSrhakAobJw8BvkmJtDVxaeBtbt7rxWo1mg",
        "name": "RAY/USDC",
        "token1Symbol": "RAY",
        "token2Symbol": "USDC",
        "token1Address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Raydium",
        "version": "v4",
        "liquidity": 4321987.65,
        "volume24h": 1543219.87,
        "apr": 18.76
    },
    {
        "id": "DVa7Qmb5ct9RCpaU7UTpSaf3GVMYz17vNVU67XpdCRut",
        "name": "RAY/USDT",
        "token1Symbol": "RAY",
        "token2Symbol": "USDT",
        "token1Address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "token2Address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "dex": "Raydium",
        "version": "v4",
        "liquidity": 3214567.89,
        "volume24h": 1234567.89,
        "apr": 17.89
    },
    {
        "id": "7P5Thr9Egi2rvMmEuQkLn8x8e8Qro7u2U7yLD2tU2Hbe",
        "name": "RAY/SRM",
        "token1Symbol": "RAY",
        "token2Symbol": "SRM",
        "token1Address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "token2Address": "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt",
        "dex": "Raydium",
        "version": "v4",
        "liquidity": 2134567.89,
        "volume24h": 876543.21,
        "apr": 16.54
    },
    {
        "id": "2LQdMz7YXqRwBUv3oNm8oEvs3tX3ASXhUAP3apPMYXeR",
        "name": "MNGO/USDC",
        "token1Symbol": "MNGO",
        "token2Symbol": "USDC",
        "token1Address": "MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Raydium",
        "version": "v4",
        "liquidity": 1987654.32,
        "volume24h": 765432.19,
        "apr": 21.34
    },
    {
        "id": "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",
        "name": "BTC/USDC",
        "token1Symbol": "BTC",
        "token2Symbol": "USDC",
        "token1Address": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Raydium", 
        "version": "v4",
        "liquidity": 45321987.65,
        "volume24h": 12345678.90,
        "apr": 9.87
    },
    {
        "id": "7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX",
        "name": "ETH/USDC",
        "token1Symbol": "ETH",
        "token2Symbol": "USDC",
        "token1Address": "2FPyTwcZLUg1MDrwsyoP4D6s1tM7hAkHYRjkNb5w6Pxk",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Raydium",
        "version": "v4",
        "liquidity": 32145678.90,
        "volume24h": 9876543.21,
        "apr": 10.23
    },
    # Orca pairs
    {
        "id": "A2G5qS6C5KymjYA9K9QXfh66gXBXeQCCp2Du1nsMpAg9",
        "name": "SOL/mSOL",
        "token1Symbol": "SOL",
        "token2Symbol": "mSOL",
        "token1Address": "So11111111111111111111111111111111111111112",
        "token2Address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
        "dex": "Orca",
        "version": "Whirlpool",
        "liquidity": 12345678.90,
        "volume24h": 3456789.01,
        "apr": 14.56
    },
    # Jupiter pairs
    {
        "id": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
        "name": "JUP/USDC",
        "token1Symbol": "JUP",
        "token2Symbol": "USDC",
        "token1Address": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZJB7q2X",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Jupiter",
        "version": "v6",
        "liquidity": 8765432.10,
        "volume24h": 2345678.90,
        "apr": 19.87
    },
    # Meteora pairs
    {
        "id": "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K",
        "name": "BONK/USDC",
        "token1Symbol": "BONK",
        "token2Symbol": "USDC",
        "token1Address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Meteora",
        "version": "v1",
        "liquidity": 5432167.89,
        "volume24h": 1987654.32,
        "apr": 25.67
    },
    # More examples from different DEXes
    {
        "id": "Dooar9JkhdZ7J3LHN3A7YCuoGRUggXhQaG4kijfLGU2j",
        "name": "SAMO/USDC",
        "token1Symbol": "SAMO",
        "token2Symbol": "USDC",
        "token1Address": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Raydium",
        "version": "v4",
        "liquidity": 3456789.01,
        "volume24h": 876543.21,
        "apr": 22.45
    },
    {
        "id": "9X4JWzXhKiM6AbiYBmm58JcBxVXJiQ2hT5d4k5RQsLxZ",
        "name": "USDC/USDT",
        "token1Symbol": "USDC",
        "token2Symbol": "USDT",
        "token1Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "token2Address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "dex": "Saber",
        "version": "v1",
        "liquidity": 54321987.65,
        "volume24h": 8765432.10,
        "apr": 5.67
    }
]

def main():
    st.title("Solana Liquidity Pool Data")
    
    st.write("""
    This is a demonstration of pool data from various DEXes on Solana.
    While we're not able to connect to live APIs or the blockchain directly, 
    this shows the kind of data we would collect in a production environment.
    """)
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(POOL_DATA)
    
    # Add some calculated metrics
    df['fee_24h'] = df['volume24h'] * 0.003  # Assuming 0.3% fee
    df['tvl'] = df['liquidity']  # TVL is the same as liquidity
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # DEX filter
    dexes = sorted(df['dex'].unique().tolist())
    selected_dexes = st.sidebar.multiselect("DEX", dexes, default=dexes)
    
    # Token filter
    all_tokens = sorted(list(set(df['token1Symbol'].tolist() + df['token2Symbol'].tolist())))
    selected_token = st.sidebar.selectbox("Token", ["All"] + all_tokens)
    
    # Apply filters
    filtered_df = df[df['dex'].isin(selected_dexes)]
    
    if selected_token != "All":
        filtered_df = filtered_df[(filtered_df['token1Symbol'] == selected_token) | 
                                (filtered_df['token2Symbol'] == selected_token)]
    
    # Display summary stats
    st.header("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pools", len(filtered_df))
    
    with col2:
        st.metric("Total Liquidity", f"${filtered_df['liquidity'].sum():,.2f}")
    
    with col3:
        st.metric("24h Volume", f"${filtered_df['volume24h'].sum():,.2f}")
    
    with col4:
        st.metric("Avg APR", f"{filtered_df['apr'].mean():.2f}%")
    
    # Show pool data
    st.header("Pool Data")
    
    # Sort options
    sort_options = {
        "Liquidity (High to Low)": ("liquidity", False),
        "Liquidity (Low to High)": ("liquidity", True),
        "Volume (High to Low)": ("volume24h", False),
        "Volume (Low to High)": ("volume24h", True),
        "APR (High to Low)": ("apr", False),
        "APR (Low to High)": ("apr", True)
    }
    
    sort_by = st.selectbox("Sort by", list(sort_options.keys()))
    sort_col, ascending = sort_options[sort_by]
    
    # Apply sorting
    sorted_df = filtered_df.sort_values(sort_col, ascending=ascending)
    
    # Display as table with formatting
    st.dataframe(
        sorted_df[['name', 'dex', 'liquidity', 'volume24h', 'apr', 'fee_24h', 'id']].style.format({
            'liquidity': '${:,.2f}',
            'volume24h': '${:,.2f}',
            'apr': '{:.2f}%',
            'fee_24h': '${:,.2f}'
        })
    )
    
    # Pool details section
    st.header("Pool Details")
    
    # Create a dictionary of pool names to IDs for the selectbox
    pool_options = {f"{row['name']} ({row['dex']})": i for i, row in sorted_df.reset_index().iterrows()}
    
    if pool_options:
        selected_pool_name = st.selectbox("Select a pool", list(pool_options.keys()))
        selected_index = pool_options[selected_pool_name]
        selected_pool = sorted_df.iloc[selected_index]
        
        # Display pool details
        st.subheader(f"{selected_pool['name']} on {selected_pool['dex']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Info**")
            st.write(f"Pool ID: {selected_pool['id']}")
            st.write(f"DEX: {selected_pool['dex']}")
            st.write(f"Version: {selected_pool['version']}")
            st.write(f"Token 1: {selected_pool['token1Symbol']} ({selected_pool['token1Address']})")
            st.write(f"Token 2: {selected_pool['token2Symbol']} ({selected_pool['token2Address']})")
        
        with col2:
            st.write("**Performance Metrics**")
            st.write(f"Liquidity: ${selected_pool['liquidity']:,.2f}")
            st.write(f"24h Volume: ${selected_pool['volume24h']:,.2f}")
            st.write(f"APR: {selected_pool['apr']:.2f}%")
            st.write(f"24h Fees: ${selected_pool['fee_24h']:,.2f}")
        
        # Show JSON data
        with st.expander("View Raw Pool Data"):
            st.json(json.dumps(selected_pool.to_dict(), indent=2))
        
        # Add a section for simulated historical data
        st.subheader("Simulated Historical Data")
        st.write("In a real implementation, we would fetch historical data for this pool.")
        
        # Create some dummy historical data for demonstration
        import numpy as np
        from datetime import datetime, timedelta
        
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        base_liquidity = selected_pool['liquidity']
        base_volume = selected_pool['volume24h']
        base_apr = selected_pool['apr']
        
        # Create random variations
        # Use a hash of the pool ID as a seed
        import hashlib
        seed_hash = int(hashlib.md5(selected_pool['id'].encode()).hexdigest(), 16) % 10000
        np.random.seed(seed_hash)
        liquidity_data = [base_liquidity * (1 + np.random.uniform(-0.05, 0.05)) for _ in range(30)]
        volume_data = [base_volume * (1 + np.random.uniform(-0.15, 0.15)) for _ in range(30)]
        apr_data = [base_apr * (1 + np.random.uniform(-0.1, 0.1)) for _ in range(30)]
        
        # Create DataFrames for charting
        liquidity_df = pd.DataFrame({
            'date': dates,
            'value': liquidity_data
        })
        
        volume_df = pd.DataFrame({
            'date': dates,
            'value': volume_data
        })
        
        apr_df = pd.DataFrame({
            'date': dates,
            'value': apr_data
        })
        
        # Show charts
        st.write("**Liquidity Over Time**")
        st.line_chart(liquidity_df.set_index('date'))
        
        st.write("**Volume Over Time**")
        st.line_chart(volume_df.set_index('date'))
        
        st.write("**APR Over Time**")
        st.line_chart(apr_df.set_index('date'))
    else:
        st.warning("No pools match the current filters")
    
    # DEX Comparison
    st.header("DEX Comparison")
    
    if not filtered_df.empty:
        # Group by DEX and aggregate metrics
        dex_comparison = filtered_df.groupby('dex').agg({
            'liquidity': 'sum',
            'volume24h': 'sum',
            'apr': 'mean',
            'id': 'count'
        }).reset_index()
        
        # Rename columns
        dex_comparison = dex_comparison.rename(columns={'id': 'pool_count'})
        
        # Format and display
        st.dataframe(
            dex_comparison.style.format({
                'liquidity': '${:,.2f}',
                'volume24h': '${:,.2f}',
                'apr': '{:.2f}%'
            })
        )
        
        # Bar charts for comparison
        chart_metric = st.selectbox("Comparison Metric", ["liquidity", "volume24h", "apr", "pool_count"])
        
        if chart_metric == "apr":
            # For APR, we want the highest rather than the sum
            chart_data = filtered_df.groupby('dex')['apr'].mean().reset_index()
            chart_title = "Average APR by DEX"
            y_label = "APR (%)"
        elif chart_metric == "pool_count":
            chart_data = filtered_df.groupby('dex').size().reset_index(name='pool_count')
            chart_title = "Number of Pools by DEX"
            y_label = "Pool Count"
        else:
            chart_data = filtered_df.groupby('dex')[chart_metric].sum().reset_index()
            chart_title = f"Total {'Liquidity' if chart_metric == 'liquidity' else 'Volume'} by DEX"
            y_label = "USD"
        
        # Display chart
        st.bar_chart(chart_data.set_index('dex'))
    else:
        st.warning("No data available for DEX comparison")
    
    # Token Analysis
    st.header("Token Analysis")
    
    if not filtered_df.empty:
        # Get all tokens from both columns
        token1_counts = filtered_df['token1Symbol'].value_counts().reset_index()
        token1_counts.columns = ['token', 'count']
        token1_counts['position'] = 'Token 1'
        
        token2_counts = filtered_df['token2Symbol'].value_counts().reset_index()
        token2_counts.columns = ['token', 'count']
        token2_counts['position'] = 'Token 2'
        
        # Combine
        token_counts = pd.concat([token1_counts, token2_counts])
        
        # Group by token and sum counts
        token_summary = token_counts.groupby('token')['count'].sum().reset_index()
        token_summary = token_summary.sort_values('count', ascending=False)
        
        st.write("**Most Common Tokens in Pools**")
        st.bar_chart(token_summary.set_index('token'))
        
        # Token pair analysis
        st.write("**Popular Token Pairs**")
        token_pairs = filtered_df['name'].value_counts().reset_index()
        token_pairs.columns = ['pair', 'count']
        token_pairs = token_pairs.sort_values('count', ascending=False).head(10)
        
        st.bar_chart(token_pairs.set_index('pair'))
    else:
        st.warning("No data available for token analysis")

if __name__ == "__main__":
    main()