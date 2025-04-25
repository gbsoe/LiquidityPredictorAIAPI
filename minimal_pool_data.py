import streamlit as st

# List of pool data formatted as dictionaries
POOL_DATA = [
    {
        "id": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
        "name": "SOL/USDC",
        "dex": "Raydium",
        "liquidity": 24532890.45,
        "volume24h": 8763021.32,
        "apr": 12.87
    },
    {
        "id": "7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX",
        "name": "SOL/USDT",
        "dex": "Raydium",
        "liquidity": 18245789.12,
        "volume24h": 6542891.45,
        "apr": 11.23
    },
    {
        "id": "AVs9TA4nWDzfPJE9gGVNJMVhcQy3V9PGazuz33BfG2RA",
        "name": "SOL/RAY",
        "dex": "Raydium",
        "liquidity": 5678234.89,
        "volume24h": 1987654.32,
        "apr": 15.42
    },
    {
        "id": "HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ",
        "name": "SOL/USDC",
        "dex": "Orca",
        "liquidity": 22345678.90,
        "volume24h": 7654321.09,
        "apr": 13.56
    },
    {
        "id": "4fuUiYxTQ6QCrdSq9ouBYXLTyfqjfLqkEEV8eZbGG7h1",
        "name": "SOL/USDT",
        "dex": "Orca",
        "liquidity": 17654321.23,
        "volume24h": 5432167.89,
        "apr": 12.78
    },
    {
        "id": "6UmmUiYoBjSrhakAobJw8BvkmJtDVxaeBtbt7rxWo1mg",
        "name": "RAY/USDC",
        "dex": "Raydium",
        "liquidity": 4321987.65,
        "volume24h": 1543219.87,
        "apr": 18.76
    },
    {
        "id": "DVa7Qmb5ct9RCpaU7UTpSaf3GVMYz17vNVU67XpdCRut",
        "name": "RAY/USDT",
        "dex": "Raydium",
        "liquidity": 3214567.89,
        "volume24h": 1234567.89,
        "apr": 17.89
    },
    {
        "id": "2LQdMz7YXqRwBUv3oNm8oEvs3tX3ASXhUAP3apPMYXeR",
        "name": "MNGO/USDC",
        "dex": "Raydium",
        "liquidity": 1987654.32,
        "volume24h": 765432.19,
        "apr": 21.34
    },
    {
        "id": "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",
        "name": "BTC/USDC",
        "dex": "Raydium",
        "liquidity": 45321987.65,
        "volume24h": 12345678.90,
        "apr": 9.87
    },
    {
        "id": "7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX",
        "name": "ETH/USDC",
        "dex": "Raydium",
        "liquidity": 32145678.90,
        "volume24h": 9876543.21,
        "apr": 10.23
    },
    {
        "id": "A2G5qS6C5KymjYA9K9QXfh66gXBXeQCCp2Du1nsMpAg9",
        "name": "SOL/mSOL",
        "dex": "Orca",
        "liquidity": 12345678.90,
        "volume24h": 3456789.01,
        "apr": 14.56
    },
    {
        "id": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
        "name": "JUP/USDC",
        "dex": "Jupiter",
        "liquidity": 8765432.10,
        "volume24h": 2345678.90,
        "apr": 19.87
    },
    {
        "id": "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K",
        "name": "BONK/USDC",
        "dex": "Meteora",
        "liquidity": 5432167.89,
        "volume24h": 1987654.32,
        "apr": 25.67
    },
    {
        "id": "Dooar9JkhdZ7J3LHN3A7YCuoGRUggXhQaG4kijfLGU2j",
        "name": "SAMO/USDC",
        "dex": "Raydium",
        "liquidity": 3456789.01,
        "volume24h": 876543.21,
        "apr": 22.45
    },
    {
        "id": "9X4JWzXhKiM6AbiYBmm58JcBxVXJiQ2hT5d4k5RQsLxZ",
        "name": "USDC/USDT",
        "dex": "Saber",
        "liquidity": 54321987.65,
        "volume24h": 8765432.10,
        "apr": 5.67
    }
]

def format_currency(value):
    return f"${value:,.2f}"

def format_percentage(value):
    return f"{value:.2f}%"

def main():
    st.title("Solana Liquidity Pool Data")
    
    st.write("""
    This viewer shows sample data from Solana liquidity pools across multiple DEXes.
    """)
    
    # Count total pools
    total_pools = len(POOL_DATA)
    
    # Calculate total liquidity across all pools
    total_liquidity = sum(pool["liquidity"] for pool in POOL_DATA)
    
    # Calculate total 24h volume
    total_volume = sum(pool["volume24h"] for pool in POOL_DATA)
    
    # Calculate average APR
    avg_apr = sum(pool["apr"] for pool in POOL_DATA) / total_pools
    
    # Show summary
    st.subheader("Summary")
    st.write(f"Total Pools: {total_pools}")
    st.write(f"Total Liquidity: {format_currency(total_liquidity)}")
    st.write(f"Total 24h Volume: {format_currency(total_volume)}")
    st.write(f"Average APR: {format_percentage(avg_apr)}")
    
    # Simple filtering by DEX
    st.subheader("Filter by DEX")
    dex_list = sorted(list(set(pool["dex"] for pool in POOL_DATA)))
    selected_dex = st.selectbox("Select DEX", ["All"] + dex_list)
    
    # Apply filter
    if selected_dex == "All":
        filtered_pools = POOL_DATA
    else:
        filtered_pools = [pool for pool in POOL_DATA if pool["dex"] == selected_dex]
    
    # Show filtered pools
    st.subheader(f"Pools ({len(filtered_pools)})")
    
    # Create a table for pool data
    table_data = []
    for pool in filtered_pools:
        table_data.append({
            "Name": pool["name"],
            "DEX": pool["dex"],
            "Liquidity": format_currency(pool["liquidity"]),
            "24h Volume": format_currency(pool["volume24h"]),
            "APR": format_percentage(pool["apr"]),
            "ID": pool["id"]
        })
    
    # Display table
    st.table(table_data)
    
    # Show detailed view for selected pool
    st.subheader("Pool Details")
    pool_names = [f"{pool['name']} ({pool['dex']})" for pool in filtered_pools]
    selected_pool_name = st.selectbox("Select Pool", pool_names)
    
    # Get the selected pool
    selected_idx = pool_names.index(selected_pool_name)
    selected_pool = filtered_pools[selected_idx]
    
    # Display pool details
    st.write(f"**ID:** {selected_pool['id']}")
    st.write(f"**Name:** {selected_pool['name']}")
    st.write(f"**DEX:** {selected_pool['dex']}")
    st.write(f"**Liquidity:** {format_currency(selected_pool['liquidity'])}")
    st.write(f"**24h Volume:** {format_currency(selected_pool['volume24h'])}")
    st.write(f"**APR:** {format_percentage(selected_pool['apr'])}")

if __name__ == "__main__":
    main()