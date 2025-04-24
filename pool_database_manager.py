import os
import sys
import psycopg2
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import json
import time
import random

# Default pool data for Solana DEXes
DEFAULT_POOLS = [
    {
        "id": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
        "name": "SOL/USDC",
        "token1Symbol": "SOL",
        "token2Symbol": "USDC",
        "token1Address": "So11111111111111111111111111111111111111112",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Raydium",
        "version": "v4"
    },
    {
        "id": "7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX",
        "name": "SOL/USDT",
        "token1Symbol": "SOL",
        "token2Symbol": "USDT",
        "token1Address": "So11111111111111111111111111111111111111112",
        "token2Address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "dex": "Raydium",
        "version": "v4"
    },
    {
        "id": "AVs9TA4nWDzfPJE9gGVNJMVhcQy3V9PGazuz33BfG2RA",
        "name": "SOL/RAY",
        "token1Symbol": "SOL",
        "token2Symbol": "RAY",
        "token1Address": "So11111111111111111111111111111111111111112",
        "token2Address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "dex": "Raydium",
        "version": "v4"
    },
    {
        "id": "HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ",
        "name": "SOL/USDC",
        "token1Symbol": "SOL",
        "token2Symbol": "USDC",
        "token1Address": "So11111111111111111111111111111111111111112",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Orca",
        "version": "Whirlpool"
    },
    {
        "id": "4fuUiYxTQ6QCrdSq9ouBYXLTyfqjfLqkEEV8eZbGG7h1",
        "name": "SOL/USDT",
        "token1Symbol": "SOL",
        "token2Symbol": "USDT",
        "token1Address": "So11111111111111111111111111111111111111112",
        "token2Address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "dex": "Orca",
        "version": "Whirlpool"
    },
    {
        "id": "6UmmUiYoBjSrhakAobJw8BvkmJtDVxaeBtbt7rxWo1mg",
        "name": "RAY/USDC",
        "token1Symbol": "RAY",
        "token2Symbol": "USDC",
        "token1Address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Raydium",
        "version": "v4"
    },
    {
        "id": "DVa7Qmb5ct9RCpaU7UTpSaf3GVMYz17vNVU67XpdCRut",
        "name": "RAY/USDT",
        "token1Symbol": "RAY",
        "token2Symbol": "USDT",
        "token1Address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "token2Address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "dex": "Raydium",
        "version": "v4"
    },
    {
        "id": "7P5Thr9Egi2rvMmEuQkLn8x8e8Qro7u2U7yLD2tU2Hbe",
        "name": "RAY/SRM",
        "token1Symbol": "RAY",
        "token2Symbol": "SRM",
        "token1Address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "token2Address": "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt",
        "dex": "Raydium",
        "version": "v4"
    },
    {
        "id": "2LQdMz7YXqRwBUv3oNm8oEvs3tX3ASXhUAP3apPMYXeR",
        "name": "MNGO/USDC",
        "token1Symbol": "MNGO",
        "token2Symbol": "USDC",
        "token1Address": "MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Raydium",
        "version": "v4"
    },
    {
        "id": "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",
        "name": "BTC/USDC",
        "token1Symbol": "BTC",
        "token2Symbol": "USDC",
        "token1Address": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Raydium",
        "version": "v4"
    },
    {
        "id": "7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX",
        "name": "ETH/USDC",
        "token1Symbol": "ETH",
        "token2Symbol": "USDC",
        "token1Address": "2FPyTwcZLUg1MDrwsyoP4D6s1tM7hAkHYRjkNb5w6Pxk",
        "token2Address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "dex": "Raydium",
        "version": "v4"
    }
]

# Random metric generators for simulation
def random_liquidity():
    return round(random.uniform(50000, 10000000), 2)

def random_volume_24h():
    return round(random.uniform(10000, 2000000), 2)

def random_apr():
    return round(random.uniform(0.5, 30), 2)

def random_fee():
    return round(random.uniform(0.01, 1), 2)

def connect_to_db():
    """Connect to PostgreSQL database"""
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            st.error("DATABASE_URL environment variable is not set")
            st.code("Current environment vars: " + ", ".join([k for k in os.environ.keys() if not k.startswith('_')]))
            return None
            
        st.info(f"Connecting to database with URL: {db_url.split('@')[0]}@...")
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        st.error("Check that the PostgreSQL database is running and accessible")
        return None

def init_db():
    """Initialize database tables if they don't exist"""
    conn = connect_to_db()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Create pool_data table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pool_data (
            pool_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            token1_symbol VARCHAR(50) NOT NULL,
            token2_symbol VARCHAR(50) NOT NULL,
            token1_address VARCHAR(255) NOT NULL,
            token2_address VARCHAR(255) NOT NULL,
            dex VARCHAR(100) NOT NULL,
            version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create pool_metrics table for historical data
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pool_metrics (
            id SERIAL PRIMARY KEY,
            pool_id VARCHAR(255) REFERENCES pool_data(pool_id),
            liquidity NUMERIC(20, 2) NOT NULL,
            volume_24h NUMERIC(20, 2) NOT NULL,
            apr NUMERIC(10, 2) NOT NULL,
            fee_24h NUMERIC(10, 2),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create tokens table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            address VARCHAR(255) PRIMARY KEY,
            symbol VARCHAR(50) NOT NULL,
            name VARCHAR(255),
            decimals INTEGER DEFAULT 9,
            chain VARCHAR(50) DEFAULT 'solana',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create token_prices table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS token_prices (
            id SERIAL PRIMARY KEY,
            token_address VARCHAR(255) REFERENCES tokens(address),
            price_usd NUMERIC(20, 6) NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        st.success("Database tables created successfully")
        return True
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def add_pool(pool_data):
    """Add a pool to the database"""
    conn = connect_to_db()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # First, make sure tokens exist in tokens table
        for token_prefix, symbol, address in [
            ("token1", pool_data["token1Symbol"], pool_data["token1Address"]),
            ("token2", pool_data["token2Symbol"], pool_data["token2Address"])
        ]:
            # Check if token exists
            cur.execute("SELECT * FROM tokens WHERE address = %s", (address,))
            if cur.fetchone() is None:
                # Add token if it doesn't exist
                cur.execute(
                    "INSERT INTO tokens (address, symbol, name) VALUES (%s, %s, %s)",
                    (address, symbol, symbol)
                )
        
        # Insert pool data
        cur.execute("""
        INSERT INTO pool_data (
            pool_id, name, token1_symbol, token2_symbol, 
            token1_address, token2_address, dex, version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (pool_id) DO UPDATE SET
            name = EXCLUDED.name,
            token1_symbol = EXCLUDED.token1_symbol,
            token2_symbol = EXCLUDED.token2_symbol,
            token1_address = EXCLUDED.token1_address,
            token2_address = EXCLUDED.token2_address,
            dex = EXCLUDED.dex,
            version = EXCLUDED.version
        """,
        (
            pool_data["id"],
            pool_data["name"],
            pool_data["token1Symbol"],
            pool_data["token2Symbol"],
            pool_data["token1Address"],
            pool_data["token2Address"],
            pool_data["dex"],
            pool_data["version"]
        ))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error adding pool: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def add_pool_metrics(pool_id, metrics):
    """Add metrics for a pool"""
    conn = connect_to_db()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Check if pool exists
        cur.execute("SELECT * FROM pool_data WHERE pool_id = %s", (pool_id,))
        if cur.fetchone() is None:
            st.error(f"Pool {pool_id} does not exist")
            return False
        
        # Insert metrics
        cur.execute("""
        INSERT INTO pool_metrics (
            pool_id, liquidity, volume_24h, apr, fee_24h, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            pool_id,
            metrics["liquidity"],
            metrics["volume_24h"],
            metrics["apr"],
            metrics.get("fee_24h", 0),
            metrics.get("timestamp", datetime.now())
        ))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error adding pool metrics: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_all_pools():
    """Get all pools from database"""
    conn = connect_to_db()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM pool_data ORDER BY name")
        
        columns = [desc[0] for desc in cur.description]
        pools = []
        
        for row in cur.fetchall():
            pool = dict(zip(columns, row))
            pools.append(pool)
        
        return pools
    except Exception as e:
        st.error(f"Error getting pools: {str(e)}")
        return []
    finally:
        conn.close()

def get_pool_by_id(pool_id):
    """Get pool details by ID"""
    conn = connect_to_db()
    if not conn:
        return None
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM pool_data WHERE pool_id = %s", (pool_id,))
        
        columns = [desc[0] for desc in cur.description]
        row = cur.fetchone()
        
        if row:
            pool = dict(zip(columns, row))
            return pool
        else:
            return None
    except Exception as e:
        st.error(f"Error getting pool details: {str(e)}")
        return None
    finally:
        conn.close()

def get_pool_metrics(pool_id, days=30):
    """Get pool metrics for the last X days"""
    conn = connect_to_db()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cur.execute("""
        SELECT * FROM pool_metrics 
        WHERE pool_id = %s AND timestamp >= %s
        ORDER BY timestamp
        """, (pool_id, cutoff_date))
        
        columns = [desc[0] for desc in cur.description]
        metrics = []
        
        for row in cur.fetchall():
            metric = dict(zip(columns, row))
            metrics.append(metric)
        
        return metrics
    except Exception as e:
        st.error(f"Error getting pool metrics: {str(e)}")
        return []
    finally:
        conn.close()

def get_pools_by_token(token_symbol):
    """Get pools containing a specific token"""
    conn = connect_to_db()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT * FROM pool_data 
        WHERE token1_symbol = %s OR token2_symbol = %s
        ORDER BY name
        """, (token_symbol, token_symbol))
        
        columns = [desc[0] for desc in cur.description]
        pools = []
        
        for row in cur.fetchall():
            pool = dict(zip(columns, row))
            pools.append(pool)
        
        return pools
    except Exception as e:
        st.error(f"Error getting pools by token: {str(e)}")
        return []
    finally:
        conn.close()

def generate_random_metrics(pool_id, num_days=30):
    """Generate random historical metrics for a pool"""
    metrics = []
    end_date = datetime.now()
    
    # Generate metrics with slight variations for realism
    base_liquidity = random_liquidity()
    base_volume = random_volume_24h()
    base_apr = random_apr()
    
    for i in range(num_days):
        date = end_date - timedelta(days=num_days-i-1)
        
        # Add some variation day by day
        liquidity_factor = random.uniform(0.95, 1.05)
        volume_factor = random.uniform(0.85, 1.15)
        apr_factor = random.uniform(0.9, 1.1)
        
        metrics.append({
            "liquidity": round(base_liquidity * liquidity_factor, 2),
            "volume_24h": round(base_volume * volume_factor, 2),
            "apr": round(base_apr * apr_factor, 2),
            "fee_24h": round(base_volume * volume_factor * 0.003, 2),  # 0.3% fee typical
            "timestamp": date
        })
    
    return metrics

def add_default_pools():
    """Add default pools to the database"""
    success_count = 0
    fail_count = 0
    
    for pool in DEFAULT_POOLS:
        if add_pool(pool):
            success_count += 1
        else:
            fail_count += 1
    
    return success_count, fail_count

def show_pools():
    """Display all pools in the database"""
    pools = get_all_pools()
    
    if pools:
        # Convert to DataFrame for better display
        df = pd.DataFrame(pools)
        st.dataframe(df)
        
        # Show stats
        st.info(f"Total pools: {len(pools)}")
        
        # Count by DEX
        dex_counts = {}
        for pool in pools:
            dex = pool.get("dex", "Unknown")
            dex_counts[dex] = dex_counts.get(dex, 0) + 1
        
        st.write("Pools by DEX:")
        for dex, count in dex_counts.items():
            st.write(f"- {dex}: {count}")
    else:
        st.warning("No pools found in database")

def add_historical_metrics():
    """Add historical metrics for all pools"""
    pools = get_all_pools()
    
    if not pools:
        st.warning("No pools found in database")
        return 0
    
    success_count = 0
    
    for pool in pools:
        pool_id = pool.get("pool_id")
        
        # Generate random metrics
        metrics_list = generate_random_metrics(pool_id, num_days=30)
        
        # Add metrics to database
        for metrics in metrics_list:
            if add_pool_metrics(pool_id, metrics):
                success_count += 1
    
    return success_count

def main():
    st.title("Solana Liquidity Pool Database Manager")
    
    st.write("""
    This tool helps manage a local database of Solana liquidity pools.
    You can add new pools, explore pool data, and simulate metrics for analysis.
    """)
    
    # Check database connection status
    db_status = st.empty()
    st.subheader("Database Connection Status")
    
    # Display environment variables (without sensitive info)
    st.write("Environment Variables:")
    env_vars = {
        k: v if not any(s in k.lower() for s in ["password", "key", "secret", "token"]) else "***" 
        for k, v in os.environ.items() 
        if k.startswith(("PG", "DATABASE")) and not k.startswith("_")
    }
    
    if env_vars:
        st.json(json.dumps(env_vars, indent=2))
    else:
        st.warning("No database-related environment variables found")
    
    # Test connection
    conn = connect_to_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT current_database(), current_user, version()")
            db, user, version = cur.fetchone()
            st.success("✅ Database connection successful")
            st.info(f"Database: {db}, User: {user}")
            st.info(f"PostgreSQL version: {version}")
            conn.close()
        except Exception as e:
            st.error(f"⚠️ Error querying database: {str(e)}")
    else:
        st.error("⚠️ Database connection failed")
    
    # Database initialization
    st.subheader("Database Setup")
    
    if st.button("Initialize Database"):
        if init_db():
            st.success("Database initialized successfully")
        else:
            st.error("Database initialization failed")
    
    # Options in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Manage Pools", "View Pools", "Generate Metrics", "API Testing"])
    
    with tab1:
        st.subheader("Add Default Pools")
        if st.button("Add Default Pools to Database"):
            success, fail = add_default_pools()
            if success > 0:
                st.success(f"Successfully added {success} pools to database")
            if fail > 0:
                st.warning(f"Failed to add {fail} pools to database")
        
        st.subheader("Add Custom Pool")
        with st.form("add_pool_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pool_id = st.text_input("Pool ID", "")
                pool_name = st.text_input("Pool Name (e.g. SOL/USDC)", "")
                token1_symbol = st.text_input("Token 1 Symbol", "")
                token1_address = st.text_input("Token 1 Address", "")
            
            with col2:
                dex = st.selectbox("DEX", ["Raydium", "Orca", "Saber", "Atrix", "Other"])
                version = st.text_input("Version", "")
                token2_symbol = st.text_input("Token 2 Symbol", "")
                token2_address = st.text_input("Token 2 Address", "")
            
            submit_button = st.form_submit_button("Add Pool")
            
            if submit_button:
                if not pool_id or not pool_name or not token1_symbol or not token2_symbol:
                    st.error("Please fill in all required fields")
                else:
                    pool_data = {
                        "id": pool_id,
                        "name": pool_name,
                        "token1Symbol": token1_symbol,
                        "token2Symbol": token2_symbol,
                        "token1Address": token1_address or "unknown",
                        "token2Address": token2_address or "unknown",
                        "dex": dex,
                        "version": version
                    }
                    
                    if add_pool(pool_data):
                        st.success(f"Successfully added pool {pool_name}")
                    else:
                        st.error("Failed to add pool")
    
    with tab2:
        st.subheader("View All Pools")
        show_pools()
        
        st.subheader("Search Pools by Token")
        token_symbol = st.text_input("Enter Token Symbol (e.g. SOL, USDC)", "")
        
        if token_symbol and st.button("Search"):
            pools = get_pools_by_token(token_symbol.upper())
            
            if pools:
                st.success(f"Found {len(pools)} pools containing {token_symbol.upper()}")
                df = pd.DataFrame(pools)
                st.dataframe(df)
            else:
                st.warning(f"No pools found containing {token_symbol.upper()}")
        
        st.subheader("Pool Details")
        pools = get_all_pools()
        
        if pools:
            pool_names = [f"{p['name']} ({p['pool_id']})" for p in pools]
            selected_pool = st.selectbox("Select Pool", pool_names)
            
            if selected_pool and st.button("View Details"):
                pool_id = selected_pool.split("(")[1].split(")")[0]
                pool = get_pool_by_id(pool_id)
                
                if pool:
                    st.json(json.dumps(pool, indent=2, default=str))
                    
                    # Get and display metrics
                    metrics = get_pool_metrics(pool_id)
                    
                    if metrics:
                        st.subheader("Historical Metrics")
                        metrics_df = pd.DataFrame(metrics)
                        
                        # Convert timestamp to datetime for proper display
                        if 'timestamp' in metrics_df.columns:
                            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
                            
                        st.dataframe(metrics_df)
                        
                        # Create charts
                        # Show APR over time
                        st.subheader("APR Over Time")
                        chart_data = metrics_df[['timestamp', 'apr']].copy()
                        st.line_chart(chart_data.set_index('timestamp'))
                        
                        # Show liquidity over time
                        st.subheader("Liquidity Over Time")
                        chart_data = metrics_df[['timestamp', 'liquidity']].copy()
                        st.line_chart(chart_data.set_index('timestamp'))
                        
                        # Show volume over time
                        st.subheader("Volume Over Time")
                        chart_data = metrics_df[['timestamp', 'volume_24h']].copy()
                        st.line_chart(chart_data.set_index('timestamp'))
                    else:
                        st.warning("No metrics found for this pool")
                else:
                    st.error("Failed to get pool details")
    
    with tab3:
        st.subheader("Generate Simulated Metrics")
        st.write("""
        This will generate 30 days of simulated metrics for all pools in the database.
        This is useful for testing visualizations and analysis.
        """)
        
        if st.button("Generate Historical Metrics"):
            with st.spinner("Generating metrics..."):
                success_count = add_historical_metrics()
                st.success(f"Successfully added {success_count} sets of metrics")
        
        st.subheader("Generate Real-time Metrics")
        st.write("""
        This function simulates real-time metric updates for all pools.
        """)
        
        if st.button("Simulate Real-time Update"):
            pools = get_all_pools()
            
            if not pools:
                st.warning("No pools found in database")
            else:
                with st.spinner("Updating metrics..."):
                    success_count = 0
                    
                    for pool in pools:
                        pool_id = pool.get("pool_id")
                        
                        # Generate random metrics for today
                        metrics = {
                            "liquidity": random_liquidity(),
                            "volume_24h": random_volume_24h(),
                            "apr": random_apr(),
                            "fee_24h": random_fee(),
                            "timestamp": datetime.now()
                        }
                        
                        if add_pool_metrics(pool_id, metrics):
                            success_count += 1
                    
                    st.success(f"Updated metrics for {success_count} pools")
    
    with tab4:
        st.subheader("API Endpoint Testing")
        st.write("""
        This section allows testing API endpoint configurations to ensure they're working.
        """)
        
        api_url = st.text_input("API URL", os.getenv("RAYDIUM_API_URL", ""))
        api_key = st.text_input("API Key", os.getenv("RAYDIUM_API_KEY", ""), type="password")
        
        if st.button("Test Connection"):
            with st.spinner("Testing API connection..."):
                # Simple check to see if the URL is valid format
                if not api_url.startswith(("http://", "https://")):
                    st.error("Invalid API URL format. Must start with http:// or https://")
                else:
                    try:
                        import requests
                        
                        headers = {
                            "x-api-key": api_key,
                            "Content-Type": "application/json"
                        }
                        
                        response = requests.get(f"{api_url}/api/status", headers=headers, timeout=10)
                        
                        if response.status_code == 200:
                            st.success("API connection successful!")
                            st.json(response.json())
                        else:
                            st.warning(f"API returned status code: {response.status_code}")
                            st.text(response.text)
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
        
        # Option to update environment variables
        st.subheader("Update Environment Variables")
        
        with st.form("update_env"):
            new_api_url = st.text_input("New API URL", api_url)
            new_api_key = st.text_input("New API Key", api_key, type="password")
            
            if st.form_submit_button("Update Environment"):
                # Write to .env file
                with open(".env", "r") as f:
                    env_lines = f.readlines()
                
                with open(".env", "w") as f:
                    for line in env_lines:
                        if line.startswith("RAYDIUM_API_URL="):
                            f.write(f"RAYDIUM_API_URL={new_api_url}\n")
                        elif line.startswith("RAYDIUM_API_KEY="):
                            f.write(f"RAYDIUM_API_KEY={new_api_key}\n")
                        else:
                            f.write(line)
                
                st.success("Environment variables updated")
                st.info("Restart the application for changes to take effect")

if __name__ == "__main__":
    main()