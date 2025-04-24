import os
import sys
import pandas as pd
from data_ingestion.raydium_api_client import RaydiumAPIClient
import streamlit as st
import json

def main():
    st.title("Raydium Pool Data Test")
    
    # Print current environment variables for debugging
    st.subheader("Environment Configuration")
    
    raydium_api_url = os.getenv("RAYDIUM_API_URL")
    raydium_api_key = os.getenv("RAYDIUM_API_KEY")
    
    if not raydium_api_url:
        st.warning("RAYDIUM_API_URL environment variable is not set")
    else:
        st.success(f"RAYDIUM_API_URL is set: {raydium_api_url}")
    
    if not raydium_api_key:
        st.warning("RAYDIUM_API_KEY environment variable is not set")
    else:
        # Hide actual key but show that it exists
        st.success("RAYDIUM_API_KEY is set (hidden for security)")
    
    # Initialize the API client
    st.subheader("API Client Initialization")
    
    # Allow user to input API credentials if not set in environment
    with st.form("api_credentials"):
        input_api_url = st.text_input("API URL (if not set in environment)", "https://api.raydium.io")
        input_api_key = st.text_input("API Key (if not set in environment)", type="password")
        submitted = st.form_submit_button("Initialize API Client")
        
        if submitted:
            api_url = input_api_url if input_api_url else raydium_api_url
            api_key = input_api_key if input_api_key else raydium_api_key
            
            # Initialize client with user input or environment variables
            client = RaydiumAPIClient(api_key=api_key, base_url=api_url)
            st.session_state['client'] = client
            st.success("API client initialized")
    
    # If client is initialized, fetch and display pool data
    if 'client' in st.session_state:
        client = st.session_state['client']
        
        st.subheader("Fetch All Pools")
        fetch_all = st.button("Get All Pools")
        
        if fetch_all:
            with st.spinner("Fetching all pools..."):
                try:
                    pools = client.get_all_pools()
                    if pools:
                        st.success(f"Retrieved {len(pools)} pools in total")
                        st.write("First 10 pools:")
                        st.json(json.dumps(pools[:10], indent=2))
                    else:
                        st.error("No pools found or error fetching pools")
                except Exception as e:
                    st.error(f"Error fetching pools: {str(e)}")
        
        st.subheader("Filter Pools by Token Symbol")
        token_symbols = ["SOL", "USDC", "USDT", "RAY"]
        selected_token = st.selectbox("Select token to filter by", token_symbols)
        
        fetch_by_token = st.button(f"Fetch pools containing {selected_token}")
        
        if fetch_by_token:
            with st.spinner(f"Fetching pools with {selected_token}..."):
                try:
                    pools = client.get_filtered_pools(token_symbol=selected_token, limit=50)
                    if pools:
                        st.success(f"Found {len(pools)} pools containing {selected_token}")
                        
                        # Convert to DataFrame for better display
                        pools_df = pd.DataFrame(pools)
                        st.dataframe(pools_df)
                        
                        # Extract the specific pool pairs we want
                        if selected_token == "SOL":
                            st.subheader("SOL/USDC, SOL/USDT, and SOL/RAY Pools")
                            sol_usdc = [p for p in pools if "SOL/USDC" in p.get("name", "")]
                            sol_usdt = [p for p in pools if "SOL/USDT" in p.get("name", "")]
                            sol_ray = [p for p in pools if "SOL/RAY" in p.get("name", "")]
                            
                            st.write("SOL/USDC Pools:")
                            if sol_usdc:
                                st.json(json.dumps(sol_usdc, indent=2))
                            else:
                                st.info("No SOL/USDC pools found")
                                
                            st.write("SOL/USDT Pools:")
                            if sol_usdt:
                                st.json(json.dumps(sol_usdt, indent=2))
                            else:
                                st.info("No SOL/USDT pools found")
                                
                            st.write("SOL/RAY Pools:")
                            if sol_ray:
                                st.json(json.dumps(sol_ray, indent=2))
                            else:
                                st.info("No SOL/RAY pools found")
                    else:
                        st.error(f"No pools found containing {selected_token}")
                except Exception as e:
                    st.error(f"Error filtering pools: {str(e)}")
        
        st.subheader("Get Pool Details by ID")
        pool_id = st.text_input("Enter Pool ID")
        
        get_details = st.button("Get Pool Details")
        
        if get_details and pool_id:
            with st.spinner(f"Fetching details for pool {pool_id}..."):
                try:
                    pool_details = client.get_pool_by_id(pool_id)
                    if pool_details:
                        st.success(f"Retrieved details for pool: {pool_id}")
                        st.json(json.dumps(pool_details, indent=2))
                        
                        # Also get metrics
                        st.subheader("Pool Metrics")
                        metrics = client.get_pool_metrics(pool_id)
                        if metrics:
                            st.success(f"Retrieved metrics for pool: {pool_id}")
                            st.json(json.dumps(metrics, indent=2))
                        else:
                            st.warning("No metrics found for this pool")
                    else:
                        st.error(f"No details found for pool {pool_id}")
                except Exception as e:
                    st.error(f"Error fetching pool details: {str(e)}")
        
        st.subheader("Direct Pool Search")
        st.write("Search for SOL/USDC, SOL/USDT, and SOL/RAY pools")
        
        search_pairs = st.button("Search for Specific Pool Pairs")
        
        if search_pairs:
            with st.spinner("Searching for specific pool pairs..."):
                # Use a more direct approach to find the pools
                pairs_to_find = ["SOL/USDC", "SOL/USDT", "SOL/RAY"]
                results = {}
                
                for pair in pairs_to_find:
                    token1, token2 = pair.split('/')
                    
                    # Try finding with first token
                    try:
                        pools1 = client.get_filtered_pools(token_symbol=token1, limit=100)
                        # Filter for the exact pair
                        matching_pools = [p for p in pools1 if p.get("name", "") == pair]
                        
                        if matching_pools:
                            results[pair] = matching_pools
                        else:
                            # Try with second token if not found
                            pools2 = client.get_filtered_pools(token_symbol=token2, limit=100)
                            matching_pools = [p for p in pools2 if p.get("name", "") == pair]
                            
                            if matching_pools:
                                results[pair] = matching_pools
                            else:
                                results[pair] = []
                    except Exception as e:
                        st.error(f"Error searching for {pair}: {str(e)}")
                        results[pair] = []
                
                # Display results
                for pair, pools in results.items():
                    st.subheader(f"{pair} Pools")
                    if pools:
                        st.success(f"Found {len(pools)} {pair} pools")
                        st.json(json.dumps(pools, indent=2))
                        
                        # If we have the pool IDs, we can get more detailed information
                        if len(pools) > 0:
                            pool_id = pools[0].get("id")
                            if pool_id:
                                st.write(f"Getting detailed information for first {pair} pool (ID: {pool_id})")
                                try:
                                    pool_details = client.get_pool_by_id(pool_id)
                                    if pool_details:
                                        st.write(f"Detailed information for {pair}:")
                                        st.json(json.dumps(pool_details, indent=2))
                                except Exception as e:
                                    st.error(f"Error getting details for {pair} pool: {str(e)}")
                    else:
                        st.info(f"No {pair} pools found")

if __name__ == "__main__":
    main()