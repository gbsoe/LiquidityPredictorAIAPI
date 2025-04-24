import requests
import json
import re
import sys
import os
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import trafilatura
from data_ingestion.raydium_api_client import RaydiumAPIClient

def scrape_raydium_pools():
    """
    Scrape pool IDs from Raydium's liquidity pools page
    """
    url = "https://raydium.io/liquidity-pools/"
    st.write(f"Attempting to scrape data from {url}")
    
    try:
        # First try using requests to get the raw HTML
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        st.write("Successfully connected to the website")
        
        # Use Beautiful Soup to parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for a script tag that might contain pool data as JSON
        scripts = soup.find_all('script')
        pool_data = []
        
        for script in scripts:
            if script.string and 'window.__NUXT__' in script.string:
                st.write("Found NUXT script data")
                # Extract JSON data from the script tag
                json_str = re.search(r'window\.__NUXT__\s*=\s*(\{.*\})', script.string)
                if json_str:
                    nuxt_data = json.loads(json_str.group(1))
                    # Navigate through the nuxt data to find pools
                    # This will require inspection of the actual data structure
                    st.json(json.dumps(nuxt_data, indent=2)[:1000] + "...")  # Show a preview
                    
                    # Attempt to extract pool data (this will need adjusting based on actual structure)
                    if 'state' in nuxt_data and 'pools' in nuxt_data['state']:
                        pool_data = nuxt_data['state']['pools']
                        st.success(f"Found {len(pool_data)} pools in NUXT data")
        
        # If we couldn't find pools in the scripts, try looking for data in the HTML
        if not pool_data:
            st.write("Trying to find pool data in HTML elements...")
            
            # Look for table rows or divs that might contain pool information
            pool_elements = soup.select('.pool-item, .liquidity-pool-row, tr.pool')  # Adjust selectors based on actual page
            
            if pool_elements:
                st.success(f"Found {len(pool_elements)} pool elements in HTML")
                
                # Extract data from these elements
                for element in pool_elements[:5]:  # Show first 5 as examples
                    st.code(element.prettify())
            else:
                st.warning("Could not find pool elements in HTML with our selectors")
        
        # Last resort: use trafilatura to get the cleaned text
        st.write("Using trafilatura to extract text content...")
        downloaded = trafilatura.fetch_url(url)
        text_content = trafilatura.extract(downloaded)
        
        if text_content:
            st.success("Successfully extracted text content with trafilatura")
            
            # Look for patterns that might indicate pool IDs
            # Example: SOL-USDC pool with ID pattern
            pool_id_matches = re.findall(r'(SOL[/-]USDC|SOL[/-]USDT|SOL[/-]RAY).*?([A-Za-z0-9]{32,})', text_content)
            
            if pool_id_matches:
                st.success(f"Found {len(pool_id_matches)} potential pool ID matches")
                st.write(pool_id_matches)
            else:
                st.warning("No pool ID patterns found in text content")
                
            # Show a sample of the text content
            st.text_area("Sample of extracted text", text_content[:1000] + "...", height=200)
        
        return pool_data
    
    except Exception as e:
        st.error(f"Error scraping Raydium website: {str(e)}")
        return []

def scrape_raydium_api():
    """
    Alternative approach: Use Raydium's API to get pool data
    """
    st.subheader("Attempting to get pool data from Raydium API")
    
    api_url = os.getenv("RAYDIUM_API_URL", "https://api.raydium.io")
    api_key = os.getenv("RAYDIUM_API_KEY", "")
    
    with st.form("api_credentials"):
        input_api_url = st.text_input("API URL", api_url)
        input_api_key = st.text_input("API Key (if required)", api_key, type="password")
        submitted = st.form_submit_button("Initialize API Client")
        
        if submitted:
            client = RaydiumAPIClient(api_key=input_api_key, base_url=input_api_url)
            
            # Try getting all pools
            with st.spinner("Fetching all pools from API..."):
                try:
                    pools = client.get_all_pools()
                    if pools:
                        st.success(f"Retrieved {len(pools)} pools from API")
                        
                        # Find SOL/USDC, SOL/USDT, and SOL/RAY pools
                        sol_usdc = [p for p in pools if p.get("name", "").startswith("SOL/USDC")]
                        sol_usdt = [p for p in pools if p.get("name", "").startswith("SOL/USDT")]
                        sol_ray = [p for p in pools if p.get("name", "").startswith("SOL/RAY")]
                        
                        # Display found pools
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"SOL/USDC Pools: {len(sol_usdc)}")
                            if sol_usdc:
                                st.dataframe(pd.DataFrame(sol_usdc))
                        
                        with col2:
                            st.write(f"SOL/USDT Pools: {len(sol_usdt)}")
                            if sol_usdt:
                                st.dataframe(pd.DataFrame(sol_usdt))
                        
                        with col3:
                            st.write(f"SOL/RAY Pools: {len(sol_ray)}")
                            if sol_ray:
                                st.dataframe(pd.DataFrame(sol_ray))
                        
                        return {
                            "SOL/USDC": sol_usdc,
                            "SOL/USDT": sol_usdt,
                            "SOL/RAY": sol_ray
                        }
                    else:
                        st.error("No pools returned from API")
                        
                        # Try filtering by token symbol
                        st.write("Trying to filter by token symbol...")
                        sol_pools = client.get_filtered_pools(token_symbol="SOL", limit=100)
                        
                        if sol_pools:
                            st.success(f"Retrieved {len(sol_pools)} SOL pools from API")
                            
                            # Find our target pairs
                            sol_usdc = [p for p in sol_pools if "USDC" in p.get("name", "")]
                            sol_usdt = [p for p in sol_pools if "USDT" in p.get("name", "")]
                            sol_ray = [p for p in sol_pools if "RAY" in p.get("name", "")]
                            
                            # Display found pools
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"SOL/USDC Pools: {len(sol_usdc)}")
                                if sol_usdc:
                                    st.dataframe(pd.DataFrame(sol_usdc))
                            
                            with col2:
                                st.write(f"SOL/USDT Pools: {len(sol_usdt)}")
                                if sol_usdt:
                                    st.dataframe(pd.DataFrame(sol_usdt))
                            
                            with col3:
                                st.write(f"SOL/RAY Pools: {len(sol_ray)}")
                                if sol_ray:
                                    st.dataframe(pd.DataFrame(sol_ray))
                            
                            return {
                                "SOL/USDC": sol_usdc,
                                "SOL/USDT": sol_usdt,
                                "SOL/RAY": sol_ray
                            }
                        else:
                            st.error("No SOL pools returned from API")
                except Exception as e:
                    st.error(f"Error fetching pools from API: {str(e)}")
    
    return {}

def try_solscan():
    """
    Try to get pool data from Solscan API
    """
    st.subheader("Attempting to get pool data from Solscan")
    
    # Solscan API endpoints
    base_url = "https://public-api.solscan.io"
    
    # Try to get pools from Solscan
    try:
        # Search for Raydium pools
        search_url = f"{base_url}/token/search?keyword=raydium%20lp"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        with st.spinner("Searching for Raydium LP tokens on Solscan..."):
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            search_results = response.json()
            
            if search_results and 'data' in search_results:
                st.success(f"Found {len(search_results['data'])} results from Solscan")
                
                # Look for SOL pairs
                sol_pairs = []
                for token in search_results['data']:
                    token_name = token.get('name', '').upper()
                    symbol = token.get('symbol', '').upper()
                    
                    if 'SOL' in token_name and ('USDC' in token_name or 'USDT' in token_name or 'RAY' in token_name):
                        sol_pairs.append(token)
                    elif 'SOL' in symbol and ('USDC' in symbol or 'USDT' in symbol or 'RAY' in symbol):
                        sol_pairs.append(token)
                
                if sol_pairs:
                    st.success(f"Found {len(sol_pairs)} SOL pairs on Solscan")
                    st.dataframe(pd.DataFrame(sol_pairs))
                    return sol_pairs
                else:
                    st.warning("No SOL pairs found in Solscan results")
                    # Show all results
                    st.dataframe(pd.DataFrame(search_results['data']))
            else:
                st.error("No results returned from Solscan or unexpected response format")
    
    except Exception as e:
        st.error(f"Error accessing Solscan API: {str(e)}")
    
    return []

def main():
    st.title("Raydium Pool ID Scraper")
    
    st.write("""
    This tool attempts to retrieve pool IDs for SOL/USDC, SOL/USDT, and SOL/RAY liquidity pools 
    on Raydium using several different methods.
    """)
    
    method = st.radio(
        "Select method to try",
        ["Web Scraping", "Raydium API", "Solscan API"]
    )
    
    if st.button("Run Selected Method"):
        if method == "Web Scraping":
            pools = scrape_raydium_pools()
            
            if pools:
                st.success("Successfully retrieved pool data via web scraping")
                st.json(json.dumps(pools, indent=2))
            else:
                st.warning("Web scraping did not return pool data")
        
        elif method == "Raydium API":
            pools = scrape_raydium_api()
            
            if pools:
                st.success("Successfully retrieved pool data via Raydium API")
                
                # Store results in session state for later use
                st.session_state['pool_ids'] = {
                    pair: [p.get('id') for p in pool_list] 
                    for pair, pool_list in pools.items()
                }
                
                # Display pool IDs
                st.subheader("Pool IDs Retrieved")
                for pair, ids in st.session_state['pool_ids'].items():
                    st.write(f"{pair}: {', '.join(ids) if ids else 'None found'}")
            else:
                st.warning("Raydium API did not return pool data")
        
        elif method == "Solscan API":
            pools = try_solscan()
            
            if pools:
                st.success("Successfully retrieved pool data via Solscan API")
                
                # Try to extract and organize pool IDs
                sol_usdc = [p for p in pools if 'SOL' in p.get('symbol', '').upper() and 'USDC' in p.get('symbol', '').upper()]
                sol_usdt = [p for p in pools if 'SOL' in p.get('symbol', '').upper() and 'USDT' in p.get('symbol', '').upper()]
                sol_ray = [p for p in pools if 'SOL' in p.get('symbol', '').upper() and 'RAY' in p.get('symbol', '').upper()]
                
                # Store results
                st.session_state['pool_ids'] = {
                    "SOL/USDC": [p.get('address') for p in sol_usdc],
                    "SOL/USDT": [p.get('address') for p in sol_usdt],
                    "SOL/RAY": [p.get('address') for p in sol_ray]
                }
                
                # Display pool IDs
                st.subheader("Pool IDs Retrieved")
                for pair, ids in st.session_state['pool_ids'].items():
                    st.write(f"{pair}: {', '.join(ids) if ids else 'None found'}")
            else:
                st.warning("Solscan API did not return pool data")
    
    # Section to manually input known pool IDs
    st.subheader("Manually Input Known Pool IDs")
    st.write("If you have pool IDs from other sources, you can input them here")
    
    with st.form("manual_pool_ids"):
        sol_usdc_id = st.text_input("SOL/USDC Pool ID", "")
        sol_usdt_id = st.text_input("SOL/USDT Pool ID", "")
        sol_ray_id = st.text_input("SOL/RAY Pool ID", "")
        
        if st.form_submit_button("Save Pool IDs"):
            pool_ids = {}
            
            if sol_usdc_id:
                pool_ids["SOL/USDC"] = [sol_usdc_id]
            
            if sol_usdt_id:
                pool_ids["SOL/USDT"] = [sol_usdt_id]
            
            if sol_ray_id:
                pool_ids["SOL/RAY"] = [sol_ray_id]
            
            if pool_ids:
                st.session_state['pool_ids'] = pool_ids
                st.success("Pool IDs saved")
                
                # Display saved IDs
                st.subheader("Saved Pool IDs")
                for pair, ids in pool_ids.items():
                    st.write(f"{pair}: {', '.join(ids)}")
            else:
                st.warning("No pool IDs were provided")
    
    # Test the pool IDs with Raydium API
    st.subheader("Test Pool IDs with Raydium API")
    
    if st.session_state.get('pool_ids'):
        if st.button("Test Pool IDs"):
            api_url = os.getenv("RAYDIUM_API_URL", "https://api.raydium.io")
            api_key = os.getenv("RAYDIUM_API_KEY", "")
            
            client = RaydiumAPIClient(api_key=api_key, base_url=api_url)
            
            for pair, ids in st.session_state['pool_ids'].items():
                st.write(f"Testing {pair} pools:")
                
                for pool_id in ids:
                    with st.spinner(f"Fetching details for {pair} pool ID: {pool_id}"):
                        try:
                            pool_details = client.get_pool_by_id(pool_id)
                            
                            if pool_details:
                                st.success(f"Successfully retrieved details for pool ID: {pool_id}")
                                st.json(json.dumps(pool_details, indent=2))
                                
                                # Try to get metrics too
                                try:
                                    metrics = client.get_pool_metrics(pool_id)
                                    if metrics:
                                        st.success(f"Successfully retrieved metrics for pool ID: {pool_id}")
                                        st.json(json.dumps(metrics, indent=2))
                                except Exception as e:
                                    st.warning(f"Error getting metrics: {str(e)}")
                            else:
                                st.error(f"No details found for pool ID: {pool_id}")
                        except Exception as e:
                            st.error(f"Error fetching details for pool ID {pool_id}: {str(e)}")
    else:
        st.warning("No pool IDs available to test. Please retrieve or input pool IDs first.")

if __name__ == "__main__":
    main()