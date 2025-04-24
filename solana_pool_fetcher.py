import streamlit as st
import requests
import json
import pandas as pd

def fetch_raydium_pools_via_solana_rpc():
    """
    This function attempts to fetch Raydium pools directly from Solana blockchain 
    using public Solana RPC endpoints
    """
    st.subheader("Fetch Raydium Pools via Solana RPC")
    
    # Solana public RPC endpoints - user can choose which one to use
    rpc_endpoints = {
        "Solana Mainnet RPC": "https://api.mainnet-beta.solana.com",
        "GenesysGo": "https://ssc-dao.genesysgo.net",
        "Serum": "https://solana-api.projectserum.com",
        "Triton": "https://free.rpcpool.com"
    }
    
    selected_endpoint = st.selectbox("Select Solana RPC Endpoint", list(rpc_endpoints.keys()))
    rpc_url = rpc_endpoints[selected_endpoint]
    
    st.info(f"Using Solana RPC endpoint: {rpc_url}")
    
    # Function to make RPC call
    def make_rpc_call(method, params):
        headers = {"Content-Type": "application/json"}
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            response = requests.post(rpc_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"RPC Call Error: {str(e)}")
            return None
    
    # Raydium program IDs
    # These are the known program IDs for Raydium on Solana
    raydium_programs = {
        "Liquidity Pool Program v4": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
        "AMM Program": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
        "Staking Program": "EhhTKczWMGQt46ynW3iobwXXwpBRMwW2PTEPeZZkLcQE"
    }
    
    selected_program = st.selectbox("Select Raydium Program", list(raydium_programs.keys()))
    program_id = raydium_programs[selected_program]
    
    if st.button("Fetch Accounts for Selected Program"):
        with st.spinner(f"Fetching accounts for program {program_id}..."):
            # Get all accounts owned by the selected Raydium program
            try:
                result = make_rpc_call("getProgramAccounts", [
                    program_id,
                    {
                        "encoding": "jsonParsed",
                        "limit": 50  # Start with a reasonable limit
                    }
                ])
                
                if result and "result" in result:
                    accounts = result["result"]
                    st.success(f"Retrieved {len(accounts)} accounts for program {program_id}")
                    
                    # Display summary of accounts
                    account_data = []
                    for i, account in enumerate(accounts):
                        pubkey = account.get("pubkey", "Unknown")
                        lamports = account.get("account", {}).get("lamports", 0)
                        owner = account.get("account", {}).get("owner", "Unknown")
                        data_size = len(account.get("account", {}).get("data", []))
                        
                        account_data.append({
                            "Index": i + 1,
                            "Public Key": pubkey,
                            "Lamports": lamports,
                            "Owner": owner,
                            "Data Size": data_size
                        })
                    
                    # Display as dataframe
                    if account_data:
                        account_df = pd.DataFrame(account_data)
                        st.dataframe(account_df)
                        
                        # Allow user to select an account to examine in detail
                        if len(account_data) > 0:
                            selected_account_idx = st.selectbox(
                                "Select account to examine", 
                                range(len(account_data)), 
                                format_func=lambda i: f"{i+1}. {account_data[i]['Public Key']}"
                            )
                            
                            if st.button("Examine Selected Account"):
                                selected_account = accounts[selected_account_idx]
                                st.json(json.dumps(selected_account, indent=2))
                                
                                # Try to get more details about the account
                                account_pubkey = selected_account["pubkey"]
                                
                                # Get account info with more data
                                account_info = make_rpc_call("getAccountInfo", [
                                    account_pubkey,
                                    {"encoding": "jsonParsed", "commitment": "confirmed"}
                                ])
                                
                                if account_info and "result" in account_info:
                                    st.subheader(f"Detailed info for account {account_pubkey}")
                                    st.json(json.dumps(account_info["result"], indent=2))
                    else:
                        st.warning("No account data to display")
                else:
                    st.error("No results returned or invalid response format")
                    if result:
                        st.json(json.dumps(result, indent=2))
            except Exception as e:
                st.error(f"Error fetching program accounts: {str(e)}")
    
    # Add option to get transaction history for a pool account
    st.subheader("Get Transaction History for a Pool Account")
    pool_address = st.text_input("Enter Pool Account Address", "")
    
    if pool_address and st.button("Get Transactions"):
        with st.spinner(f"Fetching transactions for account {pool_address}..."):
            try:
                # Get recent transactions
                result = make_rpc_call("getSignaturesForAddress", [
                    pool_address,
                    {"limit": 20}
                ])
                
                if result and "result" in result:
                    signatures = result["result"]
                    st.success(f"Retrieved {len(signatures)} recent transactions")
                    
                    # Display transaction signatures
                    tx_data = []
                    for i, tx in enumerate(signatures):
                        signature = tx.get("signature", "Unknown")
                        slot = tx.get("slot", 0)
                        block_time = tx.get("blockTime", 0)
                        
                        tx_data.append({
                            "Index": i + 1,
                            "Signature": signature,
                            "Slot": slot,
                            "Block Time": block_time
                        })
                    
                    if tx_data:
                        tx_df = pd.DataFrame(tx_data)
                        st.dataframe(tx_df)
                        
                        # Allow user to select a transaction to examine
                        if len(tx_data) > 0:
                            selected_tx_idx = st.selectbox(
                                "Select transaction to examine", 
                                range(len(tx_data)), 
                                format_func=lambda i: f"{i+1}. {tx_data[i]['Signature']}"
                            )
                            
                            if st.button("Examine Selected Transaction"):
                                selected_tx = signatures[selected_tx_idx]
                                tx_signature = selected_tx["signature"]
                                
                                # Get transaction details
                                tx_details = make_rpc_call("getTransaction", [
                                    tx_signature,
                                    {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
                                ])
                                
                                if tx_details and "result" in tx_details:
                                    st.subheader(f"Transaction Details for {tx_signature}")
                                    
                                    # Extract and display meaningful information
                                    tx_info = tx_details["result"]
                                    
                                    # Show transaction overview
                                    try:
                                        if tx_info:
                                            block_time = tx_info.get("blockTime")
                                            slot = tx_info.get("slot")
                                            confirmations = tx_info.get("meta", {}).get("confirmations")
                                            fee = tx_info.get("meta", {}).get("fee")
                                            
                                            st.write(f"Block Time: {block_time}")
                                            st.write(f"Slot: {slot}")
                                            st.write(f"Confirmations: {confirmations}")
                                            st.write(f"Fee: {fee} lamports")
                                            
                                            # Show instructions
                                            if "transaction" in tx_info and "message" in tx_info["transaction"]:
                                                instructions = tx_info["transaction"]["message"]["instructions"]
                                                st.subheader("Transaction Instructions")
                                                
                                                for i, instruction in enumerate(instructions):
                                                    st.write(f"Instruction {i+1}:")
                                                    program_id = instruction.get("programId", "Unknown")
                                                    st.write(f"Program: {program_id}")
                                                    
                                                    # If the instruction has parsed data, display it
                                                    if "parsed" in instruction:
                                                        st.json(json.dumps(instruction["parsed"], indent=2))
                                            
                                            # Full transaction details (expandable)
                                            with st.expander("View Full Transaction Details"):
                                                st.json(json.dumps(tx_info, indent=2))
                                    except Exception as e:
                                        st.error(f"Error parsing transaction info: {str(e)}")
                                        st.json(json.dumps(tx_details, indent=2))
                                else:
                                    st.error("Failed to retrieve transaction details")
                                    if tx_details:
                                        st.json(json.dumps(tx_details, indent=2))
                    else:
                        st.warning("No transaction data to display")
                else:
                    st.error("No results returned or invalid response format")
                    if result:
                        st.json(json.dumps(result, indent=2))
            except Exception as e:
                st.error(f"Error fetching transaction history: {str(e)}")

def fetch_known_raydium_pools():
    """
    This function uses a method to find known/documented Raydium pools
    """
    st.subheader("Search for Known Raydium Pools")
    
    # Known major DEXes on Solana
    dexes = {
        "Raydium AMM v4": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
        "Orca Whirlpools": "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",
        "Jupiter Aggregator": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
        "Meteora": "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K"
    }
    
    selected_dex = st.selectbox("Select DEX", list(dexes.keys()))
    program_id = dexes[selected_dex]
    
    # Hardcoded known pool addresses (examples)
    if selected_dex == "Raydium AMM v4":
        known_pools = {
            "SOL-USDC": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
            "SOL-USDT": "7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX",
            "RAY-SOL": "AVs9TA4nWDzfPJE9gGVNJMVhcQy3V9PGazuz33BfG2RA",
            "RAY-USDC": "6UmmUiYoBjSrhakAobJw8BvkmJtDVxaeBtbt7rxWo1mg",
            "RAY-USDT": "DVa7Qmb5ct9RCpaU7UTpSaf3GVMYz17vNVU67XpdCRut",
            "RAY-SRM": "7P5Thr9Egi2rvMmEuQkLn8x8e8Qro7u2U7yLD2tU2Hbe",
            "MNGO-USDC": "2LQdMz7YXqRwBUv3oNm8oEvs3tX3ASXhUAP3apPMYXeR"
        }
    elif selected_dex == "Orca Whirlpools":
        known_pools = {
            "SOL-USDC": "HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ",
            "SOL-USDT": "4fuUiYxTQ6QCrdSq9ouBYXLTyfqjfLqkEEV8eZbGG7h1",
            "SOL-mSOL": "A2G5qS6C5KymjYA9K9QXfh66gXBXeQCCp2Du1nsMpAg9"
        }
    else:
        known_pools = {}
    
    # Display known pools for the selected DEX
    if known_pools:
        st.success(f"Found {len(known_pools)} known pools for {selected_dex}")
        
        pool_data = []
        for pair, address in known_pools.items():
            pool_data.append({
                "Trading Pair": pair,
                "Pool Address": address
            })
        
        pool_df = pd.DataFrame(pool_data)
        st.dataframe(pool_df)
        
        # Select a pool to examine
        selected_pool = st.selectbox("Select pool to examine", list(known_pools.keys()))
        pool_address = known_pools[selected_pool]
        
        if st.button(f"Examine {selected_pool} Pool"):
            # Try to get pool details from Solana RPC
            solana_rpc_url = "https://api.mainnet-beta.solana.com"
            
            headers = {"Content-Type": "application/json"}
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [
                    pool_address,
                    {"encoding": "jsonParsed", "commitment": "confirmed"}
                ]
            }
            
            try:
                with st.spinner(f"Fetching details for {selected_pool} pool..."):
                    response = requests.post(solana_rpc_url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    if result and "result" in result:
                        st.success(f"Retrieved details for {selected_pool} pool")
                        
                        # Show account info
                        account_info = result["result"]["value"]
                        
                        # Display basic info
                        st.write(f"Owner: {account_info.get('owner', 'Unknown')}")
                        st.write(f"Lamports: {account_info.get('lamports', 0)}")
                        st.write(f"Executable: {account_info.get('executable', False)}")
                        st.write(f"Rent Epoch: {account_info.get('rentEpoch', 0)}")
                        
                        # Raw data is usually binary in base64 encoding
                        with st.expander("View Raw Account Data"):
                            st.json(json.dumps(account_info, indent=2))
                        
                        # Try to get transactions for this pool
                        tx_payload = {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "getSignaturesForAddress",
                            "params": [
                                pool_address,
                                {"limit": 10}
                            ]
                        }
                        
                        tx_response = requests.post(solana_rpc_url, headers=headers, json=tx_payload)
                        tx_response.raise_for_status()
                        tx_result = tx_response.json()
                        
                        if tx_result and "result" in tx_result:
                            signatures = tx_result["result"]
                            st.subheader(f"Recent Transactions for {selected_pool} Pool")
                            
                            tx_data = []
                            for tx in signatures:
                                signature = tx.get("signature", "Unknown")
                                slot = tx.get("slot", 0)
                                block_time = tx.get("blockTime", 0)
                                
                                tx_data.append({
                                    "Signature": signature[:15] + "...",
                                    "Full Signature": signature,
                                    "Slot": slot,
                                    "Block Time": block_time
                                })
                            
                            if tx_data:
                                tx_df = pd.DataFrame(tx_data)
                                st.dataframe(tx_df)
                            else:
                                st.warning("No transaction data to display")
                        else:
                            st.warning("No transaction history found for this pool")
                    else:
                        st.error("Failed to retrieve pool details")
                        if result:
                            st.json(json.dumps(result, indent=2))
            except Exception as e:
                st.error(f"Error examining pool: {str(e)}")
    else:
        st.warning(f"No known pools for {selected_dex} in our database")

def fetch_via_juputer_api():
    """
    Function to fetch pools via Jupiter API
    Jupiter is a popular aggregator that knows about all major pools on Solana
    """
    st.subheader("Fetch Pools via Jupiter API")
    
    # Jupiter API endpoint for tokens
    jupiter_api_url = "https://quote-api.jup.ag/v6"
    
    if st.button("Fetch Available Tokens via Jupiter"):
        with st.spinner("Fetching tokens from Jupiter API..."):
            try:
                # Get list of tokens from Jupiter
                tokens_url = f"{jupiter_api_url}/tokens"
                response = requests.get(tokens_url)
                response.raise_for_status()
                result = response.json()
                
                if result and "tokens" in result:
                    tokens = result["tokens"]
                    st.success(f"Retrieved {len(tokens)} tokens from Jupiter API")
                    
                    # Convert to DataFrame for better viewing
                    token_data = []
                    for token in tokens:
                        token_data.append({
                            "Symbol": token.get("symbol", ""),
                            "Name": token.get("name", ""),
                            "Address": token.get("address", ""),
                            "Decimals": token.get("decimals", 0),
                            "Chain ID": token.get("chainId", 0),
                            "Tags": ", ".join(token.get("tags", []))
                        })
                    
                    # Create a searchable dataframe
                    if token_data:
                        token_df = pd.DataFrame(token_data)
                        
                        # Filter options
                        st.subheader("Filter Tokens")
                        search_term = st.text_input("Search by Symbol or Name").upper()
                        
                        if search_term:
                            filtered_df = token_df[
                                token_df["Symbol"].str.upper().str.contains(search_term) | 
                                token_df["Name"].str.upper().str.contains(search_term)
                            ]
                            st.dataframe(filtered_df)
                        else:
                            st.dataframe(token_df)
                        
                        # Select tokens to check available routes/pools
                        st.subheader("Check Available Routes between Tokens")
                        
                        # Create dropdowns for input and output tokens
                        input_token = st.selectbox(
                            "Select Input Token", 
                            token_df["Symbol"].tolist(),
                            index=token_df[token_df["Symbol"] == "SOL"].index[0] if "SOL" in token_df["Symbol"].values else 0
                        )
                        
                        output_token = st.selectbox(
                            "Select Output Token", 
                            token_df["Symbol"].tolist(),
                            index=token_df[token_df["Symbol"] == "USDC"].index[0] if "USDC" in token_df["Symbol"].values else 0
                        )
                        
                        # Get the addresses for selected tokens
                        input_address = token_df[token_df["Symbol"] == input_token]["Address"].values[0]
                        output_address = token_df[token_df["Symbol"] == output_token]["Address"].values[0]
                        
                        if st.button(f"Check Routes for {input_token} → {output_token}"):
                            with st.spinner(f"Fetching routes from {input_token} to {output_token}..."):
                                try:
                                    # Get available routes
                                    amount = 1 * (10 ** token_df[token_df["Symbol"] == input_token]["Decimals"].values[0])
                                    quote_url = f"{jupiter_api_url}/quote?inputMint={input_address}&outputMint={output_address}&amount={amount}&slippageBps=50"
                                    
                                    quote_response = requests.get(quote_url)
                                    quote_response.raise_for_status()
                                    quote_result = quote_response.json()
                                    
                                    if "routes" in quote_result:
                                        routes = quote_result["routes"]
                                        st.success(f"Found {len(routes)} routes from {input_token} to {output_token}")
                                        
                                        # Display routes and their market infos (which contain pool addresses)
                                        for i, route in enumerate(routes):
                                            st.write(f"Route {i+1}:")
                                            st.write(f"- Output Amount: {int(route.get('outAmount', 0)) / (10 ** token_df[token_df['Symbol'] == output_token]['Decimals'].values[0]):.6f} {output_token}")
                                            st.write(f"- Price Impact: {route.get('priceImpactPct', 0):.6f}%")
                                            
                                            # Market infos contain the actual pool addresses
                                            market_infos = route.get("marketInfos", [])
                                            if market_infos:
                                                st.write(f"- Hops: {len(market_infos)}")
                                                
                                                for j, market in enumerate(market_infos):
                                                    st.write(f"  Hop {j+1}:")
                                                    st.write(f"  - AMM: {market.get('amm', 'Unknown')}")
                                                    st.write(f"  - Input Mint: {market.get('inputMint', 'Unknown')}")
                                                    st.write(f"  - Output Mint: {market.get('outputMint', 'Unknown')}")
                                                    
                                                    # This is where we get the actual pool address
                                                    if "id" in market:
                                                        st.write(f"  - Pool ID: {market.get('id', 'Unknown')}")
                                            
                                            # Expandable section for full route details
                                            with st.expander(f"View Full Route {i+1} Details"):
                                                st.json(json.dumps(route, indent=2))
                                    else:
                                        st.warning(f"No routes found from {input_token} to {output_token}")
                                        st.json(json.dumps(quote_result, indent=2))
                                except Exception as e:
                                    st.error(f"Error fetching routes: {str(e)}")
                else:
                    st.error("Failed to retrieve tokens from Jupiter API")
                    if result:
                        st.json(json.dumps(result, indent=2))
            except Exception as e:
                st.error(f"Error accessing Jupiter API: {str(e)}")

def main():
    st.title("Solana Liquidity Pool Explorer")
    
    st.write("""
    This tool provides multiple methods to discover and explore liquidity pools on 
    Solana. You can fetch pools using Solana RPC, explore known pools, or use the 
    Jupiter API to find routes between tokens (which reveals the underlying pools).
    """)
    
    method = st.sidebar.radio(
        "Select Method",
        ["Solana RPC", "Known Pools", "Jupiter API"]
    )
    
    if method == "Solana RPC":
        fetch_raydium_pools_via_solana_rpc()
    elif method == "Known Pools":
        fetch_known_raydium_pools()
    elif method == "Jupiter API":
        fetch_via_juputer_api()

if __name__ == "__main__":
    main()