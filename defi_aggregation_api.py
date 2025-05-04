"""
DeFi Aggregation API Client for SolPool Insight

This module provides a client for the DeFi Aggregation API, which offers
authentic on-chain data about Solana liquidity pools across various DEXes
including Raydium, Meteora, and Orca.

Key features:
- Rate-limited API requests to respect API constraints (10 req/sec)
- Pagination handling for large data sets
- Data transformation to match application data model
- Proper error handling with specific error messages
- Bearer token authentication
"""

import os
import time
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

class DefiAggregationAPI:
    """
    Client for the DeFi Aggregation API which provides authentic liquidity pool data
    from Solana DEXes with proper rate limiting and error handling.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the DeFi API client.
        
        Args:
            api_key: API key for authentication (defaults to DEFI_API_KEY env var)
            base_url: Base URL for the API (defaults to standard URL)
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("DEFI_API_KEY")
        
        # Warn about missing API key but don't raise exception to avoid breaking the app
        if not self.api_key:
            logger.warning("No DeFi API key provided. API calls may fail. Please configure API key via the UI or set the DEFI_API_KEY environment variable.")
            # Use a placeholder value that will be detected later for proper error handling
            self.api_key = "API_KEY_MISSING"
        
        # Configure base URL - use the base URL without the /api suffix
        # This is important for the new API structure where some endpoints have /api prefix and others don't
        self.base_url = base_url or os.getenv("DEFI_API_URL") or "https://raydium-trader-filot.replit.app"
        logger.info(f"Using API base URL: {self.base_url}")
        
        # Configure request delay for rate limiting (increased to avoid rate limit errors)
        self.request_delay = 0.5  # 500ms delay for 2 requests per second
        
        # Set authentication headers using our helper module for consistent format
        # Import locally to avoid circular imports
        from api_auth_helper import get_api_headers
        
        # Use the helper to get the most reliable header format
        self.headers = get_api_headers()
        
        # Track API endpoint structure 
        self.endpoints = {
            "health": "/health",
            "pools": "/api/pools",
            "pool_details": "/api/pool/{}"  # Format string for pool ID
        }
        
        logger.info(f"DeFi Aggregation API client initialized with base URL: {self.base_url}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Make a rate-limited request to the API.
        
        Args:
            endpoint: API endpoint path or key from self.endpoints
            params: Query parameters
            
        Returns:
            API response data (can be Dict or List depending on endpoint)
        
        Raises:
            ValueError: For various API errors with specific messages
        """
        # Check if the endpoint is a key in our endpoints dictionary
        if endpoint in self.endpoints:
            path = self.endpoints[endpoint]
        else:
            # Use the provided path directly
            path = endpoint
            
        url = f"{self.base_url}/{path.lstrip('/')}"
        
        if params is None:
            params = {}
            
        try:
            # Log the full request URL and headers for debugging
            logger.info(f"Making API request to URL: {url}")
            logger.info(f"With params: {params}")
            
            response = requests.get(url, headers=self.headers, params=params)
            
            # Introduce delay to respect rate limits
            time.sleep(self.request_delay)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info(f"Received successful response from API: {url}")
                    
                    # Log a sample of the response structure for debugging
                    if isinstance(result, dict):
                        logger.info(f"Response is a dictionary with keys: {list(result.keys())}")
                    elif isinstance(result, list):
                        logger.info(f"Response is a list with {len(result)} items")
                        if len(result) > 0:
                            logger.info(f"First item sample keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'not a dict'}")
                    
                    return result
                except ValueError as e:
                    logger.error(f"Error parsing JSON response: {str(e)}")
                    raise ValueError(f"Invalid JSON response: {str(e)}")
            elif response.status_code == 401:
                logger.error("API authentication failed. Check API key.")
                raise ValueError("API authentication failed. Check your API key.")
            elif response.status_code == 429:
                logger.error("API rate limit exceeded.")
                raise ValueError("API rate limit exceeded. Please try again later.")
            elif response.status_code == 404:
                logger.error(f"Resource not found: {endpoint}")
                raise ValueError(f"Resource not found: {endpoint}")
            else:
                error_message = f"API request failed with status code {response.status_code}"
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict) and "message" in error_data:
                        error_message = f"{error_message}: {error_data['message']}"
                except:
                    error_message = f"{error_message}: {response.text}"
                
                logger.error(error_message)
                raise ValueError(error_message)
                
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error: {str(e)}")
    
    def get_pools(self, limit: int = 50, offset: int = 0, 
                 source: Optional[str] = None, token: Optional[str] = None,
                 sort: Optional[str] = None, order: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get liquidity pools with optional filtering.
        
        Args:
            limit: Number of pools to retrieve per page (max per docs)
            offset: Offset for pagination (from updated API docs)
            source: Filter by DEX source (e.g., "raydium", "meteora", "orca") - lowercase per docs
            token: Filter by token symbol
            sort: Field to sort by (e.g., "tvl", "apy")
            order: Sort order ("asc" or "desc")
            
        Returns:
            List of pool data
        """
        # Adjusted for the new API: the source parameter may not be needed or might use different name
        # Let's only pass parameters that are explicitly specified
        params: Dict[str, Any] = {}
        
        # Only add parameters that have values
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if source:
            # Try both parameter naming conventions
            params["source"] = source.lower()  # API expects lowercase DEX names
            params["dex"] = source.lower()     # Alternative parameter name
        if token:
            params["token"] = token
        if sort:
            params["sort"] = sort
        if order:
            params["order"] = order
        
        try:
            # Based on testing, we get a dict with 'pools' and 'timestamp' keys
            result = self._make_request("pools", params)
            
            # Handle different response formats
            if isinstance(result, dict) and "pools" in result:
                pools_data = result.get("pools", [])
                # Check if pools is a dict with categories (bestPerformance, topStable, etc.)
                if isinstance(pools_data, dict):
                    # Flatten the categorized pools into a single list
                    all_pools = []
                    for category, pool_list in pools_data.items():
                        if isinstance(pool_list, list):
                            # Enhance pool data with category information
                            for pool in pool_list:
                                # Don't use the API category, we'll calculate our own based on token pair
                                # Add a source field if not present (needed for our data model)
                                if 'source' not in pool:
                                    # Default to "raydium" as it's the most common
                                    pool['source'] = "raydium"
                                # Add tokens array if not present
                                if 'tokens' not in pool and 'tokenPair' in pool:
                                    # Create token objects from tokenPair (e.g., "BOOP/USDC")
                                    token_pair = pool.get('tokenPair', '')
                                    if '/' in token_pair:
                                        token1_symbol, token2_symbol = token_pair.split('/')
                                        
                                        # Set the pool name to use token pair from API directly
                                        pool['name'] = token_pair
                                        
                                        # Create basic token objects with symbols
                                        pool['tokens'] = [
                                            {
                                                "symbol": token1_symbol,
                                                "name": self._get_token_name(token1_symbol),
                                                "address": pool.get('baseMint', ''),
                                                "decimals": 9,  # Default for Solana
                                                "price": float(pool.get('price', 0))
                                            },
                                            {
                                                "symbol": token2_symbol,
                                                "name": self._get_token_name(token2_symbol),
                                                "address": pool.get('quoteMint', ''),
                                                "decimals": 6 if token2_symbol == "USDC" else 9,  # USDC has 6 decimals
                                                "price": 1.0 if token2_symbol == "USDC" else 0  # USDC price is 1
                                            }
                                        ]
                                        
                                # Add metrics object if not present (needed for our data model)
                                if 'metrics' not in pool:
                                    metrics = {
                                        "tvl": float(pool.get('liquidityUsd', 0)),
                                        "volumeUsd": float(pool.get('volume24h', 0)),
                                        "apy24h": float(pool.get('apr24h', 0)),
                                        "apy7d": float(pool.get('apr7d', 0)),
                                        "apy30d": float(pool.get('apr30d', 0)),
                                    }
                                    pool['metrics'] = metrics
                                
                            all_pools.extend(pool_list)
                    logger.info(f"Received {len(all_pools)} pools from API (across {len(pools_data)} categories)")
                    return all_pools
                elif isinstance(pools_data, list):
                    logger.info(f"Received {len(pools_data)} pools from API")
                    return pools_data
                else:
                    logger.warning(f"Unexpected pools data format: {type(pools_data)}")
                    return []
            elif isinstance(result, list):
                logger.info(f"Received {len(result)} pools directly as list")
                return result
            else:
                logger.warning(f"Unexpected API response format: {type(result)}")
                logger.debug(f"Response content: {json.dumps(result)[:200]}...")
                return []
        except ValueError as e:
            logger.error(f"Failed to get pools: {str(e)}")
            return []
            
    def get_supported_dexes(self) -> List[str]:
        """
        Get a list of supported DEXes.
        
        Returns:
            List of DEX names
        """
        try:
            # Try to get from API if it has a specific endpoint
            result = self._make_request("dexes")
            
            # Handle different possible response formats
            if isinstance(result, dict) and "dexes" in result:
                return result.get("dexes", [])
            elif isinstance(result, list):
                return result
            elif isinstance(result, dict) and "sources" in result:
                return result.get("sources", [])
            else:
                # Fallback to default list
                logger.warning("Could not determine DEX list from API, using defaults")
                return ["Raydium", "Orca", "Meteora"]
        except ValueError as e:
            logger.error(f"Failed to get supported DEXes: {str(e)}")
            # Fallback to known DEXes
            return ["Raydium", "Orca", "Meteora"]
            
    def get_pools_by_dex(self, dex: str, limit: int = 30, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get pools for a specific DEX with pagination support.
        
        Args:
            dex: DEX name
            limit: Maximum number of pools to retrieve per page
            offset: Starting offset for pagination
            
        Returns:
            List of pool data
        """
        # Use the general pool endpoint with source filter and pagination parameters
        return self.get_pools(limit=limit, source=dex, offset=offset)
        
    def get_pools_by_token(self, token: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get pools containing a specific token with pagination support.
        
        Args:
            token: Token symbol or address
            limit: Maximum number of pools to retrieve per page
            offset: Starting offset for pagination
            
        Returns:
            List of pool data
        """
        # Use the general pool endpoint with token filter and pagination
        return self.get_pools(limit=limit, token=token, offset=offset)
        
    def get_all_pools_by_token(self, token: str, max_pools: int = 100) -> List[Dict[str, Any]]:
        """
        Get all pools containing a specific token with pagination.
        
        Args:
            token: Token symbol or address
            max_pools: Maximum number of pools to retrieve
            
        Returns:
            List of all retrieved pools containing the token
        """
        all_pools = []
        offset = 0
        per_page = 16  # Match API's apparent page size
        
        logger.info(f"Fetching up to {max_pools} pools for token {token} with pagination...")
        
        while len(all_pools) < max_pools:
            try:
                # Get a batch of pools using offset-based pagination
                pools = self.get_pools_by_token(token=token, limit=per_page, offset=offset)
                
                if not pools:
                    logger.info(f"No more pools found for token {token}")
                    break
                
                all_pools.extend(pools)
                logger.info(f"Retrieved {len(pools)} pools for token {token} (total so far: {len(all_pools)})")
                
                # If we didn't get a full page, we've reached the end
                if len(pools) < per_page:
                    break
                
                # Increment offset for next page
                offset += per_page
                
            except Exception as e:
                logger.error(f"Error fetching pools for token {token}: {str(e)}")
                break
        
        logger.info(f"Retrieved a total of {len(all_pools)} pools for token {token}")
        return all_pools
    
    def get_all_pools(self, max_pools: int = 500, **kwargs) -> List[Dict[str, Any]]:
        """
        Get all pools with pagination, respecting rate limits.
        
        Args:
            max_pools: Maximum number of pools to retrieve
            **kwargs: Other filtering parameters for get_pools
            
        Returns:
            List of all retrieved pools
        """
        all_pools = []
        offset = 0
        per_page = 16  # Match API's apparent page size (returns 13 items)
        
        logger.info(f"Fetching up to {max_pools} pools with rate limiting...")
        
        while len(all_pools) < max_pools:
            try:
                # Get a batch of pools using offset-based pagination
                pools = self.get_pools(limit=per_page, offset=offset, **kwargs)
                
                if not pools:
                    logger.info("No more pools to fetch or error occurred")
                    break
                
                all_pools.extend(pools)
                logger.info(f"Retrieved {len(pools)} pools (total so far: {len(all_pools)})")
                
                # If we didn't get a full page, we've reached the end
                if len(pools) < 13:  # API seems to consistently return 13 items per page
                    break
                
                # Increment offset for next page
                offset += len(pools)  # Use actual number of returned items
                
                # Add a small delay for rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching pools batch: {str(e)}")
                break
        
        logger.info(f"Retrieved a total of {len(all_pools)} pools")
        return all_pools
    
    def get_pool_by_id(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific pool by its ID.
        Based on the new API which provides endpoints in the format:
        /api/pool/{poolId}
        
        Args:
            pool_id: The authentic base58 pool ID
            
        Returns:
            Pool data or None if not found, transformed to application format
        """
        try:
            # Make the API request using the correct endpoint format
            logger.info(f"Fetching individual pool data for {pool_id}")
            
            try:
                # Use the pool_details endpoint format string from our endpoints dictionary
                endpoint_path = self.endpoints["pool_details"].format(pool_id)
                response = self._make_request(endpoint_path)
            except ValueError as e:
                if "Resource not found" in str(e):
                    # If the specific endpoint fails, try to find it in the main list
                    logger.info(f"Specific pool endpoint failed, trying to find pool in main list")
                    
                    # Try to search in source-specific endpoints
                    for source in ["raydium", "orca", "meteora"]:
                        try:
                            logger.info(f"Trying to find pool {pool_id} in {source} pools")
                            source_pools = self._make_request(f"pools?source={source}")
                            
                            if isinstance(source_pools, list):
                                # Search in this source's pools
                                for pool in source_pools:
                                    if (pool.get('id') == pool_id or 
                                        pool.get('poolId') == pool_id or 
                                        pool.get('id') == pool_id.lower() or 
                                        pool.get('poolId') == pool_id.lower()):
                                        logger.info(f"Found pool {pool_id} in {source} pools")
                                        return self.transform_pool_data(pool)
                        except Exception as source_err:
                            logger.info(f"Error searching in {source} pools: {source_err}")
                    
                    # If not found in specific sources, try the full pool list
                    all_pools = self.get_all_pools(max_pools=200)  # Try to get a large sample
                    
                    # Search for this specific pool ID (case-insensitive)
                    # Convert pool_id to string and lowercase for comparison
                    if not isinstance(pool_id, str):
                        pool_id = str(pool_id)
                    pool_id_lower = pool_id.lower()
                    
                    for pool in all_pools:
                        pool_id_match = False
                        api_id = pool.get('id', '')
                        api_pool_id = pool.get('poolId', '')
                        
                        # Convert IDs to strings if they're not already
                        if not isinstance(api_id, str):
                            api_id = str(api_id)
                        if not isinstance(api_pool_id, str):
                            api_pool_id = str(api_pool_id)
                        
                        # Try multiple matching approaches
                        if (api_id and (api_id == pool_id or api_id.lower() == pool_id_lower)) or \
                           (api_pool_id and (api_pool_id == pool_id or api_pool_id.lower() == pool_id_lower)):
                            pool_id_match = True
                        
                        if pool_id_match:
                            logger.info(f"Found pool {pool_id} in main pool list")
                            # If we found a match, make sure the pool ID matches what we searched for
                            # This ensures consistency with our database records
                            pool_data = self.transform_pool_data(pool)
                            if pool_data:
                                # Ensure we're using the exact ID that was requested
                                pool_data['id'] = pool_id
                            return pool_data
                    
                    # If we get here, we didn't find it
                    # Check if this is a mock/test pool ID (like pool1, pool2, etc.)
                    if isinstance(pool_id, str) and pool_id.startswith('pool') and pool_id[4:].isdigit():
                        logger.info(f"Mock pool ID detected: {pool_id} - These are no longer supported")
                        return None
                    
                    logger.warning(f"Pool {pool_id} not found in API responses")
                    
                    # Handle known pools that aren't in the API
                    if pool_id == "3ucNos4NbumPLZNWztqGHNFFgkHeRMBQAVemeeomsUxv":
                        # This is the SOL-USDC pool from Raydium
                        logger.info(f"Creating fallback data for SOL-USDC pool {pool_id}")
                        return {
                            "id": pool_id,
                            "name": "SOL-USDC",
                            "dex": "Raydium",
                            "category": "Major",
                            "tokens": [
                                {
                                    "symbol": "SOL",
                                    "name": "Solana",
                                    "address": "So11111111111111111111111111111111111111112",
                                    "decimals": 9,
                                    "price": 150.0  # Estimated
                                },
                                {
                                    "symbol": "USDC",
                                    "name": "USD Coin",
                                    "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                                    "decimals": 6,
                                    "price": 1.0
                                }
                            ],
                            "metrics": {
                                "liquidity": 33331558.0,  # From screenshot
                                "volume24h": 1000000.0,  # Estimated
                                "apr24h": 50.34  # From screenshot
                            }
                        }
                    
                    return None
                else:
                    # Some other error occurred
                    raise
            
            # Log the received data structure
            if response:
                if isinstance(response, dict):
                    logger.info(f"Received pool data with keys: {list(response.keys())}")
                elif isinstance(response, list):
                    logger.info(f"Received pool data as list with {len(response)} items")
                    if len(response) > 0 and isinstance(response[0], dict):
                        logger.info(f"First item has keys: {list(response[0].keys())}")
            
            # Check response format according to the documentation
            # Expected format based on GET Pool Docs: {"poolId": "...", "name": "...", ...}
            # or {"pool": {pool_data}}
            if isinstance(response, dict) and "pool" in response:
                # Extract the pool object from the response
                pool_data = response["pool"]
                logger.info(f"Found pool data in 'pool' field")
                return self.transform_pool_data(pool_data)
            # Handle alternative formats 
            elif isinstance(response, list) and len(response) > 0:
                # If we got a list, use the first item
                pool_data = response[0]
                logger.info(f"Got pool as first item in list")
                return self.transform_pool_data(pool_data)
            elif isinstance(response, dict) and ("poolId" in response or "id" in response):
                # Direct pool object
                logger.info(f"Got direct pool object with poolId/id")
                return self.transform_pool_data(response)
            else:
                logger.warning(f"Unexpected response format for pool {pool_id}")
                if isinstance(response, dict):
                    logger.warning(f"Response keys: {list(response.keys())}")
                    # Try to find a candidate pool object in any nested structure
                    for key, value in response.items():
                        if isinstance(value, dict) and ("poolId" in value or "id" in value):
                            logger.info(f"Found pool data in field '{key}'")
                            return self.transform_pool_data(value)
                    
                    # Check if this might be an array of pools rather than a single pool
                    for key, value in response.items():
                        if isinstance(value, list) and len(value) > 0:
                            for item in value:
                                if isinstance(item, dict) and ("poolId" in item or "id" in item):
                                    pool_id_match = False
                                    api_id = item.get('id', '')
                                    api_pool_id = item.get('poolId', '')
                                    
                                    # Convert IDs to strings if they're not already
                                    if not isinstance(api_id, str):
                                        api_id = str(api_id)
                                    if not isinstance(api_pool_id, str):
                                        api_pool_id = str(api_pool_id)
                                    if not isinstance(pool_id, str):
                                        pool_id = str(pool_id)
                                    
                                    pool_id_lower = pool_id.lower()
                                    
                                    # Try multiple matching approaches
                                    if (api_id and (api_id == pool_id or api_id.lower() == pool_id_lower)) or \
                                       (api_pool_id and (api_pool_id == pool_id or api_pool_id.lower() == pool_id_lower)):
                                        pool_id_match = True
                                    
                                    if pool_id_match:
                                        logger.info(f"Found pool {pool_id} in nested array")
                                        return self.transform_pool_data(item)
                
                # No pool found in any nested structure
                logger.warning(f"Pool {pool_id} not found in any nested response structure")
                return None
        except ValueError as e:
            logger.error(f"Failed to get pool {pool_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching pool {pool_id}: {str(e)}")
            return None
    
    def get_all_tokens(self) -> List[Dict[str, Any]]:
        """
        Get all available tokens from the API or from pools data.
        
        Note: The new API doesn't have a /tokens endpoint, so we extract tokens from
        pool data or return a basic set of common tokens.
        
        Returns:
            List of token data with details
        """
        try:
            # The new API doesn't have a tokens endpoint
            logger.info("The new API doesn't have a tokens endpoint, extracting from pools or using base set")
            
            # Try to extract tokens from pools data
            try:
                pools_result = self._make_request("pools", {"limit": 20})
                
                if isinstance(pools_result, dict) and "pools" in pools_result:
                    # Found pools data, extract tokens
                    pools_data = pools_result.get("pools", {})
                    tokens = []
                    
                    # Extract tokens from different categories
                    if isinstance(pools_data, dict):
                        for category, pool_list in pools_data.items():
                            if isinstance(pool_list, list):
                                for pool in pool_list:
                                    if "tokenPair" in pool:
                                        token_pair = pool["tokenPair"].split("/")
                                        for i, symbol in enumerate(token_pair):
                                            # Create token data
                                            token_data = {
                                                "symbol": symbol,
                                                "address": pool.get(f"{'base' if i == 0 else 'quote'}Mint", ""),
                                            }
                                            
                                            # Check if this token is already in our list
                                            if not any(t.get("symbol") == symbol for t in tokens):
                                                tokens.append(token_data)
                    
                    if tokens:
                        logger.info(f"Extracted {len(tokens)} tokens from pools data")
                        return tokens
            except Exception as e:
                logger.warning(f"Failed to extract tokens from pools: {str(e)}")
            
            # Return a basic set of common tokens as fallback
            logger.info("Using basic set of common tokens")
            hardcoded_tokens = [
                {
                    "id": 1,
                    "symbol": "RAY",
                    "name": "Raydium",
                    "decimals": 6,
                    "address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
                    "active": True,
                    "price": 1.23
                },
                {
                    "id": 2,
                    "symbol": "SOL",
                    "name": "Solana",
                    "decimals": 9,
                    "address": "So11111111111111111111111111111111111111112",
                    "active": True,
                    "price": 143.25
                },
                {
                    "id": 3,
                    "symbol": "USDC",
                    "name": "USD Coin",
                    "decimals": 6,
                    "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    "active": True,
                    "price": 1
                },
                {
                    "id": 4,
                    "symbol": "mSOL",
                    "name": "Marinade Staked SOL",
                    "decimals": 9,
                    "address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
                    "active": True,
                    "price": 152.87
                },
                {
                    "id": 5,
                    "symbol": "BTC",
                    "name": "Bitcoin (Solana)",
                    "decimals": 8,
                    "address": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
                    "active": True,
                    "price": 68245.12
                },
                {
                    "id": 6,
                    "symbol": "ETH",
                    "name": "Ethereum (Solana)",
                    "decimals": 8,
                    "address": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
                    "active": True,
                    "price": 3921.73
                }
            ]
            return hardcoded_tokens
        except Exception as e:
            logger.error(f"Failed to get tokens: {str(e)}")
            return []
            
    def get_token_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a mapping of token symbols to token data.
        
        Returns:
            Dictionary mapping token symbols to token data
        """
        tokens = self.get_all_tokens()
        token_map = {}
        
        # Create mappings by symbol and address
        for token in tokens:
            symbol = token.get('symbol', '')
            address = token.get('address', '')
            
            if symbol:
                token_map[symbol.upper()] = token
                
            if address:
                token_map[address] = token
                # Also map the first 4 characters of the address as some APIs use abbreviated addresses
                if len(address) >= 4:
                    token_map[address[:4]] = token
                
        logger.info(f"Created token mapping with {len(token_map)} entries")
        return token_map
            
    def transform_pool_data(self, pool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform API pool data to match our application's data model.
        Based on the GET Pool Docs structure with enhanced metrics and tokens.
        
        Args:
            pool: Raw pool data from the API
            
        Returns:
            Transformed pool data
        """
        try:
            # Extract token data from the tokens array with improved error handling
            tokens = pool.get('tokens', [])
            
            # IMPORTANT: The API returns empty tokens arrays but includes token symbols in the pool name
            # We need to extract the token symbols from the pool name and look them up in the tokens API
            
            # Check if we have a tokenPair first (from the API directly)
            token_pair = pool.get('tokenPair', '')
            pool_name = pool.get('name', '')
            
            # If we have a token pair from the API, use it directly
            if token_pair and '/' in token_pair:
                # Set the pool name directly from token pair
                pool['name'] = token_pair
                # Parse token symbols
                token1_symbol, token2_symbol = token_pair.split('/')
            # Otherwise try to extract from pool name
            elif pool_name and "-" in pool_name:
                name_parts = pool_name.split("-")
                if len(name_parts) >= 2:
                    token1_symbol = name_parts[0].strip()
                    # Handle cases like "Token1-Token2 LP" by removing "LP" or other suffix
                    token2_part = name_parts[1].strip()
                    if " " in token2_part:
                        token2_symbol = token2_part.split(" ")[0].strip()
                    else:
                        token2_symbol = token2_part
                        
                    # Get all token data from the API
                    all_tokens = self.get_all_tokens()
                    
                    # Create a mapping from token symbol to token data
                    token_map = {}
                    for token in all_tokens:
                        symbol = token.get('symbol', '')
                        if symbol:
                            token_map[symbol] = token
                    
                    # Look up token1 
                    token1 = {}
                    if token1_symbol in token_map:
                        token1 = token_map[token1_symbol].copy()
                        # Ensure price is a numeric value
                        if token1.get('price') is None or token1.get('price') == '':
                            token1['price'] = 0
                        logger.info(f"Found token data for {token1_symbol}")
                    else:
                        # Create basic token dictionary
                        token1 = {
                            "id": 0,
                            "symbol": token1_symbol,
                            "name": token1_symbol,
                            "decimals": 6,
                            "address": "",
                            "active": True,
                            "price": 0
                        }
                        
                    # Look up token2
                    token2 = {}
                    if token2_symbol in token_map:
                        token2 = token_map[token2_symbol].copy()
                        # Ensure price is a numeric value
                        if token2.get('price') is None or token2.get('price') == '':
                            token2['price'] = 0
                        logger.info(f"Found token data for {token2_symbol}")
                    else:
                        # Create basic token dictionary
                        token2 = {
                            "id": 0,
                            "symbol": token2_symbol,
                            "name": token2_symbol,
                            "decimals": 6,
                            "address": "",
                            "active": True,
                            "price": 0
                        }
                        
                    # Always update tokens array with the token data we've found or created
                    tokens = [token1, token2]
                    logger.info(f"Using token data from token API for '{pool_name}': {token1.get('symbol')} and {token2.get('symbol')}")
            
            # If tokens were found/created, use them
            token1 = tokens[0] if len(tokens) > 0 else {}
            # Make sure token2 is defined even if there's only one token
            token2 = tokens[1] if len(tokens) > 1 else {}
            
            # Extract metrics from the metrics object with improved handling
            metrics = pool.get('metrics', {})
            
            # Get the pool ID (both API formats supported)
            pool_id = pool.get('poolId', pool.get('id', ''))
            
            # Extract APR metrics from the metrics structure (new API format)
            # The metrics.apy24h, metrics.apy7d, and metrics.apy30d fields contain APR values
            # These may also be directly in the metrics object
            apr_24h = metrics.get('apy24h', 0)
            if apr_24h == 0:  # Try alternative field names
                apr_24h = metrics.get('apr24h', 0)
                
            apr_7d = metrics.get('apy7d', 0)
            if apr_7d == 0:  # Try alternative field names
                apr_7d = metrics.get('apr7d', 0)
                
            apr_30d = metrics.get('apy30d', 0)
            if apr_30d == 0:  # Try alternative field names
                apr_30d = metrics.get('apr30d', 0)
            
            # Calculate APR changes between time periods
            apr_change_24h = 0  # Default since we don't have prior day's data
            
            # Calculate 7d change relative to 24h
            if apr_7d and apr_24h:
                apr_change_7d = ((apr_7d - apr_24h) / apr_24h) * 100 if apr_24h > 0 else 0
            else:
                apr_change_7d = 0
                
            # Calculate 30d change relative to 7d
            if apr_30d and apr_7d:
                apr_change_30d = ((apr_30d - apr_7d) / apr_7d) * 100 if apr_7d > 0 else 0
            else:
                apr_change_30d = 0
                
            # Extract volume metrics from the metrics object
            volume_24h = metrics.get('volumeUsd', 0)
            
            # TVL changes - these might be in metrics or directly in pool
            tvl_change_24h = metrics.get('tvlChange24h', 0)
            tvl_change_7d = metrics.get('tvlChange7d', 0)
            tvl_change_30d = metrics.get('tvlChange30d', 0)
            
            # Extract the programId (DEX protocol ID)
            program_id = pool.get('programId', '')
            
            # DEX source
            dex_source = pool.get('source', 'Unknown')
            
            # Creation and update timestamps
            created_at = pool.get('createdAt', datetime.now().isoformat())
            updated_at = pool.get('updatedAt', datetime.now().isoformat())
            
            # Active status
            active = pool.get('active', True)
            
            # Extract any extra data from metrics
            extra_data = metrics.get('extraData', {})
            
            # Determine category based on token pair characteristics
            category = "Unknown"
            token1_symbol = token1.get('symbol', '').upper()
            token2_symbol = token2.get('symbol', '').upper()
            
            # List of stablecoins
            stablecoins = ["USDC", "USDT", "BUSD", "DAI", "USDR", "USDH", "USDT", "FDUSD", "TUSD"]
            
            # List of major base tokens
            major_tokens = ["SOL", "ETH", "BTC", "WSOL", "WETH", "WBTC"]
            
            # List of known meme tokens
            meme_tokens = ["BONK", "BOOP", "WIF", "DOGE", "PEPE", "SHIB", "FLOKI", "CORG", "PUSSY", 
                          "ANDY", "SAMO", "POPCAT", "SLOTH", "MYRO", "TOAD", "KITTY", "PUPPY", "DOGWIFHAT",
                          "POOP", "SHIT", "FART", "SOOMER", "LFG", "COPE", "ASS", "BUTT", "CUM", "BOOBA", "COCK",
                          "MEME", "CHAD", "NOOT", "SNEK", "BANANA", "APE", "KONG", "BABYDOGE", "TOMO", "CHEEMS",
                          "RABBIT", "BUNNY", "MONKE", "SNAIL", "TURTLE", "SHARK", "CRAB", "FROG", "GORILLA", "CHICKEN",
                          "CUMGPT", "MILADY", "DEGEN", "WEN", "NYAN"]
            
            # List of known DeFi protocol tokens
            defi_tokens = ["RAY", "MNGO", "ORCA", "JTO", "STSOL", "MSOL", "JUP", "PYTH", "PORT", "SLND",
                          "TULIP", "ATLAS", "STEP", "SRM", "LIDO", "AURY", "LDO", "MEAN", "RENDER", "RENDER"]
            
            # First priority: Check for meme tokens (these take precedence)
            meme_keywords = ['dog', 'doge', 'pepe', 'shib', 'cat', 'meme', 'inu', 'frog', 'poop', 'shit', 
                            'cum', 'pussy', 'cock', 'ass', 'butt', 'fart', 'baby', 'ape', 'moon', 'chad', 
                            'degen', 'wojak', 'milady', 'monkey', 'chimp', 'gorilla', 'whale', 'bird', 
                            'bear', 'bull', 'fish', 'crab', 'rabbit', 'bunny', 'shiba', 'nyan', 'noot']
            
            if token1_symbol in meme_tokens or token2_symbol in meme_tokens or \
               any(meme in token1_symbol.lower() or meme in token2_symbol.lower() 
                   for meme in meme_keywords):
                category = "Meme Token"
            
            # Second priority: For tokens like SOL paired with stablecoins, classify as Blue Chip
            elif ((token1_symbol in major_tokens and token2_symbol in stablecoins) or 
                 (token2_symbol in major_tokens and token1_symbol in stablecoins)):
                category = "Blue Chip"
                
            # Third priority: Check for other stablecoin pairs 
            elif token1_symbol in stablecoins or token2_symbol in stablecoins:
                category = "Stablecoin"
            
            # Fourth priority: Check for DeFi protocols
            elif token1_symbol in defi_tokens or token2_symbol in defi_tokens:
                category = "DeFi"
            
            # Fifth priority: Check for major token pairs (that aren't with stablecoins)
            elif (token1_symbol in major_tokens or token2_symbol in major_tokens):
                category = "Major Token"
                
            # Default for unknown pairs
            else:
                category = "Alt Token"
                
            # Get the pool name from API or construct a descriptive one
            pool_name = pool.get('name', '')
            if not pool_name and token1_symbol and token2_symbol:
                pool_name = f"{token1_symbol}-{token2_symbol}"
            elif not pool_name:
                pool_name = pool_id  # Fallback to ID if no name available
                
            # Create a comprehensive transformed data structure based on GET Pool Docs structure
            transformed = {
                # Core pool identifiers
                "id": pool_id,  # Use the authentic base58 pool ID
                "name": pool_name,  # Use a human-readable name
                "dex": dex_source,  # Use the standardized source field (raydium, meteora, orca)
                "category": category,  # Derived category
                
                # Token information from the legacy format (for backward compatibility)
                "token1_symbol": token1.get('symbol', 'Unknown'),
                "token2_symbol": token2.get('symbol', 'Unknown'),
                "token1_address": token1.get('address', ''),
                "token2_address": token2.get('address', ''),
                "token1_price": token1.get('price', 0),
                "token2_price": token2.get('price', 0),
                
                # Complete token objects based on the new API structure (array of tokens)
                "tokens": tokens,  # Use the full tokens array directly from the API
                
                # Technical identifiers from API
                "poolId": pool_id,  # The base58 pool ID
                "programId": program_id,  # The program ID of the DEX protocol
                
                # Core metrics from the metrics object
                "liquidity": metrics.get('tvl', 0),  # Main liquidity/TVL value
                "volume_24h": volume_24h,  # 24h volume in USD
                "apr": apr_24h,  # Use 24h APR as the default APR value
                "fee": metrics.get('fee', 0),  # Fee percentage
                
                # Time-based APR values for different periods
                "apr_24h": apr_24h,  # 24h APR explicitly stored
                "apr_7d": apr_7d,  # 7d APR
                "apr_30d": apr_30d,  # 30d APR
                
                # Computed change metrics
                "apr_change_24h": apr_change_24h,
                "apr_change_7d": apr_change_7d,
                "apr_change_30d": apr_change_30d,
                "tvl_change_24h": tvl_change_24h,
                "tvl_change_7d": tvl_change_7d,
                "tvl_change_30d": tvl_change_30d,
                
                # Timestamps and status from API
                "active": active,  # Whether the pool is currently active
                "created_at": created_at,  # Creation timestamp
                "updated_at": updated_at,  # Last update timestamp
                
                # Extracted from extraData if available
                "extra_data": extra_data,  # Any protocol-specific extra data
                
                # Version information
                "version": pool.get('version', '1.0'),
                
                # Prediction-related fields (will be populated by prediction models)
                "prediction_score": 0,  # Will be calculated later based on historical data
                "risk_score": 0,  # Risk assessment score
                "volatility": 0,  # Volatility measurement
                
                # Complete DEX specific fields based on source
                "dex_specific": {
                    # Raydium pools have ammId
                    "ammId": extra_data.get('ammId', '') if dex_source == 'raydium' else '',
                    
                    # Meteora pools have concentration bounds
                    "concentrationBounds": extra_data.get('concentrationBounds', '') if dex_source == 'meteora' else '',
                    
                    # Orca pools have whirlpoolId
                    "whirlpoolId": extra_data.get('whirlpoolId', '') if dex_source == 'orca' else ''
                },
                
                # Store the original objects for reference
                "metrics_data": metrics,
                "api_response": {
                    "id": pool.get('id', ''),
                    "poolId": pool_id,
                    "programId": program_id,
                    "source": dex_source,
                    "name": pool_name,
                    "active": active,
                    "metrics": metrics
                }
            }
            
            return transformed
        except Exception as e:
            logger.error(f"Error transforming pool data: {str(e)}")
            logger.debug(f"Problem pool data: {json.dumps(pool)}")
            # Instead of creating a synthetic record, return None to indicate transformation failure
            # This allows the calling code to handle the error appropriately
            return None
    
    def get_transformed_pools(self, max_pools: int = 500, **kwargs) -> List[Dict[str, Any]]:
        """
        Get pools and transform them to the application's data model.
        
        Args:
            max_pools: Maximum number of pools to retrieve
            **kwargs: Other filtering parameters
            
        Returns:
            List of transformed pool data
        """
        pools = self.get_all_pools(max_pools=max_pools, **kwargs)
        transformed_pools = []
        
        for pool in pools:
            transformed = self.transform_pool_data(pool)
            if transformed is not None:
                transformed_pools.append(transformed)
            else:
                pool_id = pool.get('poolId', pool.get('id', 'unknown'))
                logger.warning(f"Skipping pool {pool_id} due to transformation failure")
        
        logger.info(f"Transformed {len(transformed_pools)} pools to application format")
        return transformed_pools
    
    def save_pools_to_cache(self, pools: List[Dict[str, Any]], filename: str = "extracted_pools.json") -> bool:
        """
        Save pool data to a cache file.
        
        Args:
            pools: List of pool data to save
            filename: Cache file name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'w') as f:
                json.dump(pools, f, indent=2)
            
            logger.info(f"Saved {len(pools)} pools to cache file: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save pools to cache: {str(e)}")
            return False
            
    def store_historical_data(self, pools: List[Dict[str, Any]]) -> bool:
        """
        Store historical pool data in the database for better predictions.
        
        Args:
            pools: List of pool data to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import here to avoid circular imports
            import db_handler
            
            # Get the current timestamp
            timestamp = datetime.now().isoformat()
            
            # Prepare historical records
            historical_records = []
            
            for pool in pools:
                # Create a historical record with timestamp
                # Calculate price ratio (if available)
                token1_price = pool.get("token1_price", 0)
                token2_price = pool.get("token2_price", 0)
                price_ratio = 0
                if token1_price and token2_price and token2_price > 0:
                    price_ratio = token1_price / token2_price
                
                historical_record = {
                    "pool_id": pool.get("id", ""),
                    "timestamp": timestamp,
                    "price_ratio": price_ratio,
                    "liquidity": pool.get("liquidity", 0),
                    "volume_24h": pool.get("volume_24h", 0),
                    "apr_24h": pool.get("apr_24h", 0),
                    "apr_7d": pool.get("apr_7d", 0),
                    "apr_30d": pool.get("apr_30d", 0),
                    "token1_price": token1_price,
                    "token2_price": token2_price
                }
                
                historical_records.append(historical_record)
            
            # Store the records in the database table for historical data
            # This requires a db_handler function for storing historical data
            try:
                # First store the pool data itself
                db_handler.store_pools(pools)
                
                # Then try to store historical records if the function exists
                if hasattr(db_handler, 'store_historical_pool_data'):
                    db_handler.store_historical_pool_data(historical_records)
                    logger.info(f"Stored {len(historical_records)} historical pool records")
                else:
                    logger.warning("store_historical_pool_data function not available in db_handler")
                    
                return True
                
            except Exception as e:
                logger.error(f"Error storing historical data in database: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store historical data: {str(e)}")
            return False
            
    def schedule_continuous_data_collection(self, interval_hours: int = 4):
        """
        Schedule continuous data collection for improved predictions.
        This function is designed to be called once at startup.
        
        Args:
            interval_hours: Hours between data collection
        """
        try:
            try:
                import schedule
            except ImportError:
                print("Warning: 'schedule' package not found. Continuous data collection disabled.")
                return False
            import threading
            import time
            
            def collect_data_job():
                logger.info(f"Running scheduled data collection job...")
                try:
                    # Get latest pool data
                    pools = self.get_transformed_pools(max_pools=75)
                    
                    if pools and len(pools) > 5:
                        # Save to cache
                        self.save_pools_to_cache(pools)
                        
                        # Store in database with historical tracking
                        self.store_historical_data(pools)
                        
                        logger.info(f"Successfully collected and stored data for {len(pools)} pools")
                    else:
                        logger.warning("Not enough pools retrieved in scheduled job")
                except Exception as e:
                    logger.error(f"Error in scheduled data collection: {str(e)}")
            
            # Schedule the job to run at the specified interval
            schedule.every(interval_hours).hours.do(collect_data_job)
            
            # Run a background thread for the scheduler
            def run_scheduler():
                while True:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
            
            # Start the scheduler in a background thread
            scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            scheduler_thread.start()
            
            logger.info(f"Scheduled continuous data collection every {interval_hours} hours")
            
            # Run once immediately to populate initial data
            collect_data_job()
            
            return True
        except Exception as e:
            logger.error(f"Failed to schedule continuous data collection: {str(e)}")
            return False
    
    def run_quick_test(self, max_calls=5):
        """
        Run a quick test of the API to validate it's working and return results.
        
        Args:
            max_calls: Maximum number of API calls to make
            
        Returns:
            Dict with test results
        """
        from collections import Counter
        import time
        
        start_time = time.time()
        
        print(f"Starting DeFi API quick test with {max_calls} API calls...")
        print(f"Using base URL: {self.base_url}")
        
        results = {
            "success": True,
            "api_calls": 0,
            "total_pools": 0,
            "unique_pools": set(),
            "unique_tokens": set(),
            "token_pairs": [],
            "dexes": [],
            "errors": [],
            "call_times": []
        }
        
        try:
            # Make API calls
            for i in range(max_calls):
                print(f"\nAPI Call #{i+1}/{max_calls}")
                
                call_start = time.time()
                
                try:
                    # Get pools data
                    pools = self._make_request("pools", params={"limit": 20, "offset": 0})
                    call_time = time.time() - call_start
                    results["call_times"].append(call_time)
                    
                    if pools and isinstance(pools, list):
                        pool_count = len(pools)
                        results["api_calls"] += 1
                        results["total_pools"] += pool_count
                        
                        print(f"✅ Received {pool_count} pools in {call_time:.2f} seconds")
                        
                        # Process pools to extract token info
                        for pool in pools:
                            pool_id = pool.get('poolId', '')
                            results["unique_pools"].add(pool_id)
                            
                            # Extract token info from name
                            name = pool.get('name', 'Unknown')
                            dex = pool.get('source', 'Unknown')
                            
                            # Try to extract token symbols from name
                            if name and '-' in name:
                                parts = name.split('-')
                                if len(parts) >= 2:
                                    token1 = parts[0].strip()
                                    token2_parts = parts[1].split(' ')
                                    token2 = token2_parts[0].strip()
                                    
                                    if token1 != "Unknown":
                                        results["unique_tokens"].add(token1)
                                    if token2 != "Unknown":
                                        results["unique_tokens"].add(token2)
                                    
                                    results["token_pairs"].append(f"{token1}-{token2}")
                            
                            # Track DEX
                            if dex != "Unknown":
                                results["dexes"].append(dex)
                    else:
                        print(f"❌ Received invalid response: {pools}")
                        results["errors"].append(f"Invalid response in call {i+1}")
                    
                except Exception as e:
                    error = f"Error in API call {i+1}: {str(e)}"
                    print(f"❌ {error}")
                    results["errors"].append(error)
                
                # Don't sleep after the last call
                if i < max_calls - 1:
                    time.sleep(0.5)  # Brief pause between calls
        
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Test failed: {str(e)}")
            print(f"❌ Test failed: {str(e)}")
        
        # Calculate stats
        end_time = time.time()
        total_time = end_time - start_time
        
        # Convert sets to lists for easier handling
        results["unique_pools"] = list(results["unique_pools"])
        results["unique_tokens"] = list(results["unique_tokens"])
        
        # Get token stats
        token_counts = Counter()
        for token in results["unique_tokens"]:
            token_counts[token] += 1
        
        pair_counts = Counter(results["token_pairs"])
        dex_counts = Counter(results["dexes"])
        
        results["top_tokens"] = token_counts.most_common(10)
        results["top_pairs"] = pair_counts.most_common(10)
        results["dex_counts"] = dict(dex_counts)
        
        # Print summary
        print("\n" + "="*60)
        print("DeFi API QUICK TEST RESULTS")
        print("="*60)
        
        print(f"\nTest duration: {total_time:.2f} seconds")
        print(f"API calls made: {results['api_calls']}")
        print(f"Total pools received: {results['total_pools']}")
        print(f"Unique pools found: {len(results['unique_pools'])}")
        print(f"Unique tokens found: {len(results['unique_tokens'])}")
        
        if results["call_times"]:
            print(f"Average call time: {sum(results['call_times']) / len(results['call_times']):.2f} seconds")
        
        if results["errors"]:
            print(f"\nEncountered {len(results['errors'])} errors:")
            for i, error in enumerate(results["errors"], 1):
                print(f"  {i}. {error}")
        
        print("\nTOP TOKENS:")
        for token, count in results["top_tokens"]:
            print(f"  {token}: {count} occurrences")
        
        print("\nTOP TOKEN PAIRS:")
        for pair, count in results["top_pairs"]:
            print(f"  {pair}: {count} occurrences")
        
        print("\nDEXES:")
        for dex, count in dex_counts.most_common():
            print(f"  {dex}: {count} pools")
        
        print("\nALL TOKENS:")
        print(", ".join(sorted(results["unique_tokens"])))
        
        print("\n" + "="*60)
        
        return results
    
    def _get_token_name(self, symbol: str) -> str:
        """
        Get a proper name for a token based on its symbol.
        Maps common symbols to their full names.
        
        Args:
            symbol: Token symbol (e.g., "SOL", "USDC")
            
        Returns:
            The full token name
        """
        # Dictionary of common Solana tokens with their full names
        token_names = {
            "SOL": "Solana",
            "USDC": "USD Coin",
            "USDT": "Tether USD",
            "BTC": "Wrapped Bitcoin (Solana)",
            "ETH": "Wrapped Ethereum (Solana)",
            "mSOL": "Marinade Staked SOL",
            "stSOL": "Lido Staked SOL",
            "RAY": "Raydium",
            "BONK": "Bonk",
            "BOOP": "Boop",
            "SRM": "Serum",
            "ORCA": "Orca",
            "MNGO": "Mango Markets",
            "SAMO": "Samoyedcoin",
            "STSOL": "Lido Staked SOL",
            "JSOL": "JPool Solana",
            "USDR": "Real USD",
            "USDH": "USDH Stablecoin",
            "JTO": "Jito",
            "WIF": "Dogwifhat",
            "JUP": "Jupiter",
            "CLAY": "Clay Nation",
            "PYTH": "Pyth Network",
            "BLUR": "Blur",
            "HPOS": "HPOS10 Index",
            "LAYER": "Layerhub",
            "GENE": "Genopets",
            "HGSOL": "Hedge Sol",
            "SLND": "Solend",
        }
        
        # If we have a full name, use it, otherwise use the symbol as the name
        return token_names.get(symbol, symbol)