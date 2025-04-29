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
        if not self.api_key:
            raise ValueError("API key is required. Set DEFI_API_KEY environment variable or provide api_key parameter.")
        
        # Configure base URL - using the base URL from the documentation
        self.base_url = base_url or "https://filotdefiapi.replit.app/api/v1"
        logger.info(f"Using API base URL: {self.base_url}")
        
        # Configure request delay for rate limiting (10 req/sec)
        self.request_delay = 0.1  # 100ms delay for 10 requests per second 
        
        # Set authentication headers - using Bearer token authentication (confirmed by testing)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"DeFi Aggregation API client initialized with base URL: {self.base_url}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Make a rate-limited request to the API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            API response data (can be Dict or List depending on endpoint)
        
        Raises:
            ValueError: For various API errors with specific messages
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
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
        # Updated params based on API docs
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        
        if source:
            params["source"] = source.lower()  # API expects lowercase DEX names
        if token:
            params["token"] = token
        if sort:
            params["sort"] = sort
        if order:
            params["order"] = order
        
        try:
            # Based on updated docs, API returns dict with 'pools' property
            result = self._make_request("pools", params)
            
            # Handle different response formats
            if isinstance(result, dict) and "pools" in result:
                logger.info(f"Received {len(result.get('pools', []))} pools from API")
                return result.get("pools", [])
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
        per_page = 10  # Smaller batch size to avoid rate limits
        
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
                if len(pools) < per_page:
                    break
                
                # Increment offset for next page
                offset += per_page
                
            except Exception as e:
                logger.error(f"Error fetching pools batch: {str(e)}")
                break
        
        logger.info(f"Retrieved a total of {len(all_pools)} pools")
        return all_pools
    
    def get_pool_by_id(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific pool by its ID.
        Based on the GET Pool Docs which provides endpoints in the format:
        /api/v1/pools/{poolId}
        
        Args:
            pool_id: The authentic base58 pool ID
            
        Returns:
            Pool data or None if not found, transformed to application format
        """
        try:
            # Make the API request using the documented endpoint format
            logger.info(f"Fetching individual pool data for {pool_id}")
            
            try:
                # First try the specific pool endpoint
                response = self._make_request(f"pools/{pool_id}")
            except ValueError as e:
                if "Resource not found" in str(e):
                    # If the specific endpoint fails, try to find it in the main list
                    logger.info(f"Specific pool endpoint failed, trying to find pool in main list")
                    all_pools = self.get_all_pools(max_pools=200)  # Try to get a large sample
                    
                    # Search for this specific pool ID
                    for pool in all_pools:
                        if pool.get('id') == pool_id or pool.get('poolId') == pool_id:
                            logger.info(f"Found pool {pool_id} in main pool list")
                            return self.transform_pool_data(pool)
                    
                    # If we get here, we didn't find it
                    logger.warning(f"Pool {pool_id} not found in API responses")
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
            
            # Check response format - might be a single object or a list with one item
            # Based on the API documentation, it should be a single pool object
            if isinstance(response, list) and len(response) > 0:
                # If we got a list, use the first item (this handles both API response formats)
                pool_data = response[0]
                # Transform to our standard format
                return self.transform_pool_data(pool_data)
            elif isinstance(response, dict):
                # If we got a direct object, transform it
                # This is the expected format based on the documentation
                return self.transform_pool_data(response)
            else:
                logger.warning(f"Unexpected response format for pool {pool_id}")
                return None
        except ValueError as e:
            logger.error(f"Failed to get pool {pool_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching pool {pool_id}: {str(e)}")
            return None
    
    def transform_pool_data(self, pool: Dict[str, Any]) -> Dict[str, Any]:
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
            token1 = tokens[0] if len(tokens) > 0 else {}
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
            
            # Determine category based on token pair
            category = "Unknown"
            token1_symbol = token1.get('symbol', '').upper()
            token2_symbol = token2.get('symbol', '').upper()
            
            if "USD" in token1_symbol or "USD" in token2_symbol or "USDT" in token1_symbol or "USDT" in token2_symbol:
                category = "Stablecoin"
            elif token1_symbol == "SOL" or token2_symbol == "SOL":
                category = "Major"
            elif any(meme in token1_symbol.lower() or meme in token2_symbol.lower() for meme in ['doge', 'pepe', 'shib', 'meme']):
                category = "Meme"
            else:
                category = "DeFi"
                
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
            # Return a minimal valid record to avoid crashes, but with the same structure
            # as our comprehensive transformed data for consistency
            pool_id = pool.get('poolId', 'unknown-id')
            
            # Create current timestamps for fallback
            current_time = datetime.now().isoformat()
            
            return {
                # Core pool identifiers
                "id": pool_id,
                "name": pool_id,  # Use pool ID as the name
                "dex": pool.get('source', 'Unknown'),
                "category": "Unknown",
                
                # Token information (legacy format)
                "token1_symbol": "Unknown",
                "token2_symbol": "Unknown",
                "token1_address": "",
                "token2_address": "",
                "token1_price": 0,
                "token2_price": 0,
                
                # Complete token objects (new format)
                "tokens": [
                    {
                        "id": 0,
                        "symbol": "Unknown",
                        "name": "Unknown",
                        "address": "",
                        "decimals": 0,
                        "price": 0,
                        "active": True
                    },
                    {
                        "id": 0,
                        "symbol": "Unknown",
                        "name": "Unknown",
                        "address": "",
                        "decimals": 0,
                        "price": 0,
                        "active": True
                    }
                ],
                
                # Technical identifiers
                "poolId": pool.get('poolId', ''),
                "programId": pool.get('programId', ''),
                
                # Core metrics
                "liquidity": 0,
                "volume_24h": 0,
                "apr": 0,
                "fee": 0,
                
                # Time-based APR values
                "apr_24h": 0,
                "apr_7d": 0,
                "apr_30d": 0,
                
                # Change metrics
                "apr_change_24h": 0,
                "apr_change_7d": 0,
                "apr_change_30d": 0,
                "tvl_change_24h": 0,
                "tvl_change_7d": 0,
                "tvl_change_30d": 0,
                
                # Status and timestamps
                "active": True,
                "created_at": current_time,
                "updated_at": current_time,
                
                # Extra data
                "extra_data": {},
                "version": "1.0",
                
                # Prediction fields
                "prediction_score": 0,
                "risk_score": 0,
                "volatility": 0,
                
                # DEX specific fields
                "dex_specific": {
                    "ammId": "",
                    "concentrationBounds": "",
                    "whirlpoolId": ""
                },
                
                # Original data references (empty for fallback)
                "metrics_data": {},
                "api_response": {
                    "id": "",
                    "poolId": pool_id,
                    "programId": "",
                    "source": pool.get('source', 'Unknown'),
                    "name": pool_id,
                    "active": True,
                    "metrics": {}
                }
            }
    
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
            transformed_pools.append(transformed)
        
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
            import schedule
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