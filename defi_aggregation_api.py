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
        
        # Configure base URL
        self.base_url = base_url or "https://filotdefiapi.replit.app/api/v1"
        
        # Configure request delay for rate limiting (10 req/sec)
        self.request_delay = 0.1  # 100ms delay for 10 requests per second 
        
        # Set authentication headers - using Bearer token format
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
            response = requests.get(url, headers=self.headers, params=params)
            
            # Introduce delay to respect rate limits
            time.sleep(self.request_delay)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise ValueError("API authentication failed. Check your API key.")
            elif response.status_code == 429:
                raise ValueError("API rate limit exceeded. Please try again later.")
            elif response.status_code == 404:
                raise ValueError(f"Resource not found: {endpoint}")
            else:
                error_message = f"API request failed with status code {response.status_code}"
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict) and "message" in error_data:
                        error_message = f"{error_message}: {error_data['message']}"
                except:
                    error_message = f"{error_message}: {response.text}"
                
                raise ValueError(error_message)
                
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error: {str(e)}")
    
    def get_pools(self, limit: int = 50, page: int = 1, 
                 source: Optional[str] = None, token: Optional[str] = None,
                 sort: Optional[str] = None, order: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get liquidity pools with optional filtering.
        
        Args:
            limit: Number of pools to retrieve per page
            page: Page number for pagination
            source: Filter by DEX source (e.g., "Raydium", "Meteora", "Orca")
            token: Filter by token symbol
            sort: Field to sort by (e.g., "apr24h", "tvl")
            order: Sort order ("asc" or "desc")
            
        Returns:
            List of pool data
        """
        params: Dict[str, Any] = {"limit": limit, "page": page}
        
        if source:
            params["source"] = source
        if token:
            params["token"] = token
        if sort:
            params["sort"] = sort
        if order:
            params["order"] = order
        
        try:
            # Note: From testing we found the API returns a list directly, not an object with a 'pools' property
            result = self._make_request("pools", params)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "pools" in result:
                return result.get("pools", [])
            else:
                logger.warning(f"Unexpected API response format: {type(result)}")
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
        page = 1
        per_page = 10  # Smaller batch size to avoid rate limits
        
        logger.info(f"Fetching up to {max_pools} pools with rate limiting...")
        
        while len(all_pools) < max_pools:
            try:
                # Get a batch of pools
                pools = self.get_pools(limit=per_page, page=page, **kwargs)
                
                if not pools:
                    logger.info("No more pools to fetch or error occurred")
                    break
                
                all_pools.extend(pools)
                logger.info(f"Retrieved {len(pools)} pools (total so far: {len(all_pools)})")
                
                # If we didn't get a full page, we've reached the end
                if len(pools) < per_page:
                    break
                
                page += 1
                
            except Exception as e:
                logger.error(f"Error fetching pools batch: {str(e)}")
                break
        
        logger.info(f"Retrieved a total of {len(all_pools)} pools")
        return all_pools
    
    def get_pool_by_id(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific pool by its ID.
        
        Args:
            pool_id: The authentic base58 pool ID
            
        Returns:
            Pool data or None if not found
        """
        try:
            return self._make_request(f"pools/{pool_id}")
        except ValueError as e:
            logger.error(f"Failed to get pool {pool_id}: {str(e)}")
            return None
    
    def transform_pool_data(self, pool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform API pool data to match our application's data model.
        
        Args:
            pool: Raw pool data from the API
            
        Returns:
            Transformed pool data
        """
        try:
            # Extract token data
            tokens = pool.get('tokens', [])
            token1 = tokens[0] if len(tokens) > 0 else {}
            token2 = tokens[1] if len(tokens) > 1 else {}
            
            # Extract metrics
            metrics = pool.get('metrics', {})
            
            # Create transformed pool record
            transformed = {
                "id": pool.get('poolId', ''),
                "name": pool.get('name', ''),
                "dex": pool.get('source', 'Unknown'),
                "token1_symbol": token1.get('symbol', 'Unknown'),
                "token2_symbol": token2.get('symbol', 'Unknown'),
                "token1_address": token1.get('address', ''),
                "token2_address": token2.get('address', ''),
                # Use metrics if available, otherwise look for direct properties
                "liquidity": metrics.get('tvl', pool.get('tvl', 0)),
                "volume_24h": metrics.get('volumeUsd', pool.get('volumeUsd', 0)),
                "apr": metrics.get('apy24h', pool.get('apr24h', 0)),
                "apr_change_24h": 0,  # Would need historical data to calculate change
                "apr_change_7d": metrics.get('apy7d', pool.get('apr7d', 0)) - metrics.get('apy24h', pool.get('apr24h', 0)) 
                    if metrics.get('apy7d') and metrics.get('apy24h') else 0,
                "prediction_score": 65,  # Default prediction score
                "category": "Stable" if "USD" in pool.get('name', '') or "stablecoin" in pool.get('name', '').lower() else "Standard",
                "token1_price": token1.get('price', 0),
                "token2_price": token2.get('price', 0),
                "fee_percentage": metrics.get('fee', pool.get('fee', 0)) * 100,  # Convert to percentage
                "data_source": "Real-time DeFi API",
                # Store the original object for reference and additional data
                "raw_data": pool
            }
            
            return transformed
        except Exception as e:
            logger.error(f"Error transforming pool data: {str(e)}")
            logger.debug(f"Problem pool data: {json.dumps(pool)}")
            # Return a minimal valid record to avoid crashes
            return {
                "id": pool.get('poolId', 'unknown-id'),
                "name": pool.get('name', 'Unknown Pool'),
                "dex": pool.get('source', 'Unknown'),
                "token1_symbol": "Unknown",
                "token2_symbol": "Unknown",
                "liquidity": 0,
                "volume_24h": 0,
                "apr": 0,
                "prediction_score": 0,
                "category": "Unknown",
                "data_source": "DeFi API (incomplete)",
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