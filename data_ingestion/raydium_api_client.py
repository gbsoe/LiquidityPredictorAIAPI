import sys
import os
import requests
import time
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('raydium_api_client')

class RaydiumAPIClient:
    """Client for interacting with the Raydium API Service"""
    
    def __init__(self, api_key=None, base_url=None):
        """Initialize the Raydium API client.
        
        Args:
            api_key: Raydium API key (defaults to config.RAYDIUM_API_KEY)
            base_url: API base URL (defaults to config.RAYDIUM_API_URL)
        """
        self.api_key = api_key or config.RAYDIUM_API_KEY
        self.base_url = base_url or config.RAYDIUM_API_URL
        
        # Create a session for better performance with multiple requests
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        })
        
        logger.info(f"Initialized Raydium API client with base URL: {self.base_url}")
    
    def make_request_with_retry(self, url_path, method='get', params=None, max_retries=3):
        """Make a request to the API with retry logic.
        
        Args:
            url_path: API endpoint path
            method: HTTP method (currently only 'get' is supported)
            params: Query parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            JSON response data
        """
        retries = 0
        full_url = f"{self.base_url}{url_path}"
        
        while retries < max_retries:
            try:
                if method.lower() == 'get':
                    response = self.session.get(
                        full_url, 
                        params=params,
                        timeout=10
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if (
                    hasattr(e, 'response') and 
                    e.response is not None and
                    (e.response.status_code == 429 or e.response.status_code >= 500) and
                    retries < max_retries - 1
                ):
                    # Retry on rate limit (429) or server errors (5xx)
                    retries += 1
                    logger.warning(f"Retry attempt {retries} for {url_path}")

                    # Exponential backoff
                    delay = 2 ** retries
                    time.sleep(delay)
                else:
                    # Log detailed error information
                    if hasattr(e, 'response') and e.response is not None:
                        logger.error(f"API error ({e.response.status_code}): {e.response.text}")
                    else:
                        logger.error(f"Request error: {str(e)}")
                    raise
    
    def get_all_pools(self):
        """Get all liquidity pools from Raydium.
        
        Returns:
            Dictionary with pools data
        """
        try:
            response = self.make_request_with_retry('/api/pools')
            
            # Handle case where response might be None
            if response is None:
                logger.warning("Received None response when fetching pools")
                return []
                
            pools_dict = response.get("pools", {})
            if not isinstance(pools_dict, dict):
                pools_dict = {}
                
            best_performance = pools_dict.get("bestPerformance", [])
            top_stable = pools_dict.get("topStable", [])
            
            logger.info(f"Retrieved {len(best_performance)} best performance pools")
            logger.info(f"Retrieved {len(top_stable)} top stable pools")
            
            # Combine all pools
            all_pools = best_performance + top_stable
            
            return all_pools
        except Exception as e:
            logger.error(f"Error fetching all pools: {str(e)}")
            return []
    
    def get_pool_by_id(self, pool_id):
        """Get detailed information for a specific pool.
        
        Args:
            pool_id: Pool ID
            
        Returns:
            Dictionary with pool details
        """
        try:
            response = self.make_request_with_retry(f'/api/pool/{pool_id}')
            
            # Handle case where response might be None
            if response is None:
                logger.warning(f"Received None response when fetching pool {pool_id}")
                return {}
                
            pool_data = response.get("pool", {})
            if not isinstance(pool_data, dict):
                pool_data = {}
            
            logger.info(f"Retrieved details for pool: {pool_id}")
            return pool_data
        except Exception as e:
            logger.error(f"Error fetching pool {pool_id}: {str(e)}")
            return {}
    
    def get_filtered_pools(self, token_symbol=None, min_apr=None, max_apr=None, min_liquidity=None, limit=10):
        """Filter pools based on criteria.
        
        Args:
            token_symbol: Filter pools containing this token
            min_apr: Minimum APR percentage
            max_apr: Maximum APR percentage
            min_liquidity: Minimum liquidity in USD
            limit: Maximum number of results
            
        Returns:
            List of pools matching the criteria
        """
        try:
            params = {
                "limit": limit
            }
            
            if token_symbol:
                params["tokenSymbol"] = token_symbol
            if min_apr is not None:
                params["minApr"] = min_apr
            if max_apr is not None:
                params["maxApr"] = max_apr
            if min_liquidity is not None:
                params["minLiquidity"] = min_liquidity
            
            response = self.make_request_with_retry('/api/filter', params=params)
            
            # Handle case where response might be None
            if response is None:
                logger.warning("Received None response when filtering pools")
                return []
                
            pools = response.get("pools", [])
            if not isinstance(pools, list):
                pools = []
                
            count = response.get("count", 0)
            
            logger.info(f"Found {count} pools matching the criteria")
            return pools
        except Exception as e:
            logger.error(f"Error filtering pools: {str(e)}")
            return []
    
    def get_pool_metrics(self, pool_id):
        """Get performance metrics for a specific pool.
        
        Args:
            pool_id: Pool ID
            
        Returns:
            Dictionary with pool metrics
        """
        try:
            response = self.make_request_with_retry(f'/api/metrics/{pool_id}')
            
            # Handle case where response might be None
            if response is None:
                logger.warning(f"Received None response when fetching metrics for pool {pool_id}")
                return {}
                
            metrics = response.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            
            logger.info(f"Retrieved metrics for pool: {pool_id}")
            return metrics
        except Exception as e:
            logger.error(f"Error fetching metrics for pool {pool_id}: {str(e)}")
            return {}
    
    def get_blockchain_stats(self):
        """Get current Solana blockchain statistics.
        
        Returns:
            Dictionary with blockchain statistics
        """
        try:
            response = self.make_request_with_retry('/api/blockchain/stats')
            
            # Handle case where response might be None
            if response is None:
                logger.warning("Received None response when fetching blockchain stats")
                return {}
                
            stats = response.get("stats", {})
            if not isinstance(stats, dict):
                stats = {}
            
            logger.info("Retrieved blockchain statistics")
            return stats
        except Exception as e:
            logger.error(f"Error fetching blockchain stats: {str(e)}")
            return {}

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = RaydiumAPIClient()
    
    # Test connection by getting all pools
    pools = client.get_all_pools()
    print(f"Retrieved {len(pools)} pools in total")
    
    if pools and len(pools) > 0:
        # Get details for the first pool
        first_pool = pools[0]
        if isinstance(first_pool, dict):
            first_pool_id = first_pool.get("id")
            if first_pool_id:
                pool_details = client.get_pool_by_id(first_pool_id)
                pool_name = pool_details.get('name', 'Unknown')
                print(f"First pool name: {pool_name}")
            else:
                print("First pool has no ID")
        else:
            print("First pool is not a dictionary")