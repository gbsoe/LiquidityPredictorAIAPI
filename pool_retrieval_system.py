import os
import json
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Set, Tuple
import base58
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pool_retrieval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pool_retrieval")

# Constants
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY =.5  # seconds
MAX_CONCURRENT_REQUESTS = 20

# Pool data model
@dataclass
class PoolData:
    id: str
    name: str 
    dex: str
    token1_symbol: str
    token2_symbol: str
    token1_address: str
    token2_address: str
    liquidity: float
    volume_24h: float
    apr: float
    version: str = ""
    category: str = ""
    fee: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PoolRetriever:
    """
    Comprehensive system for retrieving pool data from multiple sources:
    1. DEX APIs (Raydium, Orca, Jupiter)
    2. Solana RPC endpoints
    3. Program account scanning
    4. Known pool registries
    """
    
    def __init__(self, 
                 api_keys: Dict[str, str] = None, 
                 rpc_endpoint: str = None):
        """
        Initialize the pool retriever with necessary credentials.
        
        Args:
            api_keys: Dictionary mapping service names to API keys
            rpc_endpoint: Solana RPC endpoint
        """
        self.api_keys = api_keys or {}
        self.rpc_endpoint = rpc_endpoint or os.getenv("SOLANA_RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
        self.session = self._create_session()
        self.known_pools: Dict[str, PoolData] = {}
        self.dex_programs = {
            "Raydium": [
                "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # v4
                "27haf8L6oxUeXrHrgEgsexjSY5hbVUWEmvv9Nyxg8vQv",  # v3
            ],
            "Orca": [
                "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",  # Whirlpool program
                "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP"  # v2 pools
            ],
            "Jupiter": [
                "JUP2jxvXaqu7NQY1GmNF4m1vodw12LVXYxbFL2uJvfo",  # Jupiter router
            ],
            "Saber": [
                "SSwpkEEcbUqx4vtoEByFjSkhKdCT862DNVb52nZg1UZ"  # Saber Stable Swap
            ],
            "Meteora": [
                "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K"  # Meteora program
            ],
            "Lifinity": [
                "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"  # Lifinity AMM
            ],
            "Crema": [
                "CRMxmz2wQCTbQZ1zaP3uHRzs4pqjYekpgLdeTRYkw8A"  # Crema Finance
            ],
            "Stepn": [
                "LocktDzaV1W2Bm9DeZeiyz4J9zs4fRqNiYqQyracRXw"  # Stepn DEX
            ],
            "Cykura": [
                "cysPXAjehMpVKUapzbMCCnpFxUFFryEWEaLgnb9NrR8"  # Cykura pools
            ]
        }
        
    def _create_session(self) -> requests.Session:
        """Create and configure a requests session for API calls."""
        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SolanaPoolAnalysis/1.0"
        })
        return session
    
    def _make_request(self, url: str, method: str = "get", 
                     headers: Dict[str, str] = None, 
                     params: Dict[str, Any] = None,
                     data: Dict[str, Any] = None,
                     service: str = None) -> Dict[str, Any]:
        """
        Make a request to an API with retry logic.
        
        Args:
            url: The URL to request
            method: HTTP method (get, post)
            headers: Additional headers
            params: URL parameters
            data: JSON data for POST requests
            service: Service name for API key lookup
            
        Returns:
            JSON response data
        """
        if service and service in self.api_keys:
            if not headers:
                headers = {}
            headers["API-Key"] = self.api_keys[service]
            
        for attempt in range(MAX_RETRIES):
            try:
                if method.lower() == "get":
                    response = self.session.get(
                        url,
                        headers=headers,
                        params=params,
                        timeout=DEFAULT_TIMEOUT
                    )
                else:  # POST
                    response = self.session.post(
                        url,
                        headers=headers,
                        params=params,
                        json=data,
                        timeout=DEFAULT_TIMEOUT
                    )
                
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Request failed after {MAX_RETRIES} attempts: {url}")
                    raise
    
    def _solana_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """Make a call to the Solana RPC API."""
        data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        return self._make_request(self.rpc_endpoint, method="post", data=data)
    
    def _get_token_accounts(self, program_id: str, commitment: str = "confirmed") -> List[Dict[str, Any]]:
        """Get all token accounts for a specific program."""
        try:
            response = self._solana_rpc_call(
                "getProgramAccounts",
                [
                    program_id,
                    {
                        "encoding": "jsonParsed",
                        "commitment": commitment
                    }
                ]
            )
            
            if "result" in response:
                return response["result"]
            else:
                logger.error(f"Unexpected response format: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting program accounts for {program_id}: {e}")
            return []
    
    def _parse_raydium_pool(self, account_data: Dict[str, Any]) -> Optional[PoolData]:
        """Parse Raydium pool account data into PoolData object."""
        try:
            # Simplified parsing - in a real implementation, this would need to decode
            # the specific structure of Raydium pool accounts
            # This is a placeholder for the complex binary parsing logic
            pool_id = account_data.get("pubkey", "")
            
            # In a real implementation: parsed_data = decode_raydium_account(account_data["account"]["data"])
            # For now, just return a skeleton with the ID
            return PoolData(
                id=pool_id,
                name="Unknown",
                dex="Raydium",
                token1_symbol="",
                token2_symbol="",
                token1_address="",
                token2_address="",
                liquidity=0.0,
                volume_24h=0.0,
                apr=0.0,
                version="v4"
            )
        except Exception as e:
            logger.error(f"Error parsing Raydium pool: {e}")
            return None
    
    def _fetch_raydium_pools_from_api(self) -> List[PoolData]:
        """Fetch Raydium pools from their API."""
        try:
            url = "https://api.raydium.io/v2/sdk/liquidity/mainnet.json"
            response = self._make_request(url)
            
            pools = []
            for item in response.get("official", []):
                try:
                    pool_id = item.get("id", "")
                    base_token = item.get("baseMint", {})
                    quote_token = item.get("quoteMint", {})
                    
                    pool = PoolData(
                        id=pool_id,
                        name=f"{base_token.get('symbol', '')}/{quote_token.get('symbol', '')}",
                        dex="Raydium",
                        token1_symbol=base_token.get("symbol", ""),
                        token2_symbol=quote_token.get("symbol", ""),
                        token1_address=base_token.get("address", ""),
                        token2_address=quote_token.get("address", ""),
                        liquidity=float(item.get("liquidity", 0)),
                        volume_24h=float(item.get("volume24h", 0)),
                        apr=float(item.get("apr", 0)),
                        version=item.get("version", "v4"),
                        fee=float(item.get("fee", 0.0025))
                    )
                    
                    pools.append(pool)
                    
                except Exception as e:
                    logger.warning(f"Error parsing Raydium pool data: {e}")
                    continue
            
            logger.info(f"Retrieved {len(pools)} pools from Raydium API")
            return pools
            
        except Exception as e:
            logger.error(f"Error fetching Raydium pools: {e}")
            return []
    
    def _fetch_orca_pools_from_api(self) -> List[PoolData]:
        """Fetch Orca pools from their Whirlpools API."""
        try:
            # This is a placeholder URL - Orca doesn't have a direct public API for pool listing
            # In practice, you'd use their SDK or scan their program accounts
            url = "https://api.orca.so/pools"  # Placeholder
            
            # Simulated response since there's no direct API
            # In a real implementation, this would come from the API
            pools = [
                PoolData(
                    id="HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ",
                    name="SOL/USDC",
                    dex="Orca",
                    token1_symbol="SOL",
                    token2_symbol="USDC",
                    token1_address="So11111111111111111111111111111111111111112",
                    token2_address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    liquidity=22345678.90,
                    volume_24h=7654321.09,
                    apr=13.56,
                    version="Whirlpool",
                    fee=0.003
                ),
                # More pools would be added here in a real implementation
            ]
            
            logger.info(f"Retrieved {len(pools)} pools from Orca")
            return pools
            
        except Exception as e:
            logger.error(f"Error fetching Orca pools: {e}")
            return []
    
    def _fetch_jupiter_pools(self) -> List[PoolData]:
        """
        Fetch pool information from Jupiter API
        Jupiter aggregates pools from multiple DEXes
        """
        try:
            url = "https://quote-api.jup.ag/v6/indexed-route-map"
            response = self._make_request(url)
            
            # Processing the Jupiter response would go here
            # This is complex as Jupiter returns routes rather than direct pool info
            
            pools = []
            # Simplified example of what we'd extract from Jupiter data
            # In a real implementation, this would parse the actual response format
            for market_info in response.get("marketInfos", []):
                pool_id = market_info.get("id", "")
                amm = market_info.get("amm", {})
                
                pool = PoolData(
                    id=pool_id,
                    name=f"{amm.get('inputMint', '')[:4]}/{amm.get('outputMint', '')[:4]}",
                    dex=amm.get("label", "Jupiter"),
                    token1_symbol=amm.get("inputSymbol", ""),
                    token2_symbol=amm.get("outputSymbol", ""),
                    token1_address=amm.get("inputMint", ""),
                    token2_address=amm.get("outputMint", ""),
                    liquidity=float(amm.get("liquidity", 0)),
                    volume_24h=float(amm.get("volume24h", 0)),
                    apr=float(amm.get("apr", 0)),
                    version="v6"
                )
                
                pools.append(pool)
            
            logger.info(f"Retrieved {len(pools)} pools from Jupiter")
            return pools
            
        except Exception as e:
            logger.error(f"Error fetching Jupiter pools: {e}")
            return []
    
    def _fetch_program_accounts(self, dex: str) -> List[PoolData]:
        """Fetch all pool accounts from a specific DEX program."""
        pools = []
        
        for program_id in self.dex_programs.get(dex, []):
            try:
                accounts = self._get_token_accounts(program_id)
                
                # Process each account based on the DEX type
                for account in accounts:
                    try:
                        if dex == "Raydium":
                            pool = self._parse_raydium_pool(account)
                        elif dex == "Orca":
                            pool = self._parse_orca_pool(account)
                        # Add parsers for other DEXes
                        else:
                            pool = None
                            
                        if pool:
                            pools.append(pool)
                            
                    except Exception as e:
                        logger.warning(f"Error parsing {dex} account: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error fetching accounts for {dex} program {program_id}: {e}")
                continue
                
        logger.info(f"Retrieved {len(pools)} pools from {dex} program accounts")
        return pools
    
    def _categorize_pool(self, pool: PoolData) -> str:
        """
        Categorize a pool based on its tokens.
        
        Categories:
        - Major: Pools with major tokens (SOL, BTC, ETH, etc.)
        - Stablecoin: Pools with stablecoins
        - Meme: Pools with known meme tokens
        - DeFi: Pools with DeFi protocol tokens
        - Gaming: Pools with gaming tokens
        - Other: Everything else
        """
        # Lists of tokens by category
        major_tokens = ["SOL", "BTC", "ETH", "BNB", "ADA", "AVAX"]
        stablecoins = ["USDC", "USDT", "DAI", "BUSD", "USDH", "USDR", "TUSD"]
        meme_tokens = ["BONK", "SAMO", "DOGWIFHAT", "POPCAT", "BIGMONKEY", "WIF"]
        defi_tokens = ["RAY", "JUP", "MNGO", "MARI", "ORCA", "TULIP", "STEP", "SRM"]
        gaming_tokens = ["AURORY", "STAR", "ATLAS", "POLIS", "GARI", "COPE"]
        
        # Check token categories
        token1 = pool.token1_symbol.upper()
        token2 = pool.token2_symbol.upper()
        
        # Stablecoin pairs
        if token1 in stablecoins and token2 in stablecoins:
            return "Stablecoin"
            
        # Meme token pairs
        if token1 in meme_tokens or token2 in meme_tokens:
            return "Meme"
            
        # Gaming token pairs
        if token1 in gaming_tokens or token2 in gaming_tokens:
            return "Gaming"
            
        # DeFi token pairs
        if token1 in defi_tokens or token2 in defi_tokens:
            return "DeFi"
            
        # Major pairs
        if token1 in major_tokens or token2 in major_tokens:
            return "Major"
            
        # Default category
        return "Other"
    
    def _deduplicate_pools(self, pools: List[PoolData]) -> List[PoolData]:
        """Remove duplicate pools based on ID."""
        unique_pools = {}
        
        for pool in pools:
            if pool.id not in unique_pools:
                unique_pools[pool.id] = pool
            else:
                # If we have duplicate, keep the one with more complete information
                existing = unique_pools[pool.id]
                
                # Create a scoring system for pool data completeness
                existing_score = sum(1 for v in asdict(existing).values() if v)
                new_score = sum(1 for v in asdict(pool).values() if v)
                
                if new_score > existing_score:
                    unique_pools[pool.id] = pool
        
        return list(unique_pools.values())
    
    def _enrich_pool_data(self, pools: List[PoolData]) -> List[PoolData]:
        """Enrich pool data with additional information."""
        for pool in pools:
            if not pool.category:
                pool.category = self._categorize_pool(pool)
                
            # In a real implementation, we would enrich with more data:
            # - Token prices
            # - Historical APR
            # - On-chain metrics
            # - etc.
        
        return pools
    
    def get_all_pools(self) -> List[PoolData]:
        """Retrieve pools from all available sources."""
        all_pools = []
        
        # 1. Get pools from direct DEX APIs
        all_pools.extend(self._fetch_raydium_pools_from_api())
        all_pools.extend(self._fetch_orca_pools_from_api())
        all_pools.extend(self._fetch_jupiter_pools())
        
        # 2. Get pools by scanning program accounts
        for dex in self.dex_programs.keys():
            all_pools.extend(self._fetch_program_accounts(dex))
        
        # 3. Deduplicate pools
        unique_pools = self._deduplicate_pools(all_pools)
        
        # 4. Enrich pool data
        enriched_pools = self._enrich_pool_data(unique_pools)
        
        logger.info(f"Retrieved {len(enriched_pools)} unique pools in total")
        
        # Store pools for future reference
        self.known_pools = {pool.id: pool for pool in enriched_pools}
        
        return enriched_pools
    
    def get_pools_by_token(self, token_symbol: str) -> List[PoolData]:
        """Get all pools containing a specific token."""
        # If we don't have pools yet, fetch them
        if not self.known_pools:
            self.get_all_pools()
            
        token_symbol = token_symbol.upper()
        matching_pools = []
        
        for pool in self.known_pools.values():
            if (pool.token1_symbol.upper() == token_symbol or 
                pool.token2_symbol.upper() == token_symbol):
                matching_pools.append(pool)
                
        logger.info(f"Found {len(matching_pools)} pools containing {token_symbol}")
        return matching_pools
    
    def get_pools_by_dex(self, dex: str) -> List[PoolData]:
        """Get all pools from a specific DEX."""
        # If we don't have pools yet, fetch them
        if not self.known_pools:
            self.get_all_pools()
            
        matching_pools = [pool for pool in self.known_pools.values() 
                         if pool.dex.lower() == dex.lower()]
                
        logger.info(f"Found {len(matching_pools)} pools from {dex}")
        return matching_pools
    
    def get_pools_by_category(self, category: str) -> List[PoolData]:
        """Get all pools in a specific category."""
        # If we don't have pools yet, fetch them
        if not self.known_pools:
            self.get_all_pools()
            
        matching_pools = [pool for pool in self.known_pools.values() 
                         if pool.category.lower() == category.lower()]
                
        logger.info(f"Found {len(matching_pools)} pools in category {category}")
        return matching_pools
    
    def get_top_pools_by_metric(self, metric: str, limit: int = 10) -> List[PoolData]:
        """
        Get top pools by a specific metric.
        
        Args:
            metric: 'liquidity', 'volume_24h', or 'apr'
            limit: Maximum number of pools to return
            
        Returns:
            List of top pools
        """
        # If we don't have pools yet, fetch them
        if not self.known_pools:
            self.get_all_pools()
            
        valid_metrics = ['liquidity', 'volume_24h', 'apr']
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
            
        sorted_pools = sorted(
            self.known_pools.values(),
            key=lambda x: getattr(x, metric, 0),
            reverse=True
        )
        
        return sorted_pools[:limit]
    
    def apply_filters(self, 
                      min_liquidity: float = None,
                      max_liquidity: float = None,
                      min_apr: float = None,
                      max_apr: float = None,
                      min_volume: float = None,
                      max_volume: float = None,
                      dexes: List[str] = None,
                      categories: List[str] = None,
                      tokens: List[str] = None) -> List[PoolData]:
        """
        Apply multiple filters to find pools matching criteria.
        
        Args:
            min_liquidity: Minimum liquidity value
            max_liquidity: Maximum liquidity value
            min_apr: Minimum APR percentage
            max_apr: Maximum APR percentage
            min_volume: Minimum 24h volume
            max_volume: Maximum 24h volume
            dexes: List of DEXes to include
            categories: List of categories to include
            tokens: List of tokens that must be in the pool
            
        Returns:
            List of matching pools
        """
        # If we don't have pools yet, fetch them
        if not self.known_pools:
            self.get_all_pools()
            
        filtered_pools = []
        
        for pool in self.known_pools.values():
            # Apply numeric filters
            if min_liquidity is not None and pool.liquidity < min_liquidity:
                continue
                
            if max_liquidity is not None and pool.liquidity > max_liquidity:
                continue
                
            if min_apr is not None and pool.apr < min_apr:
                continue
                
            if max_apr is not None and pool.apr > max_apr:
                continue
                
            if min_volume is not None and pool.volume_24h < min_volume:
                continue
                
            if max_volume is not None and pool.volume_24h > max_volume:
                continue
                
            # Apply DEX filter
            if dexes and pool.dex not in dexes:
                continue
                
            # Apply category filter
            if categories and pool.category not in categories:
                continue
                
            # Apply token filter - pool must contain ALL specified tokens
            if tokens:
                pool_tokens = [pool.token1_symbol.upper(), pool.token2_symbol.upper()]
                if not all(token.upper() in pool_tokens for token in tokens):
                    continue
                    
            # If we got here, the pool passed all filters
            filtered_pools.append(pool)
            
        logger.info(f"Applied filters: found {len(filtered_pools)} matching pools")
        return filtered_pools
    
    def export_pools_to_json(self, pools: List[PoolData], filename: str) -> None:
        """Export pools to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump([pool.to_dict() for pool in pools], f, indent=2)
                
            logger.info(f"Exported {len(pools)} pools to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting pools to {filename}: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize retriever with API keys if available
    api_keys = {
        "raydium": os.getenv("RAYDIUM_API_KEY", ""),
        "jupiter": os.getenv("JUPITER_API_KEY", "")
    }
    
    rpc_endpoint = os.getenv("SOLANA_RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
    
    retriever = PoolRetriever(api_keys, rpc_endpoint)
    
    # Get all pools
    all_pools = retriever.get_all_pools()
    print(f"Retrieved {len(all_pools)} total pools")
    
    # Apply filters for high-APR pools
    high_apr_pools = retriever.apply_filters(
        min_apr=20.0,
        min_liquidity=1000000,  # $1M minimum liquidity
        categories=["Meme", "DeFi"]
    )
    
    print(f"Found {len(high_apr_pools)} high-APR pools")
    
    # Export to JSON
    retriever.export_pools_to_json(high_apr_pools, "high_apr_pools.json")