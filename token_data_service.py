"""
Token Data Service

This module provides a unified interface for retrieving token information
from multiple sources, including CoinGecko and the Raydium Trader API.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
from coingecko_api import get_coingecko_api
from defi_api_client import DefiApiClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenDataService:
    """
    Service for retrieving and managing token data
    from multiple sources, prioritizing authentic data
    """
    
    def __init__(self):
        """Initialize the token data service"""
        self.coingecko = get_coingecko_api()
        
        # Try to initialize DeFi API client (fallback gracefully if not available)
        try:
            api_key = os.getenv("DEFI_API_KEY")
            self.defi_api = DefiApiClient(api_key=api_key) if api_key else None
        except Exception as e:
            logger.warning(f"Failed to initialize DeFi API client: {e}")
            self.defi_api = None
        
        # Token data caches
        self.token_data: Dict[str, Dict[str, Any]] = {}  # By symbol
        self.token_address_data: Dict[str, Dict[str, Any]] = {}  # By address
        
        # CoinGecko ID mappings
        self.coingecko_ids: Dict[str, str] = {}  # Symbol -> CoinGecko ID
        
        # Last update timestamps
        self.last_update_time: Dict[str, float] = {}
    
    def get_token_price(self, symbol: str, max_age_seconds: int = 300) -> Optional[float]:
        """
        Get the current price for a token by symbol
        
        Args:
            symbol: Token symbol (e.g., 'SOL', 'BTC')
            max_age_seconds: Maximum age of cached price in seconds
            
        Returns:
            Token price in USD or None if not available
        """
        # Normalize symbol
        symbol = symbol.upper()
        
        # Check if we have recent data
        now = time.time()
        if (
            symbol in self.token_data and
            'price' in self.token_data[symbol] and
            symbol in self.last_update_time and
            now - self.last_update_time[symbol] < max_age_seconds
        ):
            return self.token_data[symbol]['price']
        
        # First try CoinGecko for accurate price
        price = self._get_price_from_coingecko(symbol)
        if price is not None:
            self._update_token_data(symbol, {'price': price})
            return price
        
        # If CoinGecko fails, try DeFi API
        price = self._get_price_from_defi_api(symbol)
        if price is not None:
            self._update_token_data(symbol, {'price': price})
            return price
        
        # No price found
        logger.warning(f"Could not find price for token: {symbol}")
        return None
    
    def _get_price_from_coingecko(self, symbol: str) -> Optional[float]:
        """Get price from CoinGecko"""
        try:
            price = self.coingecko.get_price_by_symbol(symbol)
            if price:
                logger.info(f"Retrieved price for {symbol} by symbol lookup: {price}")
                
                # If we found a CoinGecko ID, add it to our mappings
                if symbol.upper() not in self.coingecko_ids and symbol.upper() in self.coingecko.token_id_mapping:
                    coin_id = self.coingecko.token_id_mapping[symbol.upper()]
                    self.coingecko_ids[symbol.upper()] = coin_id
                    logger.info(f"Added {symbol} to CoinGecko mappings with ID: {coin_id}")
                
            return price
        except Exception as e:
            logger.error(f"Error getting price from CoinGecko for {symbol}: {e}")
            return None
    
    def _get_price_from_defi_api(self, symbol: str) -> Optional[float]:
        """Get price from DeFi API"""
        if not self.defi_api:
            return None
        
        try:
            # Try to get token info directly if possible
            token_info = self.defi_api.get_token_information(symbol)
            if token_info and 'price' in token_info:
                price = float(token_info['price'])
                logger.info(f"Retrieved price for {symbol} from DeFi API: {price}")
                return price
            
            # If direct lookup fails, try to find the token in pool data
            pools_response = self.defi_api.get_all_pools(token=symbol, limit=20)
            pools = pools_response.get('pools', [])
            
            if pools:
                logger.info(f"Found {len(pools)} pools that might contain token {symbol}")
                
                # Look for the token in the pools' tokens
                for pool in pools:
                    tokens = pool.get('tokens', [])
                    for token in tokens:
                        if token.get('symbol', '').upper() == symbol.upper() and 'price' in token:
                            price = float(token['price'])
                            logger.info(f"Found price for {symbol} in pool data: {price}")
                            return price
            
            logger.warning(f"Token {symbol} not found in API")
            return None
        except Exception as e:
            logger.error(f"Error getting price from DeFi API for {symbol}: {e}")
            return None
    
    def get_token_address(self, symbol: str) -> Optional[str]:
        """
        Get the blockchain address for a token by symbol
        
        Args:
            symbol: Token symbol (e.g., 'SOL', 'BTC')
            
        Returns:
            Token address or None if not available
        """
        # Normalize symbol
        symbol = symbol.upper()
        
        # Check if we have the data cached
        if symbol in self.token_data and 'address' in self.token_data[symbol]:
            return self.token_data[symbol]['address']
        
        # Try to get from DeFi API first
        address = self._get_address_from_defi_api(symbol)
        if address:
            self._update_token_data(symbol, {'address': address})
            return address
        
        # If that fails, try to get from CoinGecko
        # This is less reliable for Solana tokens
        address = self._get_address_from_coingecko(symbol)
        if address:
            self._update_token_data(symbol, {'address': address})
            return address
        
        # No address found
        logger.warning(f"Could not find address for token: {symbol}")
        return None
    
    def _get_address_from_defi_api(self, symbol: str) -> Optional[str]:
        """Get token address from DeFi API"""
        if not self.defi_api:
            return None
        
        try:
            # Try to get token info directly if possible
            token_info = self.defi_api.get_token_information(symbol)
            if token_info and 'address' in token_info:
                address = token_info['address']
                logger.info(f"Retrieved address for {symbol} from DeFi API: {address}")
                return address
            
            # If direct lookup fails, try to find the token in pool data
            pools_response = self.defi_api.get_all_pools(token=symbol, limit=20)
            pools = pools_response.get('pools', [])
            
            if pools:
                logger.info(f"Found {len(pools)} pools that might contain token {symbol}")
                
                # Look for the token in the pools' tokens
                for pool in pools:
                    tokens = pool.get('tokens', [])
                    for token in tokens:
                        if token.get('symbol', '').upper() == symbol.upper() and 'address' in token:
                            address = token['address']
                            logger.info(f"Found address for {symbol} in pool data: {address}")
                            return address
            
            logger.warning(f"Token {symbol} not found in API")
            return None
        except Exception as e:
            logger.error(f"Error getting address from DeFi API for {symbol}: {e}")
            return None
    
    def _get_address_from_coingecko(self, symbol: str) -> Optional[str]:
        """Get token address from CoinGecko"""
        try:
            # First, get the CoinGecko ID for this symbol
            coin_id = self.coingecko.get_token_id(symbol)
            if not coin_id:
                return None
            
            # Then get detailed token info
            token_details = self.coingecko.get_token_details(coin_id)
            if not token_details:
                return None
            
            # Look for Solana platform addresses
            platforms = token_details.get('platforms', {})
            if 'solana' in platforms and platforms['solana']:
                address = platforms['solana']
                logger.info(f"Retrieved Solana address for {symbol} from CoinGecko: {address}")
                return address
            
            return None
        except Exception as e:
            logger.error(f"Error getting address from CoinGecko for {symbol}: {e}")
            return None
    
    def get_token_by_address(self, address: str) -> Dict[str, Any]:
        """
        Get token information by address
        
        Args:
            address: Token contract address
            
        Returns:
            Dictionary with token information
        """
        # Check if we have the data cached
        if address in self.token_address_data:
            return self.token_address_data[address]
        
        # Try to get from DeFi API first
        token_data = self._get_token_by_address_from_defi_api(address)
        if token_data:
            self.token_address_data[address] = token_data
            
            # Also store by symbol if available
            if 'symbol' in token_data:
                symbol = token_data['symbol'].upper()
                self._update_token_data(symbol, token_data)
            
            return token_data
        
        # If that fails, try to get from CoinGecko
        token_data = self._get_token_by_address_from_coingecko(address)
        if token_data:
            self.token_address_data[address] = token_data
            
            # Also store by symbol if available
            if 'symbol' in token_data:
                symbol = token_data['symbol'].upper()
                self._update_token_data(symbol, token_data)
            
            return token_data
        
        # Return empty data if nothing found
        logger.warning(f"Could not find token with address: {address}")
        return {}
    
    def _get_token_by_address_from_defi_api(self, address: str) -> Dict[str, Any]:
        """Get token info by address from DeFi API"""
        if not self.defi_api:
            return {}
        
        try:
            # The DeFi API typically doesn't support lookup by address directly
            # So we need to find the token in the pools
            # Note: This is inefficient but a workaround for the API limitations
            
            # Search in all pools (this is relatively expensive)
            pools_response = self.defi_api.get_all_pools(limit=50)
            pools = pools_response.get('pools', [])
            
            for pool in pools:
                tokens = pool.get('tokens', [])
                for token in tokens:
                    if token.get('address', '') == address:
                        # Found a match
                        result = {
                            'symbol': token.get('symbol', ''),
                            'name': token.get('name', ''),
                            'address': address,
                            'price': token.get('price', 0),
                            'decimals': token.get('decimals', 9)
                        }
                        logger.info(f"Found token with address {address} in pools: {result['symbol']}")
                        return result
            
            logger.warning(f"Token with address {address} not found in DeFi API")
            return {}
        except Exception as e:
            logger.error(f"Error searching for token address {address} in DeFi API: {e}")
            return {}
    
    def _get_token_by_address_from_coingecko(self, address: str) -> Dict[str, Any]:
        """Get token info by address from CoinGecko"""
        try:
            # First, get the CoinGecko ID for this address
            coin_id = self.coingecko.get_token_id_by_address(address)
            if not coin_id:
                return {}
            
            # Then get detailed token info
            token_details = self.coingecko.get_token_details(coin_id)
            if not token_details:
                return {}
            
            # Extract basic token info
            result = {
                'symbol': token_details.get('symbol', '').upper(),
                'name': token_details.get('name', ''),
                'address': address,
                'decimals': 9  # Default for Solana
            }
            
            # Get price
            market_data = token_details.get('market_data', {})
            if 'current_price' in market_data and 'usd' in market_data['current_price']:
                result['price'] = market_data['current_price']['usd']
            
            logger.info(f"Retrieved token info for address {address} from CoinGecko: {result['symbol']}")
            return result
        except Exception as e:
            logger.error(f"Error getting token details for address {address} from CoinGecko: {e}")
            return {}
    
    def _update_token_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Update the token data cache"""
        # Normalize symbol
        symbol = symbol.upper()
        
        # Initialize if needed
        if symbol not in self.token_data:
            self.token_data[symbol] = {}
        
        # Update token data
        for key, value in data.items():
            self.token_data[symbol][key] = value
        
        # Update timestamp
        self.last_update_time[symbol] = time.time()
    
    def preload_token_data(self, symbols: List[str]) -> None:
        """
        Preload token data for a list of symbols
        
        Args:
            symbols: List of token symbols to preload
        """
        logger.info(f"Preloading token data for {len(symbols)} symbols")
        for symbol in symbols:
            try:
                logger.info(f"Making API request for token: {symbol}")
                # Get token price (this will automatically cache the token data)
                self.get_token_price(symbol)
                
                # Get token address (this will also cache the address)
                self.get_token_address(symbol)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error preloading token data for {symbol}: {e}")
    
    def preload_common_tokens(self) -> None:
        """Preload data for common Solana tokens"""
        common_tokens = [
            "SOL", "USDC", "USDT", "ETH", "BTC", "RAY", "BONK", "SAMO", "MNGO", "SRM",
            "mSOL", "stSOL", "ORCA", "ATLAS", "POLIS", "JTO", "DUST", "GENE", "WIF", "JUP"
        ]
        self.preload_token_data(common_tokens)
        
    def preload_all_tokens(self) -> None:
        """Preload data for all common tokens plus additional popular ones"""
        # Start with common tokens
        self.preload_common_tokens()
        
        # Then try to load additional tokens if we have access to the DeFi API
        if self.defi_api:
            try:
                # Get some popular pools
                pools_response = self.defi_api.get_all_pools(limit=50, sort="tvl", order="desc")
                pools = pools_response.get('pools', [])
                
                # Extract token symbols from pools
                symbols = set()
                for pool in pools:
                    tokens = pool.get('tokens', [])
                    for token in tokens:
                        if 'symbol' in token:
                            symbols.add(token['symbol'].upper())
                
                # Preload data for these tokens (excluding ones we've already loaded)
                common_set = {token.upper() for token in [
                    "SOL", "USDC", "USDT", "ETH", "BTC", "RAY", "BONK", "SAMO", "MNGO", "SRM",
                    "mSOL", "stSOL", "ORCA", "ATLAS", "POLIS", "JTO", "DUST", "GENE", "WIF", "JUP"
                ]}
                additional_tokens = symbols - common_set
                
                logger.info(f"Preloading data for {len(additional_tokens)} additional tokens")
                self.preload_token_data(list(additional_tokens))
                logger.info(f"Completed preloading additional token data")
                
            except Exception as e:
                logger.error(f"Error preloading additional tokens: {e}")
                
    def get_token_data(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive token data by symbol
        
        Args:
            symbol: Token symbol (e.g., 'SOL', 'BTC')
            force_refresh: Whether to force a refresh from API
            
        Returns:
            Dictionary with token data
        """
        symbol = symbol.upper()
        
        # Check if we need to refresh the data
        if force_refresh or symbol not in self.token_data:
            # Create basic token data object
            token_data = {
                'symbol': symbol,
                'name': symbol,  # Default to symbol as name
                'decimals': 9,   # Default decimals for Solana tokens
                'price': 0,
                'address': '',
                'active': True,   # Default to active
                'id': 0,          # Default token ID
                'updated_at': time.time(),
                'price_source': 'none'
            }
            
            # First try to get detailed token info from CoinGecko
            if self.coingecko:
                try:
                    # Try to get CoinGecko ID
                    coin_id = self.coingecko.get_token_id(symbol)
                    if coin_id:
                        # Get token details
                        details = self.coingecko.get_token_details(coin_id)
                        if details:
                            # Set name
                            token_data['name'] = details.get('name', symbol)
                            
                            # Set ID
                            token_data['id'] = details.get('id', 0)
                            
                            # Set decimals if available
                            if 'detail_platforms' in details and 'solana' in details.get('detail_platforms', {}):
                                solana_details = details['detail_platforms']['solana']
                                token_data['decimals'] = solana_details.get('decimal_place', 9)
                            
                            # Set price
                            if 'market_data' in details and 'current_price' in details['market_data']:
                                token_data['price'] = details['market_data']['current_price'].get('usd', 0)
                                token_data['price_source'] = 'coingecko'
                            
                            # Set address
                            if 'platforms' in details and 'solana' in details['platforms']:
                                token_data['address'] = details['platforms']['solana']
                except Exception as e:
                    logger.warning(f"Error getting detailed token data from CoinGecko for {symbol}: {e}")
            
            # Get price if not already set from CoinGecko details
            if token_data['price'] == 0:
                price = self.get_token_price(symbol)
                if price:
                    token_data['price'] = price
                    if 'price_source' not in token_data or token_data['price_source'] == 'none':
                        token_data['price_source'] = 'defi_api'
            
            # Get address if not already set from CoinGecko details
            if not token_data['address']:
                address = self.get_token_address(symbol)
                if address:
                    token_data['address'] = address
            
            # Get DEXes that use this token
            token_data['dexes'] = []
            try:
                categories = self.get_token_categories()
                for dex, tokens in categories.items():
                    if symbol in tokens:
                        token_data['dexes'].append(dex)
            except Exception as e:
                logger.warning(f"Error getting DEXes for token {symbol}: {e}")
            
            # Update cache
            self._update_token_data(symbol, token_data)
        
        # Get the latest data from cache
        result = self.token_data.get(symbol, {})
        
        # Ensure active status is set
        if 'active' not in result:
            result['active'] = result.get('price', 0) > 0
        
        return result
        
    def get_token_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get token metadata by symbol
        
        Args:
            symbol: Token symbol (e.g., 'SOL', 'BTC')
            
        Returns:
            Dictionary with token metadata
        """
        # This is just an alias for get_token_data for backward compatibility
        return self.get_token_data(symbol)
    
    def get_all_tokens(self) -> List[Dict[str, Any]]:
        """
        Get all tokens available in the system
        
        Returns:
            List of token data dictionaries
        """
        tokens = []
        
        # First try to get tokens from DeFi API
        if self.defi_api:
            try:
                # Get tokens from pools to ensure we have a complete list
                pools_response = self.defi_api.get_all_pools(limit=100)
                pools = pools_response.get('pools', [])
                
                # Create a dictionary to track unique tokens
                token_dict = {}
                
                # Extract tokens from pools
                for pool in pools:
                    token_list = pool.get('tokens', [])
                    for token in token_list:
                        symbol = token.get('symbol', '').upper()
                        if symbol and symbol != 'UNKNOWN' and symbol not in token_dict:
                            # Create a standardized token entry
                            token_dict[symbol] = {
                                'symbol': symbol,
                                'name': token.get('name', symbol),
                                'address': token.get('address', ''),
                                'price': token.get('price', 0),
                                'decimals': token.get('decimals', 9),
                                'active': True
                            }
                
                # Convert dictionary to list
                tokens = list(token_dict.values())
                logger.info(f"Retrieved {len(tokens)} tokens from DeFi API pools")
            except Exception as e:
                logger.error(f"Error getting tokens from DeFi API: {e}")
        
        # If we don't have tokens from the API, use our local cache
        if not tokens and self.token_data:
            for symbol, data in self.token_data.items():
                token_entry = {
                    'symbol': symbol,
                    'name': data.get('name', symbol),
                    'address': data.get('address', ''),
                    'price': data.get('price', 0),
                    'decimals': data.get('decimals', 9),
                    'active': True
                }
                tokens.append(token_entry)
            logger.info(f"Retrieved {len(tokens)} tokens from local cache")
        
        # Try to enhance with CoinGecko data for tokens with missing info
        if self.coingecko:
            for token in tokens:
                symbol = token['symbol']
                # Only try to enhance tokens with missing data
                if token['price'] == 0 or not token['address']:
                    try:
                        # Try to get CoinGecko ID
                        coin_id = self.coingecko.get_token_id(symbol)
                        if coin_id:
                            # Get token details
                            details = self.coingecko.get_token_details(coin_id)
                            if details:
                                # Update price if missing
                                if token['price'] == 0 and 'market_data' in details and 'current_price' in details['market_data']:
                                    token['price'] = details['market_data']['current_price'].get('usd', 0)
                                    token['price_source'] = 'coingecko'
                                
                                # Update address if missing
                                if not token['address'] and 'platforms' in details and 'solana' in details['platforms']:
                                    token['address'] = details['platforms']['solana']
                                
                                # Add any missing name
                                if not token.get('name') or token['name'] == token['symbol']:
                                    token['name'] = details.get('name', token['symbol'])
                    except Exception as e:
                        logger.warning(f"Error enhancing token {symbol} with CoinGecko data: {e}")
        
        return tokens
    
    def get_token_categories(self) -> Dict[str, List[str]]:
        """
        Get token categories organized by DEX
        
        Returns:
            Dictionary mapping DEX names to lists of token symbols
        """
        categories = {}
        
        # Try to get categories from DeFi API
        if self.defi_api:
            try:
                # Get pools to extract DEX information
                pools_response = self.defi_api.get_all_pools(limit=100)
                pools = pools_response.get('pools', [])
                
                # Extract DEXes and tokens
                for pool in pools:
                    dex = pool.get('source', 'Unknown')
                    if dex == 'Unknown':
                        continue
                    
                    # Initialize category if needed
                    if dex not in categories:
                        categories[dex] = []
                    
                    # Add tokens from this pool
                    token_list = pool.get('tokens', [])
                    for token in token_list:
                        symbol = token.get('symbol', '').upper()
                        if symbol and symbol != 'UNKNOWN' and symbol not in categories[dex]:
                            categories[dex].append(symbol)
            except Exception as e:
                logger.error(f"Error getting token categories from DeFi API: {e}")
        
        # If we don't have data, create a simple fallback
        if not categories:
            categories = {
                'Raydium': ['SOL', 'USDC', 'USDT', 'RAY', 'BONK'],
                'Orca': ['SOL', 'USDC', 'USDT', 'ORCA'],
                'Meteora': ['SOL', 'USDC', 'USDT', 'MNGO']
            }
        
        return categories
    
    def get_tokens_by_dex(self, dex_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get tokens used by a specific DEX
        
        Args:
            dex_name: Name of the DEX
            
        Returns:
            Dictionary mapping token symbols to token data
        """
        tokens = {}
        
        # Try to get tokens from DeFi API
        if self.defi_api:
            try:
                # Get pools for this DEX
                pools_response = self.defi_api.get_all_pools(source=dex_name, limit=100)
                pools = pools_response.get('pools', [])
                
                # Extract tokens from pools
                for pool in pools:
                    token_list = pool.get('tokens', [])
                    for token in token_list:
                        symbol = token.get('symbol', '').upper()
                        if symbol and symbol != 'UNKNOWN' and symbol not in tokens:
                            # Create a standardized token entry
                            tokens[symbol] = {
                                'symbol': symbol,
                                'name': token.get('name', symbol),
                                'address': token.get('address', ''),
                                'price': token.get('price', 0),
                                'decimals': token.get('decimals', 9),
                                'active': True,
                                'dex': dex_name
                            }
            except Exception as e:
                logger.error(f"Error getting tokens for DEX {dex_name} from API: {e}")
        
        # If we don't have data, use token categories as fallback
        if not tokens:
            categories = self.get_token_categories()
            if dex_name in categories:
                for symbol in categories[dex_name]:
                    # Try to get token data from our cache
                    token_data = self.get_token_data(symbol)
                    tokens[symbol] = {
                        'symbol': symbol,
                        'name': token_data.get('name', symbol),
                        'address': token_data.get('address', ''),
                        'price': token_data.get('price', 0),
                        'decimals': token_data.get('decimals', 9),
                        'active': True,
                        'dex': dex_name
                    }
        
        return tokens

# Singleton instance
_instance = None

def get_token_data_service() -> TokenDataService:
    """
    Get a singleton instance of the token data service
    
    Returns:
        Token data service instance
    """
    global _instance
    if _instance is None:
        _instance = TokenDataService()
    return _instance


def get_token_service() -> TokenDataService:
    """
    Alias for get_token_data_service for backward compatibility
    
    Returns:
        Token data service instance
    """
    return get_token_data_service()


if __name__ == "__main__":
    # Example usage
    service = get_token_data_service()
    service.preload_common_tokens()
    
    # Get some token data
    sol_price = service.get_token_price("SOL")
    print(f"SOL price: ${sol_price}")
    
    usdc_address = service.get_token_address("USDC")
    print(f"USDC address: {usdc_address}")