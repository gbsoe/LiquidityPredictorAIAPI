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
            # Get price
            price = self.get_token_price(symbol)
            
            # Get address
            address = self.get_token_address(symbol)
            
            # Create token data object
            token_data = {
                'symbol': symbol,
                'price': price,
                'address': address,
                'updated_at': time.time()
            }
            
            # Update cache
            self._update_token_data(symbol, token_data)
            
        return self.token_data.get(symbol, {})
        
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


if __name__ == "__main__":
    # Example usage
    service = get_token_data_service()
    service.preload_common_tokens()
    
    # Get some token data
    sol_price = service.get_token_price("SOL")
    print(f"SOL price: ${sol_price}")
    
    usdc_address = service.get_token_address("USDC")
    print(f"USDC address: {usdc_address}")