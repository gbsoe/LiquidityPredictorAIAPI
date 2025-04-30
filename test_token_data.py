"""
Test script to check token data retrieval
"""

import logging
import json
from defi_aggregation_api import DefiAggregationAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test token retrieval"""
    api = DefiAggregationAPI()
    
    logger.info("Retrieving all tokens...")
    tokens = api.get_all_tokens()
    logger.info(f"Retrieved {len(tokens)} tokens")
    
    if tokens:
        logger.info("First few tokens:")
        for token in tokens[:3]:
            logger.info(json.dumps(token, indent=2))
    
    logger.info("Creating token mapping...")
    token_map = api.get_token_mapping()
    logger.info(f"Created token map with {len(token_map)} entries")
    
    # Try to retrieve some common tokens
    common_symbols = ["SOL", "USDC", "BTC", "RAY", "mSOL"]
    for symbol in common_symbols:
        if symbol in token_map:
            logger.info(f"Found {symbol} in token map: {token_map[symbol]['symbol']}")
        else:
            logger.info(f"{symbol} not found in token map")
    
    # Test pool retrieval with token extraction
    logger.info("Retrieving pools...")
    pools = api.get_pools(limit=2)
    for pool in pools:
        transformed = api.transform_pool_data(pool)
        logger.info(f"Pool: {transformed['name']}")
        logger.info(f"  Token 1: {transformed['token1_symbol']}, Price: ${transformed['token1_price']}")
        logger.info(f"  Token 2: {transformed['token2_symbol']}, Price: ${transformed['token2_price']}")

if __name__ == "__main__":
    main()