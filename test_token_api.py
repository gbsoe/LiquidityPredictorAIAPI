import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def test_tokens_endpoint():
    """Test the /tokens endpoint"""
    try:
        url = "https://filotdefiapi.replit.app/api/v1/tokens"
        logger.info(f"Testing endpoint: {url}")
        
        response = requests.get(url, timeout=10)
        logger.info(f"Status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                logger.info(f"Data type: {type(data)}")
                
                if isinstance(data, list):
                    logger.info(f"Found {len(data)} tokens")
                    if data:
                        logger.info(f"First token: {json.dumps(data[0], indent=2)}")
                elif isinstance(data, dict):
                    logger.info(f"Response is a dictionary with keys: {list(data.keys())}")
                    if 'data' in data and isinstance(data['data'], list):
                        logger.info(f"Found {len(data['data'])} tokens in data field")
                        if data['data']:
                            logger.info(f"First token: {json.dumps(data['data'][0], indent=2)}")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.info(f"Raw response: {response.text[:500]}")
        else:
            logger.error(f"Bad status code: {response.status_code}")
            logger.info(f"Response text: {response.text[:500]}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")

def test_token_by_symbol_endpoint(symbol):
    """Test the /tokens/{symbol} endpoint"""
    try:
        url = f"https://filotdefiapi.replit.app/api/v1/tokens/{symbol}"
        logger.info(f"Testing endpoint: {url}")
        
        response = requests.get(url, timeout=10)
        logger.info(f"Status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                logger.info(f"Data type: {type(data)}")
                logger.info(f"Response: {json.dumps(data, indent=2)}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.info(f"Raw response: {response.text[:500]}")
        else:
            logger.error(f"Bad status code: {response.status_code}")
            logger.info(f"Response text: {response.text[:500]}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    logger.info("\n=== Testing /tokens endpoint ===")
    test_tokens_endpoint()
    
    logger.info("\n=== Testing /tokens/SOL endpoint ===")
    test_token_by_symbol_endpoint("SOL")