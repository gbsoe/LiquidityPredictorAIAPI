# Simple API Test Script
# This script tests the basic functionality of the SolPool Insight API

import requests
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_test')

# API configuration
API_KEY = "dev_api_key_solpool_insight"  # Default development key
BASE_URL = "http://localhost:5100/api"  # Local testing URL

def test_health_check():
    """
    Test the health check endpoint (no auth required)
    """
    try:
        response = requests.get(f"{BASE_URL}/health")
        logger.info(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"API Version: {data.get('version')}")
            logger.info(f"API Status: {data.get('status')}")
            return True
        else:
            logger.error(f"Health check failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing health check: {str(e)}")
        return False

def test_api_docs():
    """
    Test the API documentation endpoint (no auth required)
    """
    try:
        response = requests.get(f"{BASE_URL}/docs")
        logger.info(f"API docs status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            endpoints = data.get('endpoints', [])
            logger.info(f"API Version: {data.get('api_version')}")
            logger.info(f"Available endpoints: {len(endpoints)}")
            return True
        else:
            logger.error(f"API docs failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing API docs: {str(e)}")
        return False

def test_get_pools():
    """
    Test the get pools endpoint (requires auth)
    """
    try:
        headers = {
            "X-API-Key": API_KEY
        }
        response = requests.get(f"{BASE_URL}/pools", headers=headers)
        logger.info(f"Get pools status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            pools = data.get('pools', [])
            logger.info(f"Retrieved {len(pools)} pools")
            if pools:
                # Print the first pool as a sample
                logger.info(f"Sample pool: {json.dumps(pools[0], indent=2)}")
            return True
        else:
            logger.error(f"Get pools failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing get pools: {str(e)}")
        return False

def test_top_predictions():
    """
    Test the top predictions endpoint (requires auth)
    """
    try:
        headers = {
            "X-API-Key": API_KEY
        }
        params = {
            "category": "apr",
            "limit": 5
        }
        response = requests.get(f"{BASE_URL}/predictions/top", headers=headers, params=params)
        logger.info(f"Top predictions status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            logger.info(f"Retrieved {len(predictions)} predictions")
            if predictions:
                # Print the first prediction as a sample
                logger.info(f"Sample prediction: {json.dumps(predictions[0], indent=2)}")
            return True
        else:
            logger.error(f"Top predictions failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing top predictions: {str(e)}")
        return False

def run_all_tests():
    """
    Run all API tests and report results
    """
    logger.info("=== Starting API Tests ===")
    logger.info(f"Test time: {datetime.now().isoformat()}")
    logger.info(f"Base URL: {BASE_URL}")
    
    results = {
        "health_check": test_health_check(),
        "api_docs": test_api_docs(),
        "get_pools": test_get_pools(),
        "top_predictions": test_top_predictions()
    }
    
    # Report overall results
    success_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    logger.info(f"=== Test Results: {success_count}/{total_count} tests passed ===")
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {'✓ PASS' if result else '✗ FAIL'}")
    
    logger.info("=== Tests Completed ===")
    
    return results

if __name__ == "__main__":
    run_all_tests()
