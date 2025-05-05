# SolPool Insight API Endpoints

import os
import json
import logging
import time
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, abort
from database.db_operations import get_db_manager
from data_services.data_service import get_data_service
from token_price_service import get_token_price_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='api_server.log'
)
logger = logging.getLogger('api_endpoints')

# Initialize Flask app
app = Flask(__name__)

# Set up API key validation (simple implementation)
# In production, consider using a more secure approach
API_KEYS = [os.getenv('API_KEY', 'dev_api_key_solpool_insight')]

def validate_api_key():
    """
    Validate API key from request headers
    """
    api_key = request.headers.get('X-API-Key')
    if not api_key or api_key not in API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key}")
        abort(401, description="Invalid API key")
    return True

@app.before_request
def before_request():
    """
    Pre-request middleware for API key validation
    Skip validation for documentation routes
    """
    if request.path == '/api/docs' or request.path == '/api/health':
        return
    validate_api_key()

# Health check endpoint (no auth required)
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running
    """
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# API documentation endpoint (no auth required)
@app.route('/api/docs', methods=['GET'])
def api_docs():
    """
    API documentation endpoint
    """
    return jsonify({
        'api_version': '1.0.0',
        'description': 'SolPool Insight API provides data and predictions for Solana liquidity pools',
        'endpoints': [
            {
                'path': '/api/health',
                'method': 'GET',
                'description': 'Health check endpoint',
                'auth_required': False
            },
            {
                'path': '/api/pools',
                'method': 'GET',
                'description': 'Get list of all pools with basic metrics',
                'auth_required': True,
                'query_params': [
                    {'name': 'limit', 'type': 'integer', 'description': 'Max number of pools to return', 'default': 50},
                    {'name': 'dex', 'type': 'string', 'description': 'Filter by DEX name'}
                ]
            },
            {
                'path': '/api/pools/{pool_id}',
                'method': 'GET',
                'description': 'Get detailed data for a specific pool',
                'auth_required': True
            },
            {
                'path': '/api/pools/{pool_id}/metrics',
                'method': 'GET',
                'description': 'Get historical metrics for a specific pool',
                'auth_required': True,
                'query_params': [
                    {'name': 'days', 'type': 'integer', 'description': 'Number of days of historical data', 'default': 7}
                ]
            },
            {
                'path': '/api/predictions/top',
                'method': 'GET',
                'description': 'Get top pool predictions',
                'auth_required': True,
                'query_params': [
                    {'name': 'category', 'type': 'string', 'description': 'Sort category (apr, risk, performance)', 'default': 'apr'},
                    {'name': 'limit', 'type': 'integer', 'description': 'Max number of predictions to return', 'default': 10},
                    {'name': 'ascending', 'type': 'boolean', 'description': 'Sort direction', 'default': False}
                ]
            },
            {
                'path': '/api/pools/{pool_id}/predictions',
                'method': 'GET',
                'description': 'Get predictions for a specific pool',
                'auth_required': True,
                'query_params': [
                    {'name': 'days', 'type': 'integer', 'description': 'Number of days of prediction history', 'default': 30}
                ]
            }
        ],
        'authentication': {
            'type': 'API Key',
            'header': 'X-API-Key'
        }
    })

# Pool list endpoint
@app.route('/api/pools', methods=['GET'])
def get_pools():
    """
    Get list of all pools with basic metrics
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        dex = request.args.get('dex', None)
        
        # Get pools from data service
        data_service = get_data_service()
        if not data_service:
            return jsonify({'error': 'Data service unavailable'}), 503
        
        # Get raw pool data
        pools = data_service.get_all_pools()
        
        # Filter and transform
        result = []
        count = 0
        
        for pool in pools:
            if count >= limit:
                break
                
            # Apply DEX filter if specified
            if dex and pool.get('dex', '').lower() != dex.lower():
                continue
                
            # Transform to API response format
            result.append({
                'id': pool.get('id', ''),
                'name': pool.get('name', ''),
                'dex': pool.get('dex', ''),
                'token1': pool.get('token1_symbol', ''),
                'token2': pool.get('token2_symbol', ''),
                'tvl': pool.get('liquidity', 0),
                'volume_24h': pool.get('volume_24h', 0),
                'apr': pool.get('apr', 0),
                'fee': pool.get('fee', 0),
                'timestamp': pool.get('timestamp', datetime.now().isoformat())
            })
            count += 1
        
        # Return response
        return jsonify({
            'pools': result,
            'total': len(result),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting pools: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Pool details endpoint
@app.route('/api/pools/<pool_id>', methods=['GET'])
def get_pool_details(pool_id):
    """
    Get detailed data for a specific pool
    """
    try:
        # Get pool details from data service
        data_service = get_data_service()
        if not data_service:
            return jsonify({'error': 'Data service unavailable'}), 503
        
        # Get pool data
        pool = data_service.get_pool_by_id(pool_id)
        
        if not pool:
            return jsonify({'error': 'Pool not found'}), 404
        
        # Get token prices for additional context
        token_price_service = get_token_price_service()
        token1_price = 0
        token2_price = 0
        
        if token_price_service:
            token1_price = token_price_service.get_token_price(pool.get('token1_symbol', ''))
            token2_price = token_price_service.get_token_price(pool.get('token2_symbol', ''))
        
        # Transform to API response format with additional details
        result = {
            'id': pool.get('id', ''),
            'name': pool.get('name', ''),
            'dex': pool.get('dex', ''),
            'token1': {
                'symbol': pool.get('token1_symbol', ''),
                'address': pool.get('token1_address', ''),
                'price_usd': token1_price
            },
            'token2': {
                'symbol': pool.get('token2_symbol', ''),
                'address': pool.get('token2_address', ''),
                'price_usd': token2_price
            },
            'metrics': {
                'liquidity': pool.get('liquidity', 0),
                'volume_24h': pool.get('volume_24h', 0),
                'apr': pool.get('apr', 0),
                'fee': pool.get('fee', 0),
                'tvl_change_24h': pool.get('tvl_change_24h', 0),
                'apr_change_24h': pool.get('apr_change_24h', 0),
            },
            'timestamp': pool.get('timestamp', datetime.now().isoformat())
        }
        
        # Return response
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting pool details: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Pool metrics endpoint
@app.route('/api/pools/<pool_id>/metrics', methods=['GET'])
def get_pool_metrics(pool_id):
    """
    Get historical metrics for a specific pool
    """
    try:
        # Get query parameters
        days = request.args.get('days', 7, type=int)
        
        # Get pool metrics from database
        db = get_db_manager()
        if not db:
            return jsonify({'error': 'Database unavailable'}), 503
        
        # Get metrics
        metrics_df = db.get_pool_metrics(pool_id, days)
        
        if metrics_df.empty:
            return jsonify({'error': 'No metrics found for this pool'}), 404
        
        # Convert DataFrame to list of dictionaries for JSON serialization
        metrics_list = []
        for _, row in metrics_df.iterrows():
            metrics_list.append({
                'timestamp': row.get('timestamp').isoformat() if hasattr(row.get('timestamp'), 'isoformat') else row.get('timestamp'),
                'liquidity': float(row.get('liquidity')),
                'volume': float(row.get('volume')),
                'apr': float(row.get('apr'))
            })
        
        # Return response
        return jsonify({
            'pool_id': pool_id,
            'metrics': metrics_list,
            'days': days
        })
        
    except Exception as e:
        logger.error(f"Error getting pool metrics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Top predictions endpoint
@app.route('/api/predictions/top', methods=['GET'])
def get_top_predictions():
    """
    Get top pool predictions
    """
    try:
        # Get query parameters
        category = request.args.get('category', 'apr')
        limit = request.args.get('limit', 10, type=int)
        ascending = request.args.get('ascending', 'false').lower() == 'true'
        
        # Get predictions from database
        db = get_db_manager()
        if not db:
            return jsonify({'error': 'Database unavailable'}), 503
        
        # Get top predictions
        predictions_df = db.get_top_predictions(category, limit, ascending)
        
        if predictions_df.empty:
            return jsonify({'error': 'No predictions available'}), 404
        
        # Convert DataFrame to list of dictionaries for JSON serialization
        predictions_list = []
        for _, row in predictions_df.iterrows():
            predictions_list.append({
                'pool_id': row.get('pool_id'),
                'pool_name': row.get('pool_name'),
                'predicted_apr': float(row.get('predicted_apr')),
                'risk_score': float(row.get('risk_score')),
                'performance_class': row.get('performance_class'),
                'prediction_timestamp': row.get('prediction_timestamp').isoformat() if hasattr(row.get('prediction_timestamp'), 'isoformat') else row.get('prediction_timestamp')
            })
        
        # Return response
        return jsonify({
            'predictions': predictions_list,
            'category': category,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting top predictions: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Pool predictions endpoint
@app.route('/api/pools/<pool_id>/predictions', methods=['GET'])
def get_pool_predictions(pool_id):
    """
    Get predictions for a specific pool
    """
    try:
        # Get query parameters
        days = request.args.get('days', 30, type=int)
        
        # Get predictions from database
        db = get_db_manager()
        if not db:
            return jsonify({'error': 'Database unavailable'}), 503
        
        # Get predictions
        predictions_df = db.get_pool_predictions(pool_id, days)
        
        if predictions_df.empty:
            return jsonify({'error': 'No predictions found for this pool'}), 404
        
        # Convert DataFrame to list of dictionaries for JSON serialization
        predictions_list = []
        for _, row in predictions_df.iterrows():
            predictions_list.append({
                'predicted_apr': float(row.get('predicted_apr')),
                'risk_score': float(row.get('risk_score')),
                'performance_class': row.get('performance_class'),
                'prediction_timestamp': row.get('prediction_timestamp').isoformat() if hasattr(row.get('prediction_timestamp'), 'isoformat') else row.get('prediction_timestamp')
            })
        
        # Return response
        return jsonify({
            'pool_id': pool_id,
            'pool_name': predictions_df.iloc[0].get('pool_name') if not predictions_df.empty else '',
            'predictions': predictions_list
        })
        
    except Exception as e:
        logger.error(f"Error getting pool predictions: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Run the API server if executed directly
if __name__ == '__main__':
    # Set host to 0.0.0.0 to make it externally accessible
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5100)), debug=False)
