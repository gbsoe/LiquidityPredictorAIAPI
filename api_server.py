from flask import Flask, request, jsonify
import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
import json
from functools import wraps
import uuid
import random  # For sample data generation

# Import custom modules
try:
    from pool_retrieval_system import PoolRetriever, PoolData
    from advanced_filtering import AdvancedFilteringSystem
except ImportError:
    # For deployment without custom modules
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_server")

# Initialize Flask app
app = Flask(__name__)

# Configuration
DEBUG = True
API_VERSION = "v1"
DEFAULT_LIMIT = 100
RATE_LIMIT = {
    "free": 100,  # per hour
    "standard": 1000,  # per hour
    "enterprise": 10000  # per hour
}

# In-memory rate limiting (replace with Redis in production)
rate_limit_store = {}

# In-memory API keys for demo (use a proper database in production)
API_KEYS = {
    "test_key": {"tier": "free", "owner": "test_user"},
    "demo_key": {"tier": "standard", "owner": "demo_user"},
    "enterprise_key": {"tier": "enterprise", "owner": "enterprise_user"}
}

# Sample data loading
def load_sample_data():
    """Load sample data for demonstration"""
    # Check if we have sample data saved
    if os.path.exists("sample_pool_data.json"):
        try:
            with open("sample_pool_data.json", "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
    
    # If we can't load from file, return empty array
    return []

# API key authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({
                "status": "error",
                "error": "Missing API key",
                "code": "AUTHENTICATION_ERROR",
                "details": {"message": "API key is required in the X-API-Key header"}
            }), 401
            
        if api_key not in API_KEYS:
            return jsonify({
                "status": "error",
                "error": "Invalid API key",
                "code": "AUTHENTICATION_ERROR",
                "details": {"message": "The provided API key is invalid"}
            }), 401
            
        # Rate limiting
        tier = API_KEYS[api_key]["tier"]
        current_hour = int(time.time() / 3600)
        rate_key = f"{api_key}:{current_hour}"
        
        if rate_key not in rate_limit_store:
            rate_limit_store[rate_key] = 0
            
        rate_limit_store[rate_key] += 1
        
        if rate_limit_store[rate_key] > RATE_LIMIT[tier]:
            return jsonify({
                "status": "error",
                "error": "Rate limit exceeded",
                "code": "RATE_LIMIT_EXCEEDED",
                "details": {
                    "message": f"You have exceeded the rate limit for your tier ({tier})",
                    "limit": RATE_LIMIT[tier],
                    "reset_in_seconds": 3600 - (int(time.time()) % 3600)
                }
            }), 429
            
        return f(*args, **kwargs)
    return decorated_function

# Error handling
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "status": "error",
        "error": "Resource not found",
        "code": "RESOURCE_NOT_FOUND",
        "details": {"message": "The requested resource does not exist"}
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "status": "error",
        "error": "Internal server error",
        "code": "INTERNAL_ERROR",
        "details": {"message": "An unexpected error occurred"}
    }), 500

# Utility functions
def parse_query_params():
    """Parse and validate common query parameters"""
    try:
        params = {
            # Pagination
            "limit": min(int(request.args.get("limit", DEFAULT_LIMIT)), 1000),
            "offset": max(int(request.args.get("offset", 0)), 0),
            
            # Filtering
            "dex": request.args.get("dex"),
            "category": request.args.get("category"),
            "min_tvl": float(request.args.get("min_tvl", 0)),
            "max_tvl": float(request.args.get("max_tvl")) if request.args.get("max_tvl") else None,
            "min_apr": float(request.args.get("min_apr", 0)),
            "max_apr": float(request.args.get("max_apr")) if request.args.get("max_apr") else None,
            "min_volume": float(request.args.get("min_volume", 0)),
            "token": request.args.get("token"),
            
            # Sorting
            "sort_by": request.args.get("sort_by", "liquidity"),
            "sort_dir": request.args.get("sort_dir", "desc"),
            
            # Advanced filtering
            "min_prediction": float(request.args.get("min_prediction", 0)),
            "trend": request.args.get("trend"),
            "trend_field": request.args.get("trend_field", "apr"),
            "trend_days": int(request.args.get("trend_days", 7)),
            "trend_threshold": float(request.args.get("trend_threshold", 1))
        }
        
        return params
    except ValueError as e:
        raise ValueError(f"Invalid parameter format: {str(e)}")

def apply_filters(pools, params):
    """Apply filters to pool data based on query parameters"""
    filtered_pools = pools.copy()
    
    # Basic filters
    if params.get("dex"):
        filtered_pools = [p for p in filtered_pools if p.get("dex") == params["dex"]]
        
    if params.get("category"):
        filtered_pools = [p for p in filtered_pools if p.get("category") == params["category"]]
        
    if params.get("min_tvl") > 0:
        filtered_pools = [p for p in filtered_pools if p.get("liquidity", 0) >= params["min_tvl"]]
        
    if params.get("max_tvl"):
        filtered_pools = [p for p in filtered_pools if p.get("liquidity", 0) <= params["max_tvl"]]
        
    if params.get("min_apr") > 0:
        filtered_pools = [p for p in filtered_pools if p.get("apr", 0) >= params["min_apr"]]
        
    if params.get("max_apr"):
        filtered_pools = [p for p in filtered_pools if p.get("apr", 0) <= params["max_apr"]]
        
    if params.get("min_volume") > 0:
        filtered_pools = [p for p in filtered_pools if p.get("volume_24h", 0) >= params["min_volume"]]
        
    if params.get("token"):
        filtered_pools = [
            p for p in filtered_pools if 
            params["token"] == p.get("token1_symbol") or params["token"] == p.get("token2_symbol")
        ]
    
    # Advanced filters
    if params.get("min_prediction") > 0:
        filtered_pools = [p for p in filtered_pools if p.get("prediction_score", 0) >= params["min_prediction"]]
    
    if params.get("trend"):
        trend_field = f"{params['trend_field']}_change_{params['trend_days']}d"
        threshold = params["trend_threshold"]
        
        if params["trend"] == "increasing":
            filtered_pools = [p for p in filtered_pools if p.get(trend_field, 0) >= threshold]
        elif params["trend"] == "decreasing":
            filtered_pools = [p for p in filtered_pools if p.get(trend_field, 0) <= -threshold]
        elif params["trend"] == "stable":
            filtered_pools = [
                p for p in filtered_pools if 
                abs(p.get(trend_field, 0)) < threshold
            ]
    
    return filtered_pools

def sort_pools(pools, sort_by, sort_dir):
    """Sort pools based on specified criteria"""
    reverse = sort_dir.lower() != "asc"
    
    if sort_by == "apr":
        return sorted(pools, key=lambda x: x.get("apr", 0), reverse=reverse)
    elif sort_by == "volume_24h" or sort_by == "volume":
        return sorted(pools, key=lambda x: x.get("volume_24h", 0), reverse=reverse)
    elif sort_by == "prediction_score" or sort_by == "score":
        return sorted(pools, key=lambda x: x.get("prediction_score", 0), reverse=reverse)
    else:  # Default to liquidity/TVL
        return sorted(pools, key=lambda x: x.get("liquidity", 0), reverse=reverse)

def paginate_results(data, limit, offset):
    """Apply pagination to results"""
    return data[offset:offset + limit]

# Routes
@app.route("/", methods=["GET"])
def home():
    """API Home / Documentation redirect"""
    return jsonify({
        "status": "success",
        "message": "Solana Liquidity Pool Analysis API",
        "version": API_VERSION,
        "documentation_url": "/docs"
    })

@app.route("/docs", methods=["GET"])
def docs():
    """API Documentation"""
    return jsonify({
        "status": "success",
        "message": "API Documentation",
        "endpoints": [
            {"path": "/v1/pools", "method": "GET", "description": "Get all pools with filtering"},
            {"path": "/v1/pools/{pool_id}", "method": "GET", "description": "Get details for a specific pool"},
            {"path": "/v1/pools/{pool_id}/history", "method": "GET", "description": "Get historical data for a specific pool"},
            {"path": "/v1/dexes/{dex_name}", "method": "GET", "description": "Get statistics for a specific DEX"},
            {"path": "/v1/categories/{category_name}", "method": "GET", "description": "Get statistics for a specific pool category"},
            {"path": "/v1/tokens/{token_symbol}/pools", "method": "GET", "description": "Get all pools containing a specific token"},
            {"path": "/v1/tokens/{token_symbol}", "method": "GET", "description": "Get information about a specific token"},
            {"path": "/v1/tokens/{token_symbol}/price-history", "method": "GET", "description": "Get historical price data for a specific token"},
            {"path": "/v1/predictions", "method": "GET", "description": "Get ML-based predictions for pools"},
            {"path": "/v1/pools/{pool_id}/similar", "method": "GET", "description": "Find pools similar to a reference pool"},
            {"path": "/v1/market-overview", "method": "GET", "description": "Get aggregate market statistics"}
        ],
        "full_documentation": "https://docs.solanapoolanalytics.com"
    })

@app.route(f"/{API_VERSION}/pools", methods=["GET"])
@require_api_key
def get_pools():
    """Get all pools with optional filtering"""
    try:
        # Parse query parameters
        params = parse_query_params()
        
        # Load pool data
        pools = load_sample_data()
        
        # Apply filters
        filtered_pools = apply_filters(pools, params)
        
        # Sort results
        sorted_pools = sort_pools(filtered_pools, params["sort_by"], params["sort_dir"])
        
        # Apply pagination
        paginated_results = paginate_results(sorted_pools, params["limit"], params["offset"])
        
        # Return response
        return jsonify({
            "status": "success",
            "count": len(filtered_pools),
            "data": paginated_results
        })
    except ValueError as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "code": "INVALID_PARAMETER",
            "details": {"message": "One or more parameters are invalid"}
        }), 400
    except Exception as e:
        logger.error(f"Error in get_pools: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route(f"/{API_VERSION}/pools/<pool_id>", methods=["GET"])
@require_api_key
def get_pool_by_id(pool_id):
    """Get detailed information about a specific pool"""
    try:
        # Load pool data
        pools = load_sample_data()
        
        # Find the specified pool
        pool = next((p for p in pools if p.get("id") == pool_id), None)
        
        if not pool:
            return jsonify({
                "status": "error",
                "error": "Pool not found",
                "code": "RESOURCE_NOT_FOUND",
                "details": {"message": f"No pool found with ID: {pool_id}"}
            }), 404
        
        # Add extra details for single pool view
        pool["token1_price_usd"] = 103.45 if pool.get("token1_symbol") == "SOL" else random.uniform(0.1, 1000)
        pool["token2_price_usd"] = 1.0 if pool.get("token2_symbol") == "USDC" else random.uniform(0.1, 1000)
        pool["created_at"] = "2023-05-12T00:00:00Z"
        pool["last_updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Return response
        return jsonify({
            "status": "success",
            "data": pool
        })
    except Exception as e:
        logger.error(f"Error in get_pool_by_id: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route(f"/{API_VERSION}/pools/<pool_id>/history", methods=["GET"])
@require_api_key
def get_pool_history(pool_id):
    """Get historical data for a specific pool"""
    try:
        # Parse query parameters
        days = min(int(request.args.get("days", 30)), 365)
        interval = request.args.get("interval", "day")
        metrics = request.args.get("metrics", "all")
        
        # Validate parameters
        if interval not in ["hour", "day", "week"]:
            return jsonify({
                "status": "error",
                "error": "Invalid interval",
                "code": "INVALID_PARAMETER",
                "details": {"message": "Interval must be one of: hour, day, week"}
            }), 400
        
        # Load pool data
        pools = load_sample_data()
        
        # Find the specified pool
        pool = next((p for p in pools if p.get("id") == pool_id), None)
        
        if not pool:
            return jsonify({
                "status": "error",
                "error": "Pool not found",
                "code": "RESOURCE_NOT_FOUND",
                "details": {"message": f"No pool found with ID: {pool_id}"}
            }), 404
        
        # Generate historical data
        historical_data = []
        current_date = datetime.now()
        
        # Set interval in hours
        interval_hours = 1 if interval == "hour" else 24 if interval == "day" else 168
        
        # Calculate total intervals
        total_intervals = days * 24 // interval_hours
        
        base_liquidity = pool.get("liquidity", 10_000_000)
        base_volume = pool.get("volume_24h", 1_000_000)
        base_apr = pool.get("apr", 10)
        
        # Generate historical data points
        for i in range(total_intervals):
            timestamp = current_date - timedelta(hours=i * interval_hours)
            
            # Add some randomness to historical values
            random_factor = 0.05  # 5% variation
            liquidity = base_liquidity * (1 + random.uniform(-random_factor, random_factor))
            volume = base_volume * (1 + random.uniform(-random_factor, random_factor))
            apr = base_apr * (1 + random.uniform(-random_factor, random_factor))
            
            # Gradual trend for more realistic data
            trend_factor = i / total_intervals * 0.2  # up to 20% change over the whole period
            liquidity = liquidity * (1 - trend_factor)  # slightly decreasing
            volume = volume * (1 - trend_factor * 0.5)  # slightly decreasing but less than liquidity
            apr = apr * (1 + trend_factor * 0.3)  # slightly increasing
            
            # Create data point
            data_point = {
                "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "liquidity": liquidity,
                "volume": volume,
                "apr": apr,
                "token1_price_usd": 103.45 * (1 + random.uniform(-0.02, 0.02)) if pool.get("token1_symbol") == "SOL" else random.uniform(0.1, 1000),
                "token2_price_usd": 1.0 * (1 + random.uniform(-0.001, 0.001)) if pool.get("token2_symbol") == "USDC" else random.uniform(0.1, 1000)
            }
            
            historical_data.append(data_point)
        
        # Reverse to get chronological order
        historical_data.reverse()
        
        # Return response
        return jsonify({
            "status": "success",
            "pool_id": pool_id,
            "interval": interval,
            "data": historical_data
        })
    except ValueError as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "code": "INVALID_PARAMETER",
            "details": {"message": "One or more parameters are invalid"}
        }), 400
    except Exception as e:
        logger.error(f"Error in get_pool_history: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route(f"/{API_VERSION}/dexes/<dex_name>", methods=["GET"])
@require_api_key
def get_dex_statistics(dex_name):
    """Get statistics and aggregate data for a specific DEX"""
    try:
        # Load pool data
        pools = load_sample_data()
        
        # Filter pools for the specified DEX
        dex_pools = [p for p in pools if p.get("dex") == dex_name]
        
        if not dex_pools:
            return jsonify({
                "status": "error",
                "error": "DEX not found",
                "code": "RESOURCE_NOT_FOUND",
                "details": {"message": f"No pools found for DEX: {dex_name}"}
            }), 404
        
        # Calculate statistics
        total_liquidity = sum(p.get("liquidity", 0) for p in dex_pools)
        total_volume = sum(p.get("volume_24h", 0) for p in dex_pools)
        avg_apr = sum(p.get("apr", 0) for p in dex_pools) / len(dex_pools) if dex_pools else 0
        
        # Find highest APR pool
        highest_apr_pool = max(dex_pools, key=lambda x: x.get("apr", 0))
        
        # Get top pools by liquidity
        top_pools = sorted(dex_pools, key=lambda x: x.get("liquidity", 0), reverse=True)[:5]
        top_pools_simplified = [
            {"id": p.get("id"), "name": p.get("name"), "liquidity": p.get("liquidity")}
            for p in top_pools
        ]
        
        # Count pools by category
        categories = {}
        for pool in dex_pools:
            category = pool.get("category", "Other")
            categories[category] = categories.get(category, 0) + 1
        
        # Return response
        return jsonify({
            "status": "success",
            "data": {
                "name": dex_name,
                "pool_count": len(dex_pools),
                "total_liquidity": total_liquidity,
                "total_volume_24h": total_volume,
                "average_apr": avg_apr,
                "highest_apr_pool": {
                    "id": highest_apr_pool.get("id"),
                    "name": highest_apr_pool.get("name"),
                    "apr": highest_apr_pool.get("apr")
                },
                "top_pools_by_liquidity": top_pools_simplified,
                "categories": categories
            }
        })
    except Exception as e:
        logger.error(f"Error in get_dex_statistics: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route(f"/{API_VERSION}/categories/<category_name>", methods=["GET"])
@require_api_key
def get_category_statistics(category_name):
    """Get statistics and aggregate data for a specific pool category"""
    try:
        # Load pool data
        pools = load_sample_data()
        
        # Filter pools for the specified category
        category_pools = [p for p in pools if p.get("category") == category_name]
        
        if not category_pools:
            return jsonify({
                "status": "error",
                "error": "Category not found",
                "code": "RESOURCE_NOT_FOUND",
                "details": {"message": f"No pools found for category: {category_name}"}
            }), 404
        
        # Calculate statistics
        total_liquidity = sum(p.get("liquidity", 0) for p in category_pools)
        total_volume = sum(p.get("volume_24h", 0) for p in category_pools)
        avg_apr = sum(p.get("apr", 0) for p in category_pools) / len(category_pools) if category_pools else 0
        
        # Find highest APR pool
        highest_apr_pool = max(category_pools, key=lambda x: x.get("apr", 0))
        
        # Get top pools by liquidity
        top_pools = sorted(category_pools, key=lambda x: x.get("liquidity", 0), reverse=True)[:5]
        top_pools_simplified = [
            {"id": p.get("id"), "name": p.get("name"), "liquidity": p.get("liquidity")}
            for p in top_pools
        ]
        
        # Count pools by DEX
        dexes = {}
        for pool in category_pools:
            dex = pool.get("dex", "Other")
            dexes[dex] = dexes.get(dex, 0) + 1
        
        # Return response
        return jsonify({
            "status": "success",
            "data": {
                "name": category_name,
                "pool_count": len(category_pools),
                "total_liquidity": total_liquidity,
                "total_volume_24h": total_volume,
                "average_apr": avg_apr,
                "highest_apr_pool": {
                    "id": highest_apr_pool.get("id"),
                    "name": highest_apr_pool.get("name"),
                    "apr": highest_apr_pool.get("apr")
                },
                "top_pools_by_liquidity": top_pools_simplified,
                "dexes": dexes
            }
        })
    except Exception as e:
        logger.error(f"Error in get_category_statistics: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route(f"/{API_VERSION}/tokens/<token_symbol>/pools", methods=["GET"])
@require_api_key
def get_token_pools(token_symbol):
    """Get all pools containing a specific token"""
    try:
        # Parse query parameters
        params = parse_query_params()
        
        # Load pool data
        pools = load_sample_data()
        
        # Find pools containing the token
        token_pools = [
            p for p in pools if 
            token_symbol.lower() == p.get("token1_symbol", "").lower() or 
            token_symbol.lower() == p.get("token2_symbol", "").lower()
        ]
        
        if not token_pools:
            return jsonify({
                "status": "error",
                "error": "Token not found",
                "code": "RESOURCE_NOT_FOUND",
                "details": {"message": f"No pools found for token: {token_symbol}"}
            }), 404
        
        # Get token address from the first pool
        token_address = ""
        first_pool = token_pools[0]
        if token_symbol.lower() == first_pool.get("token1_symbol", "").lower():
            token_address = first_pool.get("token1_address", "")
        else:
            token_address = first_pool.get("token2_address", "")
        
        # Apply additional filters
        filtered_pools = apply_filters(token_pools, params)
        
        # Sort results
        sorted_pools = sort_pools(filtered_pools, params["sort_by"], params["sort_dir"])
        
        # Apply pagination
        paginated_results = paginate_results(sorted_pools, params["limit"], params["offset"])
        
        # Return response
        return jsonify({
            "status": "success",
            "token": token_symbol,
            "token_address": token_address,
            "count": len(filtered_pools),
            "data": paginated_results
        })
    except ValueError as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "code": "INVALID_PARAMETER",
            "details": {"message": "One or more parameters are invalid"}
        }), 400
    except Exception as e:
        logger.error(f"Error in get_token_pools: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route(f"/{API_VERSION}/tokens/<token_symbol>", methods=["GET"])
@require_api_key
def get_token_information(token_symbol):
    """Get information about a specific token"""
    try:
        # Load pool data
        pools = load_sample_data()
        
        # Find pools containing the token
        token_pools = [
            p for p in pools if 
            token_symbol.lower() == p.get("token1_symbol", "").lower() or 
            token_symbol.lower() == p.get("token2_symbol", "").lower()
        ]
        
        if not token_pools:
            return jsonify({
                "status": "error",
                "error": "Token not found",
                "code": "RESOURCE_NOT_FOUND",
                "details": {"message": f"No pools found for token: {token_symbol}"}
            }), 404
        
        # Get token information from the first pool
        token_address = ""
        first_pool = token_pools[0]
        if token_symbol.lower() == first_pool.get("token1_symbol", "").lower():
            token_address = first_pool.get("token1_address", "")
        else:
            token_address = first_pool.get("token2_address", "")
        
        # Calculate total liquidity in all pools with this token
        total_liquidity = sum(p.get("liquidity", 0) for p in token_pools)
        
        # Find highest APR pool
        highest_apr_pool = max(token_pools, key=lambda x: x.get("apr", 0))
        
        # Generate some sample market data
        if token_symbol.lower() == "sol":
            price = 103.45
            market_cap = 45_000_000_000
            volume_24h = 1_200_000_000
            price_change_24h = 3.2
            price_change_7d = 8.7
        elif token_symbol.lower() == "usdc" or token_symbol.lower() == "usdt":
            price = 1.0
            market_cap = 30_000_000_000
            volume_24h = 5_000_000_000
            price_change_24h = 0.01
            price_change_7d = -0.02
        elif token_symbol.lower() == "bonk":
            price = 0.00000234
            market_cap = 1_345_678_901
            volume_24h = 234_567_890
            price_change_24h = 5.67
            price_change_7d = 12.34
        else:
            # Generate random data for other tokens
            price = random.uniform(0.01, 100)
            market_cap = random.uniform(10_000_000, 1_000_000_000)
            volume_24h = random.uniform(1_000_000, 100_000_000)
            price_change_24h = random.uniform(-10, 10)
            price_change_7d = random.uniform(-20, 20)
        
        # Return response
        return jsonify({
            "status": "success",
            "data": {
                "symbol": token_symbol,
                "name": token_symbol,  # In a real implementation, we'd have actual token names
                "address": token_address,
                "decimals": 9 if token_symbol.lower() == "sol" else 6 if token_symbol.lower() == "usdc" else 5 if token_symbol.lower() == "bonk" else 8,
                "current_price_usd": price,
                "market_cap": market_cap,
                "volume_24h": volume_24h,
                "price_change_24h": price_change_24h,
                "price_change_7d": price_change_7d,
                "pool_count": len(token_pools),
                "total_liquidity": total_liquidity,
                "highest_apr_pool": {
                    "id": highest_apr_pool.get("id"),
                    "name": highest_apr_pool.get("name"),
                    "apr": highest_apr_pool.get("apr")
                }
            }
        })
    except Exception as e:
        logger.error(f"Error in get_token_information: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route(f"/{API_VERSION}/tokens/<token_symbol>/price-history", methods=["GET"])
@require_api_key
def get_token_price_history(token_symbol):
    """Get historical price data for a specific token"""
    try:
        # Parse query parameters
        days = min(int(request.args.get("days", 30)), 365)
        interval = request.args.get("interval", "day")
        
        # Validate parameters
        if interval not in ["hour", "day", "week"]:
            return jsonify({
                "status": "error",
                "error": "Invalid interval",
                "code": "INVALID_PARAMETER",
                "details": {"message": "Interval must be one of: hour, day, week"}
            }), 400
        
        # Load pool data
        pools = load_sample_data()
        
        # Find pools containing the token
        token_pools = [
            p for p in pools if 
            token_symbol.lower() == p.get("token1_symbol", "").lower() or 
            token_symbol.lower() == p.get("token2_symbol", "").lower()
        ]
        
        if not token_pools:
            return jsonify({
                "status": "error",
                "error": "Token not found",
                "code": "RESOURCE_NOT_FOUND",
                "details": {"message": f"No pools found for token: {token_symbol}"}
            }), 404
        
        # Set current price based on token
        if token_symbol.lower() == "sol":
            current_price = 103.45
        elif token_symbol.lower() == "usdc" or token_symbol.lower() == "usdt":
            current_price = 1.0
        elif token_symbol.lower() == "bonk":
            current_price = 0.00000234
        else:
            # Generate random price for other tokens
            current_price = random.uniform(0.01, 100)
        
        # Generate historical price data
        historical_data = []
        current_date = datetime.now()
        
        # Set interval in hours
        interval_hours = 1 if interval == "hour" else 24 if interval == "day" else 168
        
        # Calculate total intervals
        total_intervals = days * 24 // interval_hours
        
        # Generate historical data points
        for i in range(total_intervals):
            timestamp = current_date - timedelta(hours=i * interval_hours)
            
            # Add some randomness to historical values
            random_factor = 0.02  # 2% variation
            price = current_price * (1 + random.uniform(-random_factor, random_factor))
            
            # Gradual trend for more realistic data
            trend_factor = i / total_intervals * 0.1  # up to 10% change over the whole period
            if token_symbol.lower() == "bonk":
                # Meme coins tend to be more volatile with upward trend
                vol_multiplier = 3
                trend_direction = -1  # Going back in time, price was lower
            elif token_symbol.lower() == "usdc" or token_symbol.lower() == "usdt":
                # Stablecoins have minimal volatility
                vol_multiplier = 0.1
                trend_direction = 0  # No trend for stablecoins
            else:
                vol_multiplier = 1
                trend_direction = -0.5  # Slight downward trend going back in time
            
            price = price * (1 + trend_direction * trend_factor * vol_multiplier)
            
            # Create data point
            volume = sum(p.get("volume_24h", 0) for p in token_pools) * (1 + random.uniform(-0.1, 0.1))
            market_cap = price * (10_000_000_000 if token_symbol.lower() == "sol" else 
                                30_000_000_000 if token_symbol.lower() in ["usdc", "usdt"] else
                                random.uniform(100_000_000, 5_000_000_000))
            
            data_point = {
                "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "price_usd": price,
                "volume": volume,
                "market_cap": market_cap
            }
            
            historical_data.append(data_point)
        
        # Reverse to get chronological order
        historical_data.reverse()
        
        # Return response
        return jsonify({
            "status": "success",
            "token": token_symbol,
            "interval": interval,
            "data": historical_data
        })
    except ValueError as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "code": "INVALID_PARAMETER",
            "details": {"message": "One or more parameters are invalid"}
        }), 400
    except Exception as e:
        logger.error(f"Error in get_token_price_history: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route(f"/{API_VERSION}/predictions", methods=["GET"])
@require_api_key
def get_predictions():
    """Get ML-based predictions for pools"""
    try:
        # Parse query parameters
        min_score = float(request.args.get("min_score", 0))
        category = request.args.get("category")
        dex = request.args.get("dex")
        min_tvl = float(request.args.get("min_tvl", 0))
        limit = min(int(request.args.get("limit", 20)), 100)
        offset = max(int(request.args.get("offset", 0)), 0)
        sort_by = request.args.get("sort_by", "score")
        
        # Load pool data
        pools = load_sample_data()
        
        # Apply filters
        filtered_pools = pools.copy()
        
        if min_score > 0:
            filtered_pools = [p for p in filtered_pools if p.get("prediction_score", 0) >= min_score]
            
        if category:
            filtered_pools = [p for p in filtered_pools if p.get("category") == category]
            
        if dex:
            filtered_pools = [p for p in filtered_pools if p.get("dex") == dex]
            
        if min_tvl > 0:
            filtered_pools = [p for p in filtered_pools if p.get("liquidity", 0) >= min_tvl]
        
        # Sort results
        if sort_by == "potential_apr":
            # Sort by a combination of current APR and prediction score
            sorted_pools = sorted(
                filtered_pools, 
                key=lambda x: (x.get("apr", 0) * (x.get("prediction_score", 0) / 100)),
                reverse=True
            )
        else:  # Default to prediction score
            sorted_pools = sorted(
                filtered_pools,
                key=lambda x: x.get("prediction_score", 0),
                reverse=True
            )
        
        # Apply pagination
        paginated_results = sorted_pools[offset:offset + limit]
        
        # Enhance prediction data
        prediction_results = []
        
        for pool in paginated_results:
            # Generate predicted ranges based on current values and prediction score
            current_apr = pool.get("apr", 10)
            prediction_score = pool.get("prediction_score", 50)
            
            # Higher prediction scores have narrower ranges (more confident)
            confidence_factor = (100 - prediction_score) / 100 * 0.5 + 0.1  # Range from 0.1 to 0.6
            
            # Create APR prediction range
            apr_low = current_apr * (1 - confidence_factor * 0.5)  # Less downside than upside
            apr_mid = current_apr * (1 + (prediction_score / 100) * 0.2)  # Higher score = higher expected APR
            apr_high = current_apr * (1 + confidence_factor * 1.5)  # More upside potential
            
            # Predicted TVL change correlates with prediction score
            tvl_change = (prediction_score - 50) / 50 * 15  # Range from -15% to +15%
            
            # Generate key factors
            key_factors = []
            
            apr_change_7d = pool.get("apr_change_7d", 0)
            tvl_change_7d = pool.get("tvl_change_7d", 0)
            category = pool.get("category", "")
            volume_to_tvl = pool.get("volume_24h", 0) / max(pool.get("liquidity", 1), 1)
            
            if apr_change_7d > 1:
                key_factors.append("Strong positive APR trend")
            elif apr_change_7d < -1:
                key_factors.append("Recent APR decline")
                
            if tvl_change_7d > 5:
                key_factors.append("Increasing liquidity")
            elif tvl_change_7d < -5:
                key_factors.append("Decreasing liquidity")
                
            if volume_to_tvl > 0.1:
                key_factors.append("High trading volume relative to TVL")
            elif volume_to_tvl < 0.03:
                key_factors.append("Low trading activity")
                
            if category == "Meme":
                key_factors.append("Meme coin volatility factor")
            elif category == "Stablecoin":
                key_factors.append("Stablecoin stability factor")
            
            # Create prediction result
            prediction_result = {
                "pool_id": pool.get("id"),
                "name": pool.get("name"),
                "dex": pool.get("dex"),
                "category": pool.get("category"),
                "current_tvl": pool.get("liquidity"),
                "current_apr": current_apr,
                "prediction_score": prediction_score,
                "predicted_apr_range": {
                    "low": apr_low,
                    "mid": apr_mid,
                    "high": apr_high
                },
                "predicted_tvl_change": tvl_change,
                "confidence_interval": confidence_factor * 10,  # Scale to a more readable number
                "key_factors": key_factors,
                "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            prediction_results.append(prediction_result)
        
        # Return response
        return jsonify({
            "status": "success",
            "count": len(filtered_pools),
            "data": prediction_results
        })
    except ValueError as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "code": "INVALID_PARAMETER",
            "details": {"message": "One or more parameters are invalid"}
        }), 400
    except Exception as e:
        logger.error(f"Error in get_predictions: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route(f"/{API_VERSION}/pools/<pool_id>/similar", methods=["GET"])
@require_api_key
def find_similar_pools(pool_id):
    """Find pools that are similar to a reference pool"""
    try:
        # Parse query parameters
        limit = min(int(request.args.get("limit", 5)), 20)
        metrics_str = request.args.get("metrics", "tvl,apr,volume")
        min_similarity = float(request.args.get("min_similarity", 50))
        
        # Parse metrics
        metrics = metrics_str.split(",")
        valid_metrics = ["tvl", "liquidity", "apr", "volume", "volume_24h", "volatility"]
        
        # Map some alternative names
        metric_mapping = {
            "tvl": "liquidity",
            "volume": "volume_24h"
        }
        
        # Normalize metrics
        normalized_metrics = []
        for metric in metrics:
            metric = metric.strip().lower()
            if metric in metric_mapping:
                metric = metric_mapping[metric]
            if metric in valid_metrics:
                normalized_metrics.append(metric)
        
        # Load pool data
        pools = load_sample_data()
        
        # Find the reference pool
        reference_pool = next((p for p in pools if p.get("id") == pool_id), None)
        
        if not reference_pool:
            return jsonify({
                "status": "error",
                "error": "Pool not found",
                "code": "RESOURCE_NOT_FOUND",
                "details": {"message": f"No pool found with ID: {pool_id}"}
            }), 404
        
        # Calculate similarity
        similarities = []
        
        for pool in pools:
            if pool.get("id") == pool_id:
                continue  # Skip the reference pool
                
            # Calculate Euclidean distance (lower = more similar)
            distance = 0
            
            for metric in normalized_metrics:
                ref_val = reference_pool.get(metric, 0)
                pool_val = pool.get(metric, 0)
                
                # Skip if either value is missing
                if ref_val is None or pool_val is None:
                    continue
                    
                # Normalize by the typical range of the metric
                if metric in ["liquidity", "volume_24h"]:
                    # Use logarithmic scale for monetary values
                    norm_distance = (math.log10(max(pool_val, 1)) - math.log10(max(ref_val, 1))) ** 2
                elif metric == "apr":
                    # APR is already a percentage
                    norm_distance = ((pool_val - ref_val) / 30) ** 2  # Assuming 0-30% APR range
                elif metric == "volatility":
                    # Volatility is usually a small decimal
                    norm_distance = ((pool_val - ref_val) / 0.3) ** 2  # Assuming 0-30% volatility
                else:
                    # Default normalization
                    norm_distance = ((pool_val - ref_val) / max(ref_val, 1)) ** 2
                    
                distance += norm_distance
            
            # Additional bonus for same category and/or DEX
            if pool.get("category") == reference_pool.get("category"):
                distance *= 0.8  # 20% bonus for same category
                
            if pool.get("dex") == reference_pool.get("dex"):
                distance *= 0.9  # 10% bonus for same DEX
                
            # Get final similarity score (0-100, higher is more similar)
            similarity = 100 * (1 / (1 + math.sqrt(distance)))
            
            # Only include pools meeting the minimum similarity threshold
            if similarity >= min_similarity:
                similarities.append((pool, similarity))
        
        # Sort by similarity (highest first) and take top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:limit]
        
        # Prepare response data
        similar_pools = []
        
        for pool, score in top_similar:
            similar_pool = {
                "id": pool.get("id"),
                "name": pool.get("name"),
                "dex": pool.get("dex"),
                "similarity_score": round(score, 1),
                "liquidity": pool.get("liquidity"),
                "volume_24h": pool.get("volume_24h"),
                "apr": pool.get("apr"),
                "prediction_score": pool.get("prediction_score")
            }
            similar_pools.append(similar_pool)
        
        # Return response
        return jsonify({
            "status": "success",
            "reference_pool": {
                "id": reference_pool.get("id"),
                "name": reference_pool.get("name"),
                "dex": reference_pool.get("dex")
            },
            "data": similar_pools
        })
    except ValueError as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "code": "INVALID_PARAMETER",
            "details": {"message": "One or more parameters are invalid"}
        }), 400
    except Exception as e:
        logger.error(f"Error in find_similar_pools: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR",
            "details": {"message": str(e)}
        }), 500

@app.route(f"/{API_VERSION}/market-overview", methods=["GET"])
@require_api_key
def get_market_overview():
    """Get aggregate market statistics"""
    try:
        # Load pool data
        pools = load_sample_data()
        
        if not pools:
            return jsonify({
                "status": "error",
                "error": "No data available",
                "code": "RESOURCE_NOT_FOUND",
                "details": {"message": "No pool data is currently available"}
            }), 404
        
        # Calculate overall statistics
        total_pools = len(pools)
        total_liquidity = sum(p.get("liquidity", 0) for p in pools)
        total_volume = sum(p.get("volume_24h", 0) for p in pools)
        avg_apr = sum(p.get("apr", 0) for p in pools) / total_pools if total_pools else 0
        
        # Calculate DEX breakdown
        dex_breakdown = {}
        for pool in pools:
            dex = pool.get("dex", "Other")
            
            if dex not in dex_breakdown:
                dex_breakdown[dex] = {
                    "pool_count": 0,
                    "total_liquidity": 0,
                    "total_volume_24h": 0,
                    "apr_sum": 0
                }
                
            dex_breakdown[dex]["pool_count"] += 1
            dex_breakdown[dex]["total_liquidity"] += pool.get("liquidity", 0)
            dex_breakdown[dex]["total_volume_24h"] += pool.get("volume_24h", 0)
            dex_breakdown[dex]["apr_sum"] += pool.get("apr", 0)
        
        # Calculate average APR for each DEX
        for dex, stats in dex_breakdown.items():
            stats["average_apr"] = stats["apr_sum"] / stats["pool_count"] if stats["pool_count"] > 0 else 0
            del stats["apr_sum"]  # Remove intermediate calculation
        
        # Calculate category breakdown
        category_breakdown = {}
        for pool in pools:
            category = pool.get("category", "Other")
            
            if category not in category_breakdown:
                category_breakdown[category] = {
                    "pool_count": 0,
                    "total_liquidity": 0,
                    "total_volume_24h": 0,
                    "apr_sum": 0
                }
                
            category_breakdown[category]["pool_count"] += 1
            category_breakdown[category]["total_liquidity"] += pool.get("liquidity", 0)
            category_breakdown[category]["total_volume_24h"] += pool.get("volume_24h", 0)
            category_breakdown[category]["apr_sum"] += pool.get("apr", 0)
        
        # Calculate average APR for each category
        for category, stats in category_breakdown.items():
            stats["average_apr"] = stats["apr_sum"] / stats["pool_count"] if stats["pool_count"] > 0 else 0
            del stats["apr_sum"]  # Remove intermediate calculation
        
        # Calculate trends (using random values for demonstration)
        # In a real implementation, these would be calculated from historical data
        trends = {
            "overall_tvl_change_24h": random.uniform(-3, 5),
            "overall_volume_change_24h": random.uniform(-5, 10),
            "overall_apr_change_7d": random.uniform(-1, 2),
            "highest_growth_category": "Meme",
            "highest_growth_dex": "Jupiter"
        }
        
        # Return response
        return jsonify({
            "status": "success",
            "data": {
                "total_pools": total_pools,
                "total_liquidity": total_liquidity,
                "total_volume_24h": total_volume,
                "average_apr": avg_apr,
                "dex_breakdown": dex_breakdown,
                "category_breakdown": category_breakdown,
                "trends": trends
            }
        })
    except Exception as e:
        logger.error(f"Error in get_market_overview: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500


# Run the app
if __name__ == "__main__":
    # Only missing imports for sample data generation
    import math
    
    app.run(host="0.0.0.0", port=3000, debug=DEBUG)