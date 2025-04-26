import streamlit as st
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure page
st.set_page_config(
    page_title="API Documentation - SolPool Insight",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
.api-header {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 12px;
}
.api-subtitle {
    color: #6b7280;
    font-size: 18px;
    margin-bottom: 20px;
}
.endpoint-container {
    padding: 20px;
    border-radius: 10px;
    background-color: #f8f9fa;
    border-left: 5px solid #3b82f6;
    margin-bottom: 20px;
}
.endpoint-header {
    font-weight: 600;
    color: #3b82f6;
    margin-bottom: 12px;
}
.endpoint-method {
    font-family: monospace;
    background-color: #dbeafe;
    padding: 4px 8px;
    border-radius: 4px;
    color: #1e40af;
    font-weight: 600;
    margin-right: 8px;
}
.endpoint-path {
    font-family: monospace;
    font-weight: 500;
}
pre {
    background-color: #1e293b;
    color: #e2e8f0;
    padding: 15px;
    border-radius: 8px;
    overflow-x: auto;
}
.parameter-table {
    width: 100%;
    border-collapse: collapse;
}
.parameter-table th, .parameter-table td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}
.parameter-table th {
    background-color: #f3f4f6;
    font-weight: 600;
}
@media (prefers-color-scheme: dark) {
    .endpoint-container {
        background-color: #1e1e1e;
        border-left: 5px solid #3b82f6;
    }
    .endpoint-method {
        background-color: #172554;
        color: #93c5fd;
    }
    .parameter-table th {
        background-color: #1e293b;
    }
    .parameter-table th, .parameter-table td {
        border-bottom: 1px solid #374151;
    }
}
</style>
""", unsafe_allow_html=True)

# Add FiLot logo to sidebar
st.sidebar.image("static/filot_logo_new.png", width=130)
st.sidebar.markdown("### FiLot Analytics")
st.sidebar.markdown("---")

# Add API Tier selection
st.sidebar.subheader("API Access Tiers")
st.sidebar.markdown("""
- **Free tier**: 100 requests per hour
- **Standard tier**: 1,000 requests per hour
- **Enterprise tier**: 10,000 requests per hour
- **Mobile tier**: 500 requests per hour
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Need an API key? [Register](https://filot.io/register) or contact us at api@filot.io")

# Main content
st.markdown('<div class="api-header">API Documentation</div>', unsafe_allow_html=True)
st.markdown('<div class="api-subtitle">Complete reference for the SolPool Insight RESTful API</div>', unsafe_allow_html=True)

st.markdown("""
Our API provides programmatic access to all the data and insights available in SolPool Insight. 
You can use it to integrate Solana liquidity pool data, analytics, and AI-powered predictions into your own applications, 
dashboards, or trading systems.
""")

# Overview tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Authentication", "Response Format"])

with tab1:
    st.subheader("Base URL")
    st.code("https://filotanalytics.replit.app/API", language="text")
    
    st.subheader("Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - Real-time liquidity pool data
        - Comprehensive pool filtering
        - Historical performance metrics
        - Token-specific analytics
        """)
    
    with col2:
        st.markdown("""
        - ML-based performance predictions
        - Risk assessment metrics
        - Market-wide statistics
        - DEX-specific analytics
        """)

with tab2:
    st.subheader("API Key Authentication")
    st.markdown("""
    All API requests require authentication using an API key. 
    Your API key should be included in the request headers as follows:
    """)
    
    st.code('X-API-Key: your_api_key', language="text")
    
    st.markdown("""
    Keep your API key secure and do not share it publicly. 
    If you believe your key has been compromised, please contact support to have it rotated.
    """)
    
    st.subheader("Rate Limiting")
    st.markdown("""
    Rate limits vary by tier and are applied on a per-hour basis. 
    Rate limit headers are included in all API responses:
    """)
    
    st.code("""
X-Rate-Limit-Limit: 1000
X-Rate-Limit-Remaining: 985
X-Rate-Limit-Reset: 1619231400
    """, language="text")

with tab3:
    st.subheader("Response Format")
    st.markdown("""
    All API responses are returned in JSON format with a consistent structure:
    """)
    
    st.code("""
{
  "status": "success",
  "data": {
    // Response data here
  }
}
    """, language="json")
    
    st.markdown("""
    ### Error Responses
    
    Error responses follow a similar structure:
    """)
    
    st.code("""
{
  "status": "error",
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {
    // Additional error details here
  }
}
    """, language="json")
    
    st.markdown("""
    Common error codes include:
    - `AUTHENTICATION_ERROR`: Missing or invalid API key
    - `RATE_LIMIT_EXCEEDED`: You've exceeded your rate limit
    - `RESOURCE_NOT_FOUND`: The requested resource doesn't exist
    - `INVALID_PARAMETER`: One or more parameters are invalid
    - `INTERNAL_ERROR`: An unexpected server error occurred
    """)

# Endpoints section
st.markdown("## API Endpoints")

# Add navigation with filter
st.markdown("### Filter Endpoints")
endpoint_filter = st.text_input("Search endpoints", placeholder="Type to filter...")

# Define all endpoints
endpoints = [
    {
        "method": "GET",
        "path": "/pools",
        "description": "Get all pools with optional filtering",
        "parameters": [
            {"name": "dex", "type": "string", "description": "Filter by DEX name", "default": "null", "example": "Raydium"},
            {"name": "category", "type": "string", "description": "Filter by pool category", "default": "null", "example": "Meme"},
            {"name": "min_tvl", "type": "number", "description": "Minimum TVL threshold", "default": "0", "example": "1000000"},
            {"name": "max_tvl", "type": "number", "description": "Maximum TVL threshold", "default": "null", "example": "50000000"},
            {"name": "min_apr", "type": "number", "description": "Minimum APR threshold (percentage)", "default": "0", "example": "10"},
            {"name": "max_apr", "type": "number", "description": "Maximum APR threshold (percentage)", "default": "null", "example": "50"},
            {"name": "min_volume", "type": "number", "description": "Minimum 24h volume", "default": "0", "example": "100000"},
            {"name": "token", "type": "string", "description": "Filter pools containing this token", "default": "null", "example": "SOL"},
            {"name": "limit", "type": "integer", "description": "Maximum number of results", "default": "100", "example": "50"},
            {"name": "offset", "type": "integer", "description": "Number of results to skip (for pagination)", "default": "0", "example": "100"},
            {"name": "sort_by", "type": "string", "description": "Field to sort by", "default": "liquidity", "example": "apr"},
            {"name": "sort_dir", "type": "string", "description": "Sort direction ('asc' or 'desc')", "default": "desc", "example": "asc"},
            {"name": "min_prediction", "type": "number", "description": "Minimum prediction score (0-100)", "default": "0", "example": "80"},
            {"name": "trend", "type": "string", "description": "Filter by trend direction ('increasing', 'decreasing', 'stable')", "default": "null", "example": "increasing"},
            {"name": "trend_field", "type": "string", "description": "Field to apply trend filter to ('apr', 'tvl')", "default": "apr", "example": "tvl"},
            {"name": "trend_days", "type": "integer", "description": "Number of days for trend calculation", "default": "7", "example": "30"},
            {"name": "trend_threshold", "type": "number", "description": "Minimum change percentage for trend filtering", "default": "1", "example": "5"}
        ],
        "example_request": """curl -X GET "https://filotanalytics.replit.app/API/pools?dex=Raydium&min_tvl=1000000&min_apr=10" \\
  -H "X-API-Key: your_api_key\"""",
        "example_response": """{
  "status": "success",
  "count": 25,
  "data": [
    {
      "id": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
      "name": "SOL/USDC",
      "dex": "Raydium",
      "category": "Major",
      "token1_symbol": "SOL",
      "token2_symbol": "USDC",
      "token1_address": "So11111111111111111111111111111111111111112",
      "token2_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
      "liquidity": 24532890.45,
      "volume_24h": 8763021.32,
      "apr": 12.87,
      "volatility": 0.05,
      "fee": 0.0025,
      "version": "v4",
      "apr_change_24h": 0.42,
      "apr_change_7d": 1.2,
      "apr_change_30d": -2.1,
      "tvl_change_24h": 1.1,
      "tvl_change_7d": 3.5,
      "tvl_change_30d": -2.1,
      "prediction_score": 85
    },
    // More pools...
  ]
}"""
    },
    {
        "method": "GET",
        "path": "/pools/{pool_id}",
        "description": "Get details for a specific pool",
        "parameters": [
            {"name": "pool_id", "type": "string", "description": "The unique identifier of the pool", "example": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2"}
        ],
        "example_request": """curl -X GET "https://filotanalytics.replit.app/API/pools/58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2" \\
  -H "X-API-Key: your_api_key\"""",
        "example_response": """{
  "status": "success",
  "data": {
    "id": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
    "name": "SOL/USDC",
    "dex": "Raydium",
    "category": "Major",
    "token1_symbol": "SOL",
    "token2_symbol": "USDC",
    "token1_address": "So11111111111111111111111111111111111111112",
    "token2_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "liquidity": 24532890.45,
    "volume_24h": 8763021.32,
    "apr": 12.87,
    "volatility": 0.05,
    "fee": 0.0025,
    "version": "v4",
    "apr_change_24h": 0.42,
    "apr_change_7d": 1.2,
    "apr_change_30d": -2.1,
    "tvl_change_24h": 1.1,
    "tvl_change_7d": 3.5,
    "tvl_change_30d": -2.1,
    "prediction_score": 85,
    "token1_price_usd": 103.45,
    "token2_price_usd": 1.0,
    "created_at": "2023-05-12T00:00:00Z",
    "last_updated": "2025-04-24T12:34:56Z"
  }
}"""
    },
    {
        "method": "GET",
        "path": "/pools/{pool_id}/history",
        "description": "Get historical data for a specific pool",
        "parameters": [
            {"name": "pool_id", "type": "string", "description": "The unique identifier of the pool", "example": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2"},
            {"name": "days", "type": "integer", "description": "Number of days of history to retrieve", "default": "30", "example": "60"},
            {"name": "interval", "type": "string", "description": "Time interval ('hour', 'day', 'week')", "default": "day", "example": "hour"},
            {"name": "metrics", "type": "string", "description": "Comma-separated list of metrics to include", "default": "all", "example": "apr,tvl"}
        ],
        "example_request": """curl -X GET "https://filotanalytics.replit.app/v1/pools/58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2/history?days=60&interval=day" \\
  -H "X-API-Key: your_api_key\"""",
        "example_response": """{
  "status": "success",
  "pool_id": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
  "interval": "day",
  "data": [
    {
      "timestamp": "2025-04-24T00:00:00Z",
      "liquidity": 24532890.45,
      "volume": 8763021.32,
      "apr": 12.87,
      "token1_price_usd": 103.45,
      "token2_price_usd": 1.0
    },
    {
      "timestamp": "2025-04-23T00:00:00Z",
      "liquidity": 24287654.32,
      "volume": 9123456.78,
      "apr": 12.45,
      "token1_price_usd": 101.78,
      "token2_price_usd": 1.0
    },
    // More historical data points...
  ]
}"""
    },
    {
        "method": "GET",
        "path": "/dexes/{dex_name}",
        "description": "Get statistics for a specific DEX",
        "parameters": [
            {"name": "dex_name", "type": "string", "description": "The name of the DEX", "example": "Raydium"}
        ],
        "example_request": """curl -X GET "https://filotanalytics.replit.app/v1/dexes/Raydium" \\
  -H "X-API-Key: your_api_key\"""",
        "example_response": """{
  "status": "success",
  "data": {
    "name": "Raydium",
    "pool_count": 245,
    "total_liquidity": 1234567890.12,
    "total_volume_24h": 345678901.23,
    "average_apr": 14.5,
    "highest_apr_pool": {
      "id": "6UmmUiYoBjSrhakAobJw8BvkmJtDVxaeBtbt7rxWo1mg",
      "name": "RAY/USDC",
      "apr": 18.76
    },
    "top_pools_by_liquidity": [
      {
        "id": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
        "name": "SOL/USDC",
        "liquidity": 24532890.45
      },
      // More pools...
    ],
    "categories": {
      "Major": 45,
      "Meme": 68,
      "DeFi": 52,
      "Gaming": 37,
      "Stablecoin": 43
    }
  }
}"""
    },
    {
        "method": "GET",
        "path": "/categories/{category_name}",
        "description": "Get statistics for a specific pool category",
        "parameters": [
            {"name": "category_name", "type": "string", "description": "The name of the category", "example": "Meme"}
        ],
        "example_request": """curl -X GET "https://filotanalytics.replit.app/v1/categories/Meme" \\
  -H "X-API-Key: your_api_key\"""",
        "example_response": """{
  "status": "success",
  "data": {
    "name": "Meme",
    "pool_count": 68,
    "total_liquidity": 234567890.12,
    "total_volume_24h": 45678901.23,
    "average_apr": 24.5,
    "highest_apr_pool": {
      "id": "P0pCaT5Ec0iNR3P0mEk0iN51T0kENpuPpY",
      "name": "POPCAT/USDC",
      "apr": 38.90
    },
    "top_pools_by_liquidity": [
      {
        "id": "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K",
        "name": "BONK/USDC",
        "liquidity": 5432167.89
      },
      // More pools...
    ],
    "dexes": {
      "Raydium": 35,
      "Orca": 15,
      "Jupiter": 12,
      "Meteora": 6
    }
  }
}"""
    },
    {
        "method": "GET",
        "path": "/tokens/{token_symbol}/pools",
        "description": "Get all pools containing a specific token",
        "parameters": [
            {"name": "token_symbol", "type": "string", "description": "The token symbol", "example": "BONK"}
        ],
        "example_request": """curl -X GET "https://filotanalytics.replit.app/v1/tokens/BONK/pools?min_liquidity=1000000" \\
  -H "X-API-Key: your_api_key\"""",
        "example_response": """{
  "status": "success",
  "token": "BONK",
  "token_address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
  "count": 12,
  "data": [
    {
      "id": "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K",
      "name": "BONK/USDC",
      "dex": "Meteora",
      "category": "Meme",
      "liquidity": 5432167.89,
      "volume_24h": 1987654.32,
      "apr": 25.67,
      // Other pool details...
    },
    // More pools...
  ]
}"""
    },
    {
        "method": "GET",
        "path": "/tokens/{token_symbol}",
        "description": "Get information about a specific token",
        "parameters": [
            {"name": "token_symbol", "type": "string", "description": "The token symbol", "example": "BONK"}
        ],
        "example_request": """curl -X GET "https://filotanalytics.replit.app/v1/tokens/BONK" \\
  -H "X-API-Key: your_api_key\"""",
        "example_response": """{
  "status": "success",
  "data": {
    "symbol": "BONK",
    "name": "Bonk",
    "address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "decimals": 5,
    "current_price_usd": 0.00000234,
    "market_cap": 1345678901,
    "volume_24h": 234567890,
    "price_change_24h": 5.67,
    "price_change_7d": 12.34,
    "pool_count": 12,
    "total_liquidity": 12345678.90,
    "highest_apr_pool": {
      "id": "B0nkD2EW5B0nK1nG51mECoiNSolANaPooL5Us3",
      "name": "BONK/SOL",
      "apr": 28.90
    }
  }
}"""
    },
    {
        "method": "GET",
        "path": "/tokens/{token_symbol}/price-history",
        "description": "Get historical price data for a specific token",
        "parameters": [
            {"name": "token_symbol", "type": "string", "description": "The token symbol", "example": "BONK"},
            {"name": "days", "type": "integer", "description": "Number of days of history to retrieve", "default": "30", "example": "60"},
            {"name": "interval", "type": "string", "description": "Time interval ('hour', 'day', 'week')", "default": "day", "example": "hour"}
        ],
        "example_request": """curl -X GET "https://filotanalytics.replit.app/v1/tokens/BONK/price-history?days=60&interval=day" \\
  -H "X-API-Key: your_api_key\"""",
        "example_response": """{
  "status": "success",
  "token": "BONK",
  "interval": "day",
  "data": [
    {
      "timestamp": "2025-04-24T00:00:00Z",
      "price_usd": 0.00000234,
      "volume": 234567890,
      "market_cap": 1345678901
    },
    {
      "timestamp": "2025-04-23T00:00:00Z",
      "price_usd": 0.00000225,
      "volume": 212345678,
      "market_cap": 1298765432
    },
    // More price data points...
  ]
}"""
    },
    {
        "method": "GET",
        "path": "/predictions",
        "description": "Get ML-based predictions for pools",
        "parameters": [
            {"name": "min_score", "type": "number", "description": "Minimum prediction score (0-100)", "default": "0", "example": "80"},
            {"name": "category", "type": "string", "description": "Filter by pool category", "default": "null", "example": "Meme"},
            {"name": "dex", "type": "string", "description": "Filter by DEX name", "default": "null", "example": "Raydium"},
            {"name": "min_tvl", "type": "number", "description": "Minimum TVL threshold", "default": "0", "example": "1000000"},
            {"name": "limit", "type": "integer", "description": "Maximum number of results", "default": "20", "example": "50"},
            {"name": "offset", "type": "integer", "description": "Number of results to skip (for pagination)", "default": "0", "example": "20"},
            {"name": "sort_by", "type": "string", "description": "Field to sort predictions by", "default": "score", "example": "potential_apr"}
        ],
        "example_request": """curl -X GET "https://filotanalytics.replit.app/v1/predictions?min_score=80&category=Meme" \\
  -H "X-API-Key: your_api_key\"""",
        "example_response": """{
  "status": "success",
  "count": 15,
  "data": [
    {
      "pool_id": "B0nkD2EW5B0nK1nG51mECoiNSolANaPooL5Us3",
      "name": "BONK/SOL",
      "dex": "Raydium",
      "category": "Meme",
      "current_tvl": 2345678.90,
      "current_apr": 28.90,
      "prediction_score": 96,
      "predicted_apr_range": {
        "low": 26.5,
        "mid": 32.4,
        "high": 38.9
      },
      "predicted_tvl_change": 12.3,
      "confidence_interval": 7.8,
      "key_factors": [
        "Strong positive APR trend",
        "Increasing liquidity",
        "High trading volume relative to TVL",
        "Popular Meme category with current market momentum"
      ],
      "last_updated": "2025-04-24T12:34:56Z"
    },
    // More predictions...
  ]
}"""
    },
    {
        "method": "GET",
        "path": "/pools/{pool_id}/similar",
        "description": "Find pools similar to a reference pool",
        "parameters": [
            {"name": "pool_id", "type": "string", "description": "The unique identifier of the reference pool", "example": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2"},
            {"name": "limit", "type": "integer", "description": "Maximum number of similar pools to return", "default": "5", "example": "10"},
            {"name": "min_similarity", "type": "number", "description": "Minimum similarity score (0-100)", "default": "50", "example": "70"},
            {"name": "metrics", "type": "string", "description": "Comma-separated list of metrics to use for similarity calculation", "default": "all", "example": "liquidity,apr,volume_24h"}
        ],
        "example_request": """curl -X GET "https://filotanalytics.replit.app/v1/pools/58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2/similar?limit=10" \\
  -H "X-API-Key: your_api_key\"""",
        "example_response": """{
  "status": "success",
  "reference_pool": {
    "id": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
    "name": "SOL/USDC",
    "dex": "Raydium"
  },
  "similar_pools": [
    {
      "id": "7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX",
      "name": "SOL/USDT",
      "dex": "Raydium",
      "similarity_score": 92,
      "liquidity": 18456789.23,
      "volume_24h": 6543210.98,
      "apr": 10.45,
      // Other pool details...
    },
    // More similar pools...
  ]
}"""
    },
    {
        "method": "GET",
        "path": "/market-overview",
        "description": "Get aggregate market statistics",
        "parameters": [
            {"name": "timeframe", "type": "string", "description": "Timeframe for change calculation ('24h', '7d', '30d')", "default": "24h", "example": "7d"}
        ],
        "example_request": """curl -X GET "https://filotanalytics.replit.app/v1/market-overview?timeframe=7d" \\
  -H "X-API-Key: your_api_key\"""",
        "example_response": """{
  "status": "success",
  "data": {
    "total_liquidity": 12345678901.23,
    "total_volume_24h": 3456789012.34,
    "average_apr": 15.67,
    "pool_count": 1234,
    "liquidity_change_7d": 3.45,
    "volume_change_7d": 2.78,
    "top_dexes": [
      {
        "name": "Raydium",
        "liquidity": 4567890123.45,
        "volume_24h": 1234567890.12,
        "pool_count": 456
      },
      // More DEXes...
    ],
    "top_categories": [
      {
        "name": "Major",
        "liquidity": 6789012345.67,
        "volume_24h": 2345678901.23,
        "pool_count": 123
      },
      // More categories...
    ],
    "top_gainers": [
      {
        "id": "T0pGa1n3rP00lIDf0rTh1sW33k",
        "name": "CORE/USDC",
        "dex": "Orca",
        "apr_change_7d": 25.45
      },
      // More gainers...
    ],
    "market_sentiment": "bullish",
    "updated_at": "2025-04-24T12:34:56Z"
  }
}"""
    }
]

# Filter endpoints based on search
if endpoint_filter:
    filtered_endpoints = [e for e in endpoints if endpoint_filter.lower() in e["path"].lower() or endpoint_filter.lower() in e["description"].lower()]
else:
    filtered_endpoints = endpoints

# Display endpoints
for endpoint in filtered_endpoints:
    method = endpoint["method"]
    path = endpoint["path"]
    description = endpoint["description"]
    
    st.markdown(f"""
    <div class="endpoint-container">
        <div class="endpoint-header">
            <span class="endpoint-method">{method}</span>
            <span class="endpoint-path">{path}</span>
        </div>
        <p>{description}</p>
    """, unsafe_allow_html=True)
    
    # Parameters
    if "parameters" in endpoint and endpoint["parameters"]:
        st.markdown("""
        <h4>Parameters</h4>
        <table class="parameter-table">
            <tr>
                <th>Parameter</th>
                <th>Type</th>
                <th>Description</th>
                <th>Default</th>
                <th>Example</th>
            </tr>
        """, unsafe_allow_html=True)
        
        for param in endpoint["parameters"]:
            default_val = param.get("default", "")
            default_cell = f"<td>{default_val}</td>" if default_val else "<td>-</td>"
            
            st.markdown(f"""
            <tr>
                <td><code>{param["name"]}</code></td>
                <td>{param["type"]}</td>
                <td>{param["description"]}</td>
                {default_cell}
                <td><code>{param["example"]}</code></td>
            </tr>
            """, unsafe_allow_html=True)
        
        st.markdown("</table>", unsafe_allow_html=True)
    
    # Example Request
    if "example_request" in endpoint:
        st.markdown("<h4>Example Request</h4>", unsafe_allow_html=True)
        st.code(endpoint["example_request"], language="bash")
    
    # Example Response
    if "example_response" in endpoint:
        st.markdown("<h4>Example Response</h4>", unsafe_allow_html=True)
        st.code(endpoint["example_response"], language="json")
    
    st.markdown("</div>", unsafe_allow_html=True)

# SDKs section
st.markdown("## Client Libraries & SDKs")

sdk_tab1, sdk_tab2 = st.tabs(["JavaScript / TypeScript", "Python"])

with sdk_tab1:
    st.markdown("""
    ### JavaScript SDK Installation
    
    ```bash
    npm install @filot/sdk
    ```
    
    ### Basic Usage
    
    ```javascript
    import { FilotClient } from '@filot/sdk';
    
    // Initialize client with your API key
    const client = new FilotClient('your_api_key');
    
    // Get all pools with filtering
    async function fetchPools() {
      try {
        const pools = await client.getPools({
          dex: 'Raydium',
          min_tvl: 1000000,
          min_apr: 10,
          limit: 20
        });
        
        console.log(`Found ${pools.length} pools matching criteria`);
        console.log(pools);
      } catch (error) {
        console.error('Error fetching pools:', error);
      }
    }
    
    fetchPools();
    ```
    """)

with sdk_tab2:
    st.markdown("""
    ### Python SDK Installation
    
    ```bash
    pip install filot-sdk
    ```
    
    ### Basic Usage
    
    ```python
    from filot_sdk import FilotClient
    
    # Initialize client with your API key
    client = FilotClient(api_key='your_api_key')
    
    # Get all pools with filtering
    try:
        pools = client.get_pools(
            dex='Raydium',
            min_tvl=1000000,
            min_apr=10,
            limit=20
        )
        
        print(f"Found {len(pools)} pools matching criteria")
        print(pools)
    except Exception as e:
        print(f"Error fetching pools: {e}")
    ```
    """)

# Contact and Support
st.markdown("## Support")

st.markdown("""
For issues, questions, or feature requests related to the API, please contact us through one of the following channels:

- **Email**: api-support@filot.io
- **Discord**: [Join our community](https://discord.gg/filot)
- **GitHub**: [File an issue](https://github.com/filot-io/api-issues)

Our support team is available Monday through Friday, 9am-5pm UTC.
""")

st.markdown("---")
st.markdown("Copyright © 2025 FiLot. All rights reserved.")