"""
API endpoint and structure documentation for the new Raydium Trader API
This serves as documentation for the new API structure and endpoints.
"""

# Base URL for the API
BASE_URL = "https://raydium-trader-filot.replit.app"

# API Endpoints
ENDPOINTS = {
    # Core endpoints
    "health": "/health",                 # Health check endpoint
    "pools": "/api/pools",               # Get liquidity pools
    "pool_details": "/api/pool/{}",      # Get specific pool details (requires pool ID)
    
    # Note: The /api/tokens endpoint doesn't exist in this API version
    # Use cached token data instead for token information
}

# Response Structure for /api/pools
"""
{
  "pools": {
    "bestPerformance": [
      {
        "id": "x4ND6LEXnrj3ufeCTY8RSuo3qbktirsz4tqPus5SjrH",
        "tokenPair": "BOOP/USDC",
        "baseMint": "boopkpWqe68MSxLqBGogs8ZbUDN4GXaLhFwNKKsiJvQ5",
        "quoteMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "apr24h": 115.32,
        "apr7d": 115.32,
        "apr30d": 115.32,
        "liquidityUsd": 107693.82,
        "price": 0.0001621,
        "volume24h": 109.15,
        "volume7d": 764.05,
        "formatted": {
          "apr24h": "115.32%",
          "apr7d": "115.32%",
          "apr30d": "115.32%",
          "liquidityUsd": "$107,693.82",
          "price": "$0.0001621",
          "volume24h": "$109.15",
          "volume7d": "$764.05"
        }
      },
      ...
    ],
    "topStable": [
      {
        "id": "3mYd7rAYK27uH19xfkCHYJ8S2ybJZKpZJKpMuaQRSYun",
        "tokenPair": "USDC/USDT",
        "baseMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "quoteMint": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "apr24h": 3.75,
        "apr7d": 3.75,
        "apr30d": 3.75,
        "liquidityUsd": 14089423.12,
        "price": 0.999,
        "volume24h": 461986.81,
        "volume7d": 3233907.67,
        "formatted": {
          "apr24h": "3.75%",
          "apr7d": "3.75%",
          "apr30d": "3.75%",
          "liquidityUsd": "$14,089,423.12",
          "price": "$0.999",
          "volume24h": "$461,986.81",
          "volume7d": "$3,233,907.67"
        }
      },
      ...
    ]
  },
  "timestamp": "2025-05-04T08:51:38.596Z"
}
"""