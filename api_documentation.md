# SolPool Insight API Documentation

## Introduction

The SolPool Insight API provides programmatic access to Solana liquidity pool data, analytics, and predictions. This documentation describes the available endpoints, how to authenticate, and how to interpret responses.

## Base URL

All API endpoints are accessible at the base URL path `/api`.

## Authentication

The API uses API key authentication. Include your API key in the request header:

```
X-API-Key: your_api_key_here
```

To obtain an API key, please contact the SolPool Insight team.

## Rate Limiting

Requests are limited to 100 requests per minute per API key. If you exceed this limit, you'll receive a 429 status code.

## Endpoints

### Health Check

Check if the API is operational.

- **URL**: `/api/health`
- **Method**: `GET`
- **Authentication**: Not required
- **Response Example**:

```json
{
  "status": "ok",
  "timestamp": "2025-05-05T07:19:41.930",
  "version": "1.0.0"
}
```

### Get All Pools

Get a list of all available pools with basic metrics.

- **URL**: `/api/pools`
- **Method**: `GET`
- **Authentication**: Required
- **Query Parameters**:
  - `limit` (integer, optional): Maximum number of pools to return. Default: 50
  - `dex` (string, optional): Filter pools by DEX name
- **Response Example**:

```json
{
  "pools": [
    {
      "id": "EGZ7tiLeH62TPV1gL8WwbXGzEPa9zmcpVnnkPKKnrE2U",
      "name": "SOL/USDC",
      "dex": "Raydium",
      "token1": "SOL",
      "token2": "USDC",
      "tvl": 12500000,
      "volume_24h": 4300000,
      "apr": 24.5,
      "fee": 0.25,
      "timestamp": "2025-05-05T07:19:41.930"
    },
    // More pools...
  ],
  "total": 39,
  "timestamp": "2025-05-05T07:20:00.240"
}
```

### Get Pool Details

Get detailed information about a specific pool.

- **URL**: `/api/pools/{pool_id}`
- **Method**: `GET`
- **Authentication**: Required
- **Path Parameters**:
  - `pool_id` (string): ID of the pool
- **Response Example**:

```json
{
  "id": "EGZ7tiLeH62TPV1gL8WwbXGzEPa9zmcpVnnkPKKnrE2U",
  "name": "SOL/USDC",
  "dex": "Raydium",
  "token1": {
    "symbol": "SOL",
    "address": "So11111111111111111111111111111111111111112",
    "price_usd": 145.14
  },
  "token2": {
    "symbol": "USDC",
    "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "price_usd": 1.0
  },
  "metrics": {
    "liquidity": 12500000,
    "volume_24h": 4300000,
    "apr": 24.5,
    "fee": 0.25,
    "tvl_change_24h": 0.032,
    "apr_change_24h": -0.005
  },
  "timestamp": "2025-05-05T07:19:41.930"
}
```

### Get Pool Metrics

Get historical metrics for a specific pool.

- **URL**: `/api/pools/{pool_id}/metrics`
- **Method**: `GET`
- **Authentication**: Required
- **Path Parameters**:
  - `pool_id` (string): ID of the pool
- **Query Parameters**:
  - `days` (integer, optional): Number of days of historical data to return. Default: 7
- **Response Example**:

```json
{
  "pool_id": "EGZ7tiLeH62TPV1gL8WwbXGzEPa9zmcpVnnkPKKnrE2U",
  "metrics": [
    {
      "timestamp": "2025-04-29T00:00:00",
      "liquidity": 12100000,
      "volume": 3900000,
      "apr": 23.8
    },
    {
      "timestamp": "2025-04-30T00:00:00",
      "liquidity": 12200000,
      "volume": 4100000,
      "apr": 24.2
    },
    // More days...
  ],
  "days": 7
}
```

### Get Top Predictions

Get top pool predictions based on specified criteria.

- **URL**: `/api/predictions/top`
- **Method**: `GET`
- **Authentication**: Required
- **Query Parameters**:
  - `category` (string, optional): Sort category - one of "apr", "risk", "performance". Default: "apr"
  - `limit` (integer, optional): Maximum number of predictions to return. Default: 10
  - `ascending` (boolean, optional): Sort direction. Default: false (descending)
- **Response Example**:

```json
{
  "predictions": [
    {
      "pool_id": "EGZ7tiLeH62TPV1gL8WwbXGzEPa9zmcpVnnkPKKnrE2U",
      "pool_name": "SOL/USDC",
      "predicted_apr": 26.2,
      "risk_score": 0.15,
      "performance_class": "high",
      "prediction_timestamp": "2025-05-05T06:00:00"
    },
    // More predictions...
  ],
  "category": "apr",
  "timestamp": "2025-05-05T07:20:00.240"
}
```

### Get Pool Predictions

Get prediction history for a specific pool.

- **URL**: `/api/pools/{pool_id}/predictions`
- **Method**: `GET`
- **Authentication**: Required
- **Path Parameters**:
  - `pool_id` (string): ID of the pool
- **Query Parameters**:
  - `days` (integer, optional): Number of days of prediction history to return. Default: 30
- **Response Example**:

```json
{
  "pool_id": "EGZ7tiLeH62TPV1gL8WwbXGzEPa9zmcpVnnkPKKnrE2U",
  "pool_name": "SOL/USDC",
  "predictions": [
    {
      "predicted_apr": 24.7,
      "risk_score": 0.15,
      "performance_class": "high",
      "prediction_timestamp": "2025-04-05T06:00:00"
    },
    {
      "predicted_apr": 25.1,
      "risk_score": 0.14,
      "performance_class": "high",
      "prediction_timestamp": "2025-04-06T06:00:00"
    },
    // More predictions...
  ]
}
```

## Error Responses

The API returns standard HTTP status codes to indicate success or failure of a request.

- **200 OK**: Request succeeded
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Missing or invalid API key
- **404 Not Found**: Resource not found
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error

Error responses include a JSON body with an error message:

```json
{
  "error": "Pool not found"
}
```

## Implementation Example

Here's an example of how to use the API in Python:

```python
import requests

API_KEY = "your_api_key_here"
BASE_URL = "https://solpoolinsight.replit.app/api"

# Set up headers with API key
headers = {
    "X-API-Key": API_KEY
}

# Get top 5 pools by APR
response = requests.get(
    f"{BASE_URL}/predictions/top", 
    params={"category": "apr", "limit": 5},
    headers=headers
)

if response.status_code == 200:
    predictions = response.json()["predictions"]
    for pred in predictions:
        print(f"{pred['pool_name']}: {pred['predicted_apr']}% APR, Risk: {pred['risk_score']}")
else:
    print(f"Error: {response.json().get('error', 'Unknown error')}")
```

## Support

If you need assistance with the API, please contact the SolPool Insight support team.

---

© 2025 SolPool Insight. All rights reserved.
