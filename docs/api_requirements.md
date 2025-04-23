# Raydium API Service Requirements

## API Overview

The Solana Liquidity Pool Analysis System requires access to a Raydium API service that provides data about liquidity pools on the Solana blockchain. This document outlines the required endpoints, expected data formats, and authentication requirements for the API service.

## Authentication

The system expects the following authentication method:
- **API Key**: Passed in the request header as `X-API-Key`

## Required Endpoints

### 1. List All Pools

**Endpoint**: `/api/pools`  
**Method**: GET  
**Description**: Returns a list of all available liquidity pools.

**Expected Response Format**:
```json
[
  {
    "id": "pool_address_string",
    "name": "TOKEN1/TOKEN2",
    "token1Symbol": "TOKEN1",
    "token2Symbol": "TOKEN2",
    "liquidity": 1000000.00,
    "volume24h": 500000.00,
    "apr": 12.5,
    "token1Address": "token1_address_string",
    "token2Address": "token2_address_string",
    "createdAt": "2023-01-01T00:00:00Z"
  },
  ...
]
```

### 2. Pool Details

**Endpoint**: `/api/pools/{pool_id}`  
**Method**: GET  
**Description**: Returns detailed information about a specific pool.

**Expected Response Format**:
```json
{
  "id": "pool_address_string",
  "name": "TOKEN1/TOKEN2",
  "token1Symbol": "TOKEN1",
  "token2Symbol": "TOKEN2",
  "token1Address": "token1_address_string",
  "token2Address": "token2_address_string",
  "liquidity": 1000000.00,
  "volume24h": 500000.00,
  "apr": 12.5,
  "fees24h": 1500.00,
  "token1Reserve": 50000.00,
  "token2Reserve": 150000.00,
  "swapCount24h": 1200,
  "createdAt": "2023-01-01T00:00:00Z",
  "updatedAt": "2023-04-01T12:30:45Z"
}
```

### 3. Pool Metrics

**Endpoint**: `/api/metrics/{pool_id}`  
**Method**: GET  
**Description**: Returns performance metrics for a specific pool.

**Expected Response Format**:
```json
{
  "pool_id": "pool_address_string",
  "liquidity": 1000000.00,
  "volume24h": 500000.00,
  "volume7d": 3000000.00,
  "apr": 12.5,
  "fees24h": 1500.00,
  "priceChange24h": 2.5,
  "priceChange7d": -1.2,
  "swapCount24h": 1200,
  "uniqueUsers24h": 350,
  "timestamp": "2023-04-23T00:00:00Z"
}
```

### 4. Blockchain Statistics

**Endpoint**: `/api/blockchain/stats`  
**Method**: GET  
**Description**: Returns current Solana blockchain statistics.

**Expected Response Format**:
```json
{
  "slot": 123456789,
  "blockHeight": 87654321,
  "blockTime": 400,
  "tps": 2500,
  "validatorCount": 1700,
  "solPrice": 102.45,
  "timestamp": "2023-04-23T00:00:00Z"
}
```

### 5. Token Prices

**Endpoint**: `/api/tokens/prices`  
**Method**: GET  
**Description**: Returns current prices for specified tokens.

**Query Parameters**:
- `symbols`: Comma-separated list of token symbols

**Expected Response Format**:
```json
{
  "SOL": 102.45,
  "USDC": 1.00,
  "RAY": 1.23,
  "MNGO": 0.45,
  "timestamp": "2023-04-23T00:00:00Z"
}
```

### 6. Pool Historical Data

**Endpoint**: `/api/pools/{pool_id}/history`  
**Method**: GET  
**Description**: Returns historical metrics for a specific pool.

**Query Parameters**:
- `days`: Number of days of history to return (default: 30)
- `interval`: Data interval in hours (default: 24)

**Expected Response Format**:
```json
[
  {
    "timestamp": "2023-04-22T00:00:00Z",
    "liquidity": 980000.00,
    "volume": 490000.00,
    "apr": 12.3,
    "token1Price": 1.02,
    "token2Price": 0.98
  },
  {
    "timestamp": "2023-04-21T00:00:00Z",
    "liquidity": 975000.00,
    "volume": 510000.00,
    "apr": 12.8,
    "token1Price": 1.01,
    "token2Price": 0.99
  },
  ...
]
```

## Error Handling

The API should return standard HTTP status codes and consistent error responses:

```json
{
  "error": "error_code",
  "message": "A descriptive error message",
  "statusCode": 404
}
```

Common error codes:
- `not_found`: The requested resource was not found
- `invalid_parameters`: The request contains invalid parameters
- `rate_limit_exceeded`: The client has exceeded rate limits
- `authentication_error`: Invalid or missing API key

## Rate Limiting

Please provide information about:
- Maximum requests per second
- Maximum requests per day
- How rate limit headers are returned
- Any special considerations for burst traffic

## Additional Requirements

1. **Data Freshness**: Pool data should be updated at least every 5 minutes
2. **Availability**: 99.9% uptime is required
3. **Response Time**: API responses should be returned within 500ms
4. **Data Completeness**: All fields in the response formats should be populated whenever possible
5. **Historical Data**: At least 90 days of historical data should be available
6. **Documentation**: Comprehensive documentation for all endpoints, including any additional features not specified here

## Implementation Notes

The current system uses Python's requests library to interact with the API, with retry logic for transient failures. The system is designed to handle API outages gracefully with appropriate error logging.

Please ensure that the API service meets these requirements to ensure proper functioning of the Solana Liquidity Pool Analysis System.