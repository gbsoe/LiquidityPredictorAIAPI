# SolPool Insight API Documentation

## Overview

SolPool Insight provides a powerful RESTful API for accessing comprehensive data on Solana-based liquidity pools across multiple DEXes. Our API includes real-time metrics, historical data, advanced predictions, and sophisticated filtering capabilities.

With our API, you can:
- Access data on thousands of liquidity pools across all major Solana DEXes
- Retrieve historical performance metrics with custom timeframes
- Get advanced ML-based predictions on future pool performance
- Filter and sort pools based on various metrics
- Compare and analyze pools across different categories
- Access specialized meme coin analytics
- Use mobile-optimized endpoints for lightweight data retrieval

## Base URL

```
https://api.solpool-insight.com/v1
```

## Authentication

All API requests require an API key passed in the `X-API-Key` header:

```
X-API-Key: your_api_key
```

To obtain an API key, please [register for an account](https://solpool-insight.com/register) or contact us at api@solpool-insight.com.

## Rate Limiting

API request limits by tier:
- **Free tier**: 100 requests per hour
- **Standard tier**: 1,000 requests per hour
- **Enterprise tier**: 10,000 requests per hour
- **Mobile tier**: 500 requests per hour (optimized for mobile apps)

Rate limiting headers are included in all responses:
```
X-Rate-Limit-Limit: 1000
X-Rate-Limit-Remaining: 985
X-Rate-Limit-Reset: 1619231400
```

## Endpoints

### Get All Pools

Retrieve a list of all pools with optional filtering.

```
GET /pools
```

#### Query Parameters

| Parameter    | Type   | Description                                          | Default | Example           |
|--------------|--------|------------------------------------------------------|---------|-------------------|
| dex          | string | Filter by DEX name                                   | null    | Raydium           |
| category     | string | Filter by pool category                              | null    | Meme              |
| min_tvl      | number | Minimum TVL threshold                                | 0       | 1000000           |
| max_tvl      | number | Maximum TVL threshold                                | null    | 50000000          |
| min_apr      | number | Minimum APR threshold (percentage)                   | 0       | 10                |
| max_apr      | number | Maximum APR threshold (percentage)                   | null    | 50                |
| min_volume   | number | Minimum 24h volume                                   | 0       | 100000            |
| token        | string | Filter pools containing this token                   | null    | SOL               |
| limit        | integer| Maximum number of results                            | 100     | 50                |
| offset       | integer| Number of results to skip (for pagination)           | 0       | 100               |
| sort_by      | string | Field to sort by                                     | liquidity| apr              |
| sort_dir     | string | Sort direction ('asc' or 'desc')                    | desc    | asc               |
| min_prediction| number | Minimum prediction score (0-100)                    | 0       | 80                |
| trend        | string | Filter by trend direction ('increasing', 'decreasing', 'stable')| null | increasing |
| trend_field  | string | Field to apply trend filter to ('apr', 'tvl')        | apr     | tvl               |
| trend_days   | integer| Number of days for trend calculation                 | 7       | 30                |
| trend_threshold| number| Minimum change percentage for trend filtering       | 1       | 5                 |

#### Example Request

```bash
curl -X GET "https://api.solanapoolanalytics.com/v1/pools?dex=Raydium&min_tvl=1000000&min_apr=10" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
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
}
```

### Get Pool by ID

Retrieve detailed information about a specific pool.

```
GET /pools/{pool_id}
```

#### Path Parameters

| Parameter | Type   | Description                   | Example                                      |
|-----------|--------|-------------------------------|----------------------------------------------|
| pool_id   | string | The unique identifier of the pool | 58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2 |

#### Example Request

```bash
curl -X GET "https://api.solanapoolanalytics.com/v1/pools/58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
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
}
```

### Get Pool Historical Data

Retrieve historical data for a specific pool.

```
GET /pools/{pool_id}/history
```

#### Path Parameters

| Parameter | Type   | Description                   | Example                                      |
|-----------|--------|-------------------------------|----------------------------------------------|
| pool_id   | string | The unique identifier of the pool | 58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2 |

#### Query Parameters

| Parameter | Type    | Description                                    | Default | Example |
|-----------|---------|------------------------------------------------|---------|---------|
| days      | integer | Number of days of history to retrieve          | 30      | 60      |
| interval  | string  | Time interval ('hour', 'day', 'week')          | day     | hour    |
| metrics   | string  | Comma-separated list of metrics to include     | all     | apr,tvl |

#### Example Request

```bash
curl -X GET "https://api.solanapoolanalytics.com/v1/pools/58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2/history?days=60&interval=day" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
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
}
```

### Get DEX Statistics

Retrieve statistics and aggregate data for a specific DEX.

```
GET /dexes/{dex_name}
```

#### Path Parameters

| Parameter | Type   | Description              | Example  |
|-----------|--------|--------------------------|----------|
| dex_name  | string | The name of the DEX      | Raydium  |

#### Example Request

```bash
curl -X GET "https://api.solanapoolanalytics.com/v1/dexes/Raydium" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
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
}
```

### Get Category Statistics

Retrieve statistics and aggregate data for a specific pool category.

```
GET /categories/{category_name}
```

#### Path Parameters

| Parameter     | Type   | Description              | Example |
|---------------|--------|--------------------------|---------|
| category_name | string | The name of the category | Meme    |

#### Example Request

```bash
curl -X GET "https://api.solanapoolanalytics.com/v1/categories/Meme" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
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
}
```

### Get Token Pools

Retrieve all pools containing a specific token.

```
GET /tokens/{token_symbol}/pools
```

#### Path Parameters

| Parameter    | Type   | Description              | Example |
|--------------|--------|--------------------------|---------|
| token_symbol | string | The token symbol         | BONK    |

#### Query Parameters

Same as `/pools` endpoint.

#### Example Request

```bash
curl -X GET "https://api.solanapoolanalytics.com/v1/tokens/BONK/pools?min_liquidity=1000000" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
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
}
```

### Get Token Information

Retrieve information about a specific token.

```
GET /tokens/{token_symbol}
```

#### Path Parameters

| Parameter    | Type   | Description              | Example |
|--------------|--------|--------------------------|---------|
| token_symbol | string | The token symbol         | BONK    |

#### Example Request

```bash
curl -X GET "https://api.solanapoolanalytics.com/v1/tokens/BONK" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
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
}
```

### Get Token Price History

Retrieve historical price data for a specific token.

```
GET /tokens/{token_symbol}/price-history
```

#### Path Parameters

| Parameter    | Type   | Description              | Example |
|--------------|--------|--------------------------|---------|
| token_symbol | string | The token symbol         | BONK    |

#### Query Parameters

| Parameter | Type    | Description                               | Default | Example |
|-----------|---------|-------------------------------------------|---------|---------|
| days      | integer | Number of days of history to retrieve     | 30      | 60      |
| interval  | string  | Time interval ('hour', 'day', 'week')     | day     | hour    |

#### Example Request

```bash
curl -X GET "https://api.solanapoolanalytics.com/v1/tokens/BONK/price-history?days=60&interval=day" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
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
}
```

### Get Predictions

Retrieve ML-based predictions for pools.

```
GET /predictions
```

#### Query Parameters

| Parameter    | Type   | Description                                   | Default | Example    |
|--------------|--------|-----------------------------------------------|---------|------------|
| min_score    | number | Minimum prediction score (0-100)              | 0       | 80         |
| category     | string | Filter by pool category                       | null    | Meme       |
| dex          | string | Filter by DEX name                            | null    | Raydium    |
| min_tvl      | number | Minimum TVL threshold                         | 0       | 1000000    |
| limit        | integer| Maximum number of results                     | 20      | 50         |
| offset       | integer| Number of results to skip (for pagination)    | 0       | 20         |
| sort_by      | string | Field to sort predictions by                  | score   | potential_apr |

#### Example Request

```bash
curl -X GET "https://api.solanapoolanalytics.com/v1/predictions?min_score=80&category=Meme" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
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
}
```

### Get Similar Pools

Find pools that are similar to a reference pool.

```
GET /pools/{pool_id}/similar
```

#### Path Parameters

| Parameter | Type   | Description                         | Example                                      |
|-----------|--------|-------------------------------------|----------------------------------------------|
| pool_id   | string | The reference pool ID               | 58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2 |

#### Query Parameters

| Parameter    | Type    | Description                            | Default | Example |
|--------------|---------|----------------------------------------|---------|---------|
| limit        | integer | Maximum number of similar pools to return | 5     | 10      |
| metrics      | string  | Comma-separated metrics to use for similarity | tvl,apr,volume | volatility,apr |
| min_similarity | number | Minimum similarity score (0-100)      | 50      | 70      |

## Mobile-Optimized Endpoints

The following endpoints are specifically optimized for mobile applications, providing lightweight responses with essential data to minimize bandwidth usage and improve loading times on mobile devices.

### Get Mobile Pool Summary

Retrieve a lightweight summary of pools, optimized for mobile devices.

```
GET /mobile/pools/summary
```

#### Query Parameters

| Parameter    | Type   | Description                                  | Default | Example    |
|--------------|--------|----------------------------------------------|---------|------------|
| category     | string | Filter by pool category                      | null    | Meme       |
| dex          | string | Filter by DEX name                           | null    | Raydium    |
| min_tvl      | number | Minimum TVL threshold                        | 0       | 1000000    |
| limit        | integer| Maximum number of results                    | 20      | 50         |
| sort_by      | string | Field to sort by                             | liquidity| apr       |

#### Example Request

```bash
curl -X GET "https://api.solpool-insight.com/v1/mobile/pools/summary?category=Meme&limit=10" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
  "status": "success",
  "count": 10,
  "data": [
    {
      "id": "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K",
      "name": "BONK/USDC",
      "dex": "Meteora",
      "category": "Meme",
      "apr": 25.67,
      "prediction_score": 92,
      "trend": "increasing"
    },
    // More pool summaries...
  ]
}
```

### Get Mobile Pool Details

Retrieve mobile-optimized details for a specific pool.

```
GET /mobile/pools/{pool_id}
```

#### Path Parameters

| Parameter | Type   | Description                   | Example                                      |
|-----------|--------|-------------------------------|----------------------------------------------|
| pool_id   | string | The unique identifier of the pool | M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K |

#### Example Request

```bash
curl -X GET "https://api.solpool-insight.com/v1/mobile/pools/M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
  "status": "success",
  "data": {
    "id": "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K",
    "name": "BONK/USDC",
    "dex": "Meteora",
    "category": "Meme",
    "token1_symbol": "BONK",
    "token2_symbol": "USDC",
    "liquidity": 5432167.89,
    "volume_24h": 1987654.32,
    "apr": 25.67,
    "apr_change_24h": 1.32,
    "apr_change_7d": 3.54,
    "tvl_change_24h": 2.1,
    "tvl_change_7d": 5.8,
    "prediction_score": 92,
    "prediction_summary": "Likely to increase APR by 2-4% in the next 7 days",
    "key_factors": [
      "Strong positive APR trend",
      "Increasing volume"
    ]
  }
}
```

### Get Mobile Top Predictions

Retrieve top predictions in a mobile-optimized format.

```
GET /mobile/predictions/top
```

#### Query Parameters

| Parameter    | Type   | Description                                   | Default | Example    |
|--------------|--------|-----------------------------------------------|---------|------------|
| category     | string | Filter by pool category                       | null    | Meme       |
| limit        | integer| Maximum number of results                     | 10      | 20         |

#### Example Request

```bash
curl -X GET "https://api.solpool-insight.com/v1/mobile/predictions/top?limit=5" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
  "status": "success",
  "count": 5,
  "data": [
    {
      "pool_id": "B0nkD2EW5B0nK1nG51mECoiNSolANaPooL5Us3",
      "name": "BONK/SOL",
      "dex": "Raydium",
      "category": "Meme",
      "prediction_score": 96,
      "predicted_change": "+32.4%",
      "confidence": "High",
      "summary": "Strong APR increase expected"
    },
    // More predictions...
  ]
}
```

### Get Mobile Historical Chart Data

Retrieve optimized chart data for mobile visualization.

```
GET /mobile/pools/{pool_id}/chart
```

#### Path Parameters

| Parameter | Type   | Description                   | Example                                      |
|-----------|--------|-------------------------------|----------------------------------------------|
| pool_id   | string | The unique identifier of the pool | M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K |

#### Query Parameters

| Parameter | Type    | Description                               | Default | Example |
|-----------|---------|-------------------------------------------|---------|---------|
| days      | integer | Number of days of history to retrieve     | 30      | 60      |
| metric    | string  | Metric to chart ('apr', 'tvl', 'volume') | apr     | tvl     |

#### Example Request

```bash
curl -X GET "https://api.solpool-insight.com/v1/mobile/pools/M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K/chart?days=30&metric=apr" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
  "status": "success",
  "pool_id": "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K",
  "metric": "apr",
  "unit": "%",
  "data": {
    "labels": ["Apr 1", "Apr 8", "Apr 15", "Apr 22", "Apr 29"],
    "values": [23.4, 24.2, 24.8, 25.1, 25.7],
    "min": 23.4,
    "max": 25.7,
    "avg": 24.64
  }
}
```

#### Example Request

```bash
curl -X GET "https://api.solanapoolanalytics.com/v1/pools/58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2/similar?limit=10" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
  "status": "success",
  "reference_pool": {
    "id": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
    "name": "SOL/USDC",
    "dex": "Raydium"
  },
  "data": [
    {
      "id": "HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ",
      "name": "SOL/USDC",
      "dex": "Orca",
      "similarity_score": 92.5,
      "liquidity": 22345678.90,
      "volume_24h": 7654321.09,
      "apr": 13.56,
      "prediction_score": 79
    },
    // More similar pools...
  ]
}
```

### Get Market Overview

Retrieve aggregate market statistics.

```
GET /market-overview
```

#### Example Request

```bash
curl -X GET "https://api.solanapoolanalytics.com/v1/market-overview" \
  -H "X-API-Key: your_api_key"
```

#### Example Response

```json
{
  "status": "success",
  "data": {
    "total_pools": 1245,
    "total_liquidity": 12345678901.23,
    "total_volume_24h": 2345678901.23,
    "average_apr": 14.5,
    "dex_breakdown": {
      "Raydium": {
        "pool_count": 245,
        "total_liquidity": 5678901234.56,
        "total_volume_24h": 1234567890.12,
        "average_apr": 14.2
      },
      "Orca": {
        "pool_count": 187,
        "total_liquidity": 3456789012.34,
        "total_volume_24h": 567890123.45,
        "average_apr": 15.1
      },
      // More DEXes...
    },
    "category_breakdown": {
      "Major": {
        "pool_count": 212,
        "total_liquidity": 6789012345.67,
        "total_volume_24h": 1345678901.23,
        "average_apr": 11.3
      },
      "Meme": {
        "pool_count": 345,
        "total_liquidity": 2345678901.23,
        "total_volume_24h": 456789012.34,
        "average_apr": 23.5
      },
      // More categories...
    },
    "trends": {
      "overall_tvl_change_24h": 2.1,
      "overall_volume_change_24h": 5.6,
      "overall_apr_change_7d": -0.3,
      "highest_growth_category": "Meme",
      "highest_growth_dex": "Jupiter"
    }
  }
}
```

## Error Handling

The API returns standard HTTP status codes:

- 200: Success
- 400: Bad request (invalid parameters)
- 401: Unauthorized (invalid API key)
- 404: Resource not found
- 429: Rate limit exceeded
- 500: Server error

Error responses include a JSON body with error details:

```json
{
  "status": "error",
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {
    "parameter": "min_tvl",
    "message": "Must be a non-negative number"
  }
}
```

Common error codes:

| Code                | Description                                      |
|---------------------|--------------------------------------------------|
| INVALID_PARAMETER   | One or more parameters are invalid               |
| AUTHENTICATION_ERROR| Missing or invalid API key                       |
| RESOURCE_NOT_FOUND  | The requested resource was not found             |
| RATE_LIMIT_EXCEEDED | You have exceeded your rate limit                |
| INTERNAL_ERROR      | An internal server error occurred                |

## SDKs and Code Examples

### JavaScript / TypeScript

```typescript
import { SolanaPoolsClient } from 'solana-pools-sdk';

// Initialize client
const client = new SolanaPoolsClient('your_api_key');

// Get top pools by APR
async function getTopPoolsByAPR() {
  const response = await client.getPools({
    min_tvl: 1000000,  // Minimum $1M TVL
    sort_by: 'apr',
    sort_dir: 'desc',
    limit: 10
  });
  
  console.log(`Found ${response.count} pools`);
  console.log(response.data);
}

// Get predictions for meme coin pools
async function getMemePoolPredictions() {
  const predictions = await client.getPredictions({
    category: 'Meme',
    min_score: 80,
    min_tvl: 500000  // $500k minimum TVL
  });
  
  console.log(`Found ${predictions.count} high-potential meme pools`);
  predictions.data.forEach(pool => {
    console.log(`${pool.name}: Score ${pool.prediction_score}/100`);
  });
}
```

### Python

```python
from solana_pools_sdk import SolanaPoolsClient

# Initialize client
client = SolanaPoolsClient('your_api_key')

# Get pools with BONK token
def get_bonk_pools():
    pools = client.get_token_pools('BONK', min_liquidity=1000000)
    print(f"Found {pools['count']} BONK pools")
    
    for pool in pools['data']:
        print(f"{pool['name']} on {pool['dex']}: {pool['apr']}% APR")

# Track historical APR for a specific pool
def track_pool_apr(pool_id, days=60):
    history = client.get_pool_history(pool_id, days=days, interval='day')
    
    # Extract APR values and dates for plotting
    dates = [data_point['timestamp'] for data_point in history['data']]
    apr_values = [data_point['apr'] for data_point in history['data']]
    
    print(f"APR range: {min(apr_values):.2f}% - {max(apr_values):.2f}%")
    
    # You can now plot this data with matplotlib or any other library
    # plt.plot(dates, apr_values)
    # plt.title(f"APR History for {history['pool_id']}")
    # plt.show()
```

## Webhook Notifications

You can set up webhooks to receive notifications when certain events or conditions occur:

- Pool APR changes beyond a threshold
- New pools being added
- High-potential pools identified by ML models
- Significant TVL changes

Contact us to set up webhook notifications.

## Support

For API support, please contact api@solanapoolanalytics.com.

Documentation last updated: April 25, 2025