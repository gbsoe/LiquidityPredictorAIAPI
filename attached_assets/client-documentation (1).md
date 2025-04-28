# DeFi Aggregation API Documentation

## Overview

This documentation explains how to use our DeFi Aggregation API to retrieve authentic on-chain data from Solana DEXes including Raydium, Meteora, and Orca. Our API provides real-time access to:

- Authentic base58-encoded Solana pool addresses
- Total Value Locked (TVL) for liquidity pools
- Fee percentages for each pool
- Annual Percentage Rate (APR) metrics for 24hr, 7d, and 30d periods
- Token prices and trading volume

## Base URL

All API endpoints should be prefixed with the following base URL:

```
https://filotdefiapi.replit.app/api/v1
```

## Authentication

All API requests require an API key for authentication. To use the API, include your API key in the request header:

```
X-API-KEY: defi_your_api_key_here
```

## Endpoints

### 1. Get All Pools

Retrieve a list of all available liquidity pools across supported DEXes.

**Endpoint:** `GET https://filotdefiapi.replit.app/api/v1/pools`

**Sample Response:**
```json
{
  "pools": [
    {
      "poolId": "8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj",
      "name": "mSOL-USDC LP",
      "source": "Meteora",
      "tvl": 3400000,
      "fee": 0.003,
      "apr24h": 3.96,
      "apr7d": 3.64,
      "apr30d": 3.37,
      "tokens": [
        {
          "symbol": "mSOL",
          "address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
          "price": 163.27
        },
        {
          "symbol": "USDC",
          "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
          "price": 1.0
        }
      ],
      "volumeUsd": 408000
    },
    // Additional pools...
  ],
  "total": 20,
  "page": 1,
  "limit": 50
}
```

### 2. Get Pool by ID

Retrieve detailed information for a specific pool using its authentic base58-encoded Solana address.

**Endpoint:** `GET https://filotdefiapi.replit.app/api/v1/pools/{poolId}`

**Example:** `GET https://filotdefiapi.replit.app/api/v1/pools/8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj`

**Sample Response:**
```json
{
  "pool": {
    "poolId": "8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj",
    "name": "mSOL-USDC LP",
    "source": "Meteora",
    "tvl": 3400000,
    "fee": 0.003,
    "apr24h": 3.96,
    "apr7d": 3.64,
    "apr30d": 3.37,
    "tokens": [
      {
        "symbol": "mSOL",
        "address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
        "price": 163.27,
        "poolShare": 0.513
      },
      {
        "symbol": "USDC",
        "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "price": 1.0,
        "poolShare": 0.487
      }
    ],
    "volumeUsd": {
      "24h": 408000,
      "7d": 2516000,
      "30d": 9350000
    },
    "reserves": {
      "mSOL": 10657.43,
      "USDC": 1740000
    },
    "historicalMetrics": {
      // Historical APR and TVL data points...
    }
  }
}
```

### 3. Get Pools by DEX

Retrieve pools from a specific DEX (Raydium, Meteora, or Orca).

**Endpoint:** `GET https://filotdefiapi.replit.app/api/v1/pools?source={dexName}`

**Example:** `GET https://filotdefiapi.replit.app/api/v1/pools?source=Meteora`

**Sample Response:**
```json
{
  "pools": [
    {
      "poolId": "8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj",
      "name": "mSOL-USDC LP",
      "source": "Meteora",
      "tvl": 3400000,
      "fee": 0.003,
      "apr24h": 3.96,
      "apr7d": 3.64,
      "apr30d": 3.37,
      "tokens": [
        // Token details...
      ],
      "volumeUsd": 408000
    },
    // Additional Meteora pools...
  ],
  "total": 6,
  "page": 1,
  "limit": 50
}
```

### 4. Get Pools by Token

Retrieve pools that include a specific token in their pair.

**Endpoint:** `GET https://filotdefiapi.replit.app/api/v1/pools?token={tokenSymbol}`

**Example:** `GET https://filotdefiapi.replit.app/api/v1/pools?token=SOL`

**Sample Response:**
```json
{
  "pools": [
    {
      "poolId": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
      "name": "SOL-USDC LP",
      "source": "Raydium",
      "tvl": 15700000,
      "fee": 0.0025,
      "apr24h": 8.02,
      "apr7d": 7.38,
      "apr30d": 6.82,
      "tokens": [
        {
          "symbol": "SOL",
          "address": "So11111111111111111111111111111111111111112",
          "price": 155.42
        },
        {
          "symbol": "USDC",
          "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
          "price": 1.0
        }
      ],
      "volumeUsd": 2826000
    },
    // Additional SOL pools...
  ],
  "total": 7,
  "page": 1,
  "limit": 50
}
```

### 5. Get Top Pools by APR

Retrieve the top-performing pools ordered by APR.

**Endpoint:** `GET https://filotdefiapi.replit.app/api/v1/pools?sort=apr24h&order=desc&limit=10`

**Sample Response:**
```json
{
  "pools": [
    // Top 10 pools by 24-hour APR...
  ],
  "total": 10,
  "page": 1,
  "limit": 10
}
```

### 6. Get Token Information

Retrieve information about a specific token.

**Endpoint:** `GET https://filotdefiapi.replit.app/api/v1/tokens/{tokenSymbol}`

**Example:** `GET https://filotdefiapi.replit.app/api/v1/tokens/SOL`

**Sample Response:**
```json
{
  "token": {
    "symbol": "SOL",
    "name": "Solana",
    "address": "So11111111111111111111111111111111111111112",
    "decimals": 9,
    "price": 155.42,
    "priceHistory": {
      // Historical price data...
    },
    "pools": [
      // Pools containing this token...
    ]
  }
}
```

## Understanding the Data

### Pool IDs

All pool IDs are authentic base58-encoded Solana addresses that represent the actual on-chain liquidity pools. These IDs can be used directly with Solana blockchain explorers or SDKs for additional on-chain verification.

### TVL (Total Value Locked)

TVL represents the total value of all assets locked in a specific liquidity pool, denominated in USD. This metric indicates the pool's liquidity depth.

### Fee Percentage

The fee percentage represents the trading fee charged by the DEX for each swap in the pool. This fee contributes to the pool's APR for liquidity providers.

### APR Metrics

Our API provides three different APR metrics:

1. **apr24h**: Annual Percentage Rate based on the last 24 hours of trading activity
2. **apr7d**: Annual Percentage Rate based on the last 7 days of trading activity
3. **apr30d**: Annual Percentage Rate based on the last 30 days of trading activity

APR is calculated using the formula:
```
APR = (Trading Volume × Fee Percentage / TVL) × 365 × 100%
```

## Error Handling

The API returns standard HTTP status codes:

- 200: Success
- 400: Bad Request
- 401: Unauthorized (invalid API key)
- 404: Not Found
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error

Errors include a message explaining the issue:

```json
{
  "error": "Invalid pool ID format. Must be a valid base58-encoded Solana address.",
  "status": 400
}
```

## Rate Limits

- Free tier: 100 requests per minute
- Premium tier: 1,000 requests per minute
- Enterprise tier: Custom limits

## SDK Integration

For direct integration with your applications, we provide SDKs in multiple programming languages:

- JavaScript/TypeScript
- Python
- Rust

Example of JavaScript SDK usage:

```javascript
import { DefiAggregationClient } from 'defi-aggregation-sdk';

const client = new DefiAggregationClient({
  apiKey: 'your_api_key_here',
  baseUrl: 'https://filotdefiapi.replit.app/api/v1'
});

async function getTopPools() {
  const pools = await client.getPools({
    sort: 'apr24h',
    order: 'desc',
    limit: 5
  });
  
  console.log('Top 5 pools by APR:', pools);
}

getTopPools();
```

## Contact and Support

For additional support or questions about the API, please contact:

- Email: support@defi-aggregation-api.com
- Discord: https://discord.gg/defi-aggregation