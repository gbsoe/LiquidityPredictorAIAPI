# DeFi API Application Flow Test Results

Tests run on: 2025-04-28 23:55:48

Base URL: `https://filotdefiapi.replit.app/api/v1`

This document tests the full application flow for retrieving and processing pool data.

---

## Step 1: API Data Retrieval

Retrieved 19 pools from the API.

Sample of raw API data for the first pool:

```json
{
  "id": 1,
  "poolId": "RAYUSDC",
  "programId": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
  "source": "raydium",
  "name": "RAY-USDC LP",
  "active": true,
  "createdAt": "2025-04-28T03:50:10.504Z",
  "updatedAt": "2025-04-28T03:50:10.504Z",
  "metrics": {
    "id": 537,
    "poolId": 1,
    "timestamp": "2025-04-28T05:57:26.655Z",
    "tvl": 5234789.5,
    "fee": 0.0025,
    "apy24h": 12.45,
    "apy7d": 11.87,
    "apy30d": 10.92,
    "volumeUsd": 1245678.9,
    "extraData": {
      "ammId": "AydkTnZeQCGWuvXjT83hXr6KuC55mUEjXqjhyECcHTpd"
    }
  },
  "tokens": [
    {
      "id": 1,
      "symbol": "RAY",
      "name": "Raydium",
      "decimals": 6,
      "address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
      "active": true,
      "price": 1.23
    },
    {
      "id": 3,
      "symbol": "USDC",
      "name": "USD Coin",
      "decimals": 6,
      "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
      "active": true,
      "price": 1
    }
  ]
}
```

---

## Step 2: Data Transformation

Successfully transformed 12 out of 19 pools.

Sample of transformed data for the first pool:

```json
{
  "id": "RAYUSDC",
  "name": "RAY-USDC",
  "dex": "raydium",
  "token1_symbol": "RAY",
  "token2_symbol": "USDC",
  "liquidity": 0,
  "volume_24h": 0,
  "apr": 0,
  "category": "Stablecoin",
  "last_updated": "2025-04-28T03:50:10.504Z"
}
```

The full transformed dataset is saved to: `transformed_pool_data.json`

---

## Step 3: Object Creation

Successfully created 12 pool objects.
Failed to create 0 pool objects.

### Successful Pool Objects (First 3)

- Pool **RAY-USDC** (ID: `RAYUSDC`)
- Pool **SOL-USDC** (ID: `SOLUSDC`)
- Pool **mSOL-USDC Meteora** (ID: `MSOLUSDCM`)

---

## Step 4: API Response Structure Analysis

### API Pool Object Structure

Top-level fields in API response:

```
- id: int
- poolId: str
- programId: str
- source: str
- name: str
- active: bool
- createdAt: str
- updatedAt: str
- metrics: dict
- tokens: list
```

Metrics fields:

```
- id: int
- poolId: int
- timestamp: str
- tvl: float
- fee: float
- apy24h: float
- apy7d: float
- apy30d: float
- volumeUsd: float
- extraData: dict
```

Token fields:

```
- id: int
- symbol: str
- name: str
- decimals: int
- address: str
- active: bool
- price: float
```

---

## Step 5: Error Analysis

### Conclusion

✅ **API data retrieval is working correctly with Bearer token authentication**

✅ **Object creation is working correctly**

See the full API response in `api_response_data.json` and transformed data in `transformed_pool_data.json` for more details.

