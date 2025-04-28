# DeFi API Authentication Method Test Results

Tests run on: 2025-04-28 23:48:25

Base URL: `https://filotdefiapi.replit.app/api/v1`

This document compares different authentication header formats to determine which one works with the API.

---

## Method 1: X-API-Key

```
Headers: {'X-API-Key': 'defi_WyJ71mVrIDzEkzwauPu_FpnRh__W83_l', 'Content-Type': 'application/json'}
Status Code: 401
```

❌ **FAILED**

Error response:
```
{"message":"API key required"}
```

---

## Method 2: Bearer token

```
Headers: {'Authorization': 'Bearer defi_WyJ71mVrIDzEkzwauPu_FpnRh__W83_l', 'Content-Type': 'application/json'}
Status Code: 200
```

✅ **SUCCESS**

Response sample:
```json
[
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
      "volumeUsd
...(truncated)...
```

---

## Method 3: lowercase x-api-key

```
Headers: {'x-api-key': 'defi_WyJ71mVrIDzEkzwauPu_FpnRh__W83_l', 'Content-Type': 'application/json'}
Status Code: 401
```

❌ **FAILED**

Error response:
```
{"message":"API key required"}
```

---

## Method 4: uppercase X-API-KEY

```
Headers: {'X-API-KEY': 'defi_WyJ71mVrIDzEkzwauPu_FpnRh__W83_l', 'Content-Type': 'application/json'}
Status Code: 401
```

❌ **FAILED**

Error response:
```
{"message":"API key required"}
```

---

## Summary

| Method | Status Code | Result |
|--------|-------------|--------|
| X-API-Key | 401 | ❌ Failed |
| Bearer token | 200 | ✅ Success |
| x-api-key (lowercase) | 401 | ❌ Failed |
| X-API-KEY (uppercase KEY) | 401 | ❌ Failed |
