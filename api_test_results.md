# DeFi API Test Results

Test run at: 2025-04-30T02:47:26.214464

## Test 1: Get Tokens


Retrieved 8 tokens

```json
[
  {
    "id": 1,
    "symbol": "mSoL",
    "name": "mSoL",
    "decimals": 6,
    "address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
    "active": true,
    "price": 0
  },
  {
    "id": 2,
    "symbol": "EPjF",
    "name": "EPjF",
    "decimals": 6,
    "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "active": true,
    "price": 0
  },
  {
    "id": 4,
    "symbol": "So11",
    "name": "So11",
    "decimals": 6,
    "address": "So11111111111111111111111111111111111111112",
    "active": true,
    "price": 0
  },
  {
    "id": 3,
    "symbol": "9n4n",
    "name": "9n4n",
    "decimals": 6,
    "address": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "active": true,
    "price": 0
  },
  {
    "id": 5,
    "symbol": "4k3D",
    "name": "4k3D",
    "decimals": 6,
    "address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "active": true,
    "price": 0
  },
  {
    "id": 8,
    "symbol": "Es9v",
    "name": "Es9v",
    "decimals": 6,
    "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "active": true,
    "price": 0
  },
  {
    "id": 12,
    "symbol": "7vfC",
    "name": "7vfC",
    "decimals": 6,
    "address": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
    "active": true,
    "price": 0
  },
  {
    "id": 14,
    "symbol": "DezX",
    "name": "DezX",
    "decimals": 6,
    "address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "active": true,
    "price": 0
  }
]
```

## Test 2: Get Pools (First 2)


Retrieved 13 pools

```json
[
  {
    "id": 1,
    "poolId": "8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj",
    "programId": "LbVRzDTJEz6PA1RBxMK2B2CalYJgF8C9tYMszWkRESZ",
    "source": "meteora",
    "name": "mSoL-EPjF Meteora LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:45.530Z",
    "updatedAt": "2025-04-29T13:34:45.530Z",
    "metrics": {
      "id": 261,
      "poolId": 1,
      "timestamp": "2025-04-30T02:17:27.005Z",
      "tvl": 2000000,
      "fee": 0.003,
      "apy24h": 11.570175,
      "apy7d": 10.644561,
      "apy30d": 9.834649,
      "volumeUsd": 200000,
      "extraData": {
        "bins": "120",
        "concentrationBounds": "medium"
      }
    },
    "tokens": []
  },
  {
    "id": 2,
    "poolId": "FLKdMRsS7dvj9mMsEBxciUJc3TKJcEmVUbQRf5UnYEpw",
    "programId": "LbVRzDTJEz6PA1RBxMK2B2CalYJgF8C9tYMszWkRESZ",
    "source": "meteora",
    "name": "So11-EPjF Meteora LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:45.565Z",
    "updatedAt": "2025-04-29T13:34:45.565Z",
    "metrics": {
      "id": 262,
      "poolId": 2,
      "timestamp": "2025-04-30T02:17:27.007Z",
      "tvl": 3000000,
      "fee": 0.003,
      "apy24h": 11.570175,
      "apy7d": 10.644561,
      "apy30d": 9.834649,
      "volumeUsd": 300000,
      "extraData": {
        "bins": "120",
        "concentrationBounds": "medium"
      }
    },
    "tokens": []
  },
  {
    "id": 3,
    "poolId": "Cv3YJJvCZQJKP5G6AHh77GVbewGZUVs4Y1WJ6Yc3HbQj",
    "programId": "LbVRzDTJEz6PA1RBxMK2B2CalYJgF8C9tYMszWkRESZ",
    "source": "meteora",
    "name": "9n4n-EPjF Meteora LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:45.735Z",
    "updatedAt": "2025-04-29T13:34:45.735Z",
    "metrics": {
      "id": 263,
      "poolId": 3,
      "timestamp": "2025-04-30T02:17:27.019Z",
      "tvl": 4000000,
      "fee": 0.003,
      "apy24h": 11.570175,
      "apy7d": 10.644561,
      "apy30d": 9.834649,
      "volumeUsd": 400000,
      "extraData": {
        "bins": "120",
        "concentrationBounds": "medium"
      }
    },
    "tokens": []
  },
  {
    "id": 4,
    "poolId": "7UF3m8hDGZ6bNnHzaT2YHrhp7A7n9qFfBj6QEpHPv5S8",
    "programId": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    "source": "raydium",
    "name": "4k3D-EPjF LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:45.768Z",
    "updatedAt": "2025-04-29T13:34:45.768Z",
    "metrics": {
      "id": 264,
      "poolId": 4,
      "timestamp": "2025-04-30T02:17:27.021Z",
      "tvl": 5000000,
      "fee": 0.0025,
      "apy24h": 14.665538,
      "apy7d": 13.198984,
      "apy30d": 12.465708,
      "volumeUsd": 750000,
      "extraData": {
        "ammId": "7UF3m8hDGZ6bNnHzaT2YHrhp7A7n9qFfBj6QEpHPv5S8"
      }
    },
    "tokens": []
  },
  {
    "id": 5,
    "poolId": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
    "programId": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    "source": "raydium",
    "name": "So11-EPjF LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:45.801Z",
    "updatedAt": "2025-04-29T13:34:45.801Z",
    "metrics": {
      "id": 265,
      "poolId": 5,
      "timestamp": "2025-04-30T02:17:27.022Z",
      "tvl": 6500000,
      "fee": 0.0025,
      "apy24h": 14.665538,
      "apy7d": 13.198984,
      "apy30d": 12.465708,
      "volumeUsd": 975000,
      "extraData": {
        "ammId": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2"
      }
    },
    "tokens": []
  },
  {
    "id": 6,
    "poolId": "6UmmUiYoBjSrhakAobJw8BvkmJtDVxaeBtbt7rxWo1mg",
    "programId": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    "source": "raydium",
    "name": "So11-Es9v LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:45.968Z",
    "updatedAt": "2025-04-29T13:34:45.968Z",
    "metrics": {
      "id": 267,
      "poolId": 6,
      "timestamp": "2025-04-30T02:17:27.037Z",
      "tvl": 8000000,
      "fee": 0.0025,
      "apy24h": 14.665538,
      "apy7d": 13.198984,
      "apy30d": 12.465708,
      "volumeUsd": 1200000,
      "extraData": {
        "ammId": "6UmmUiYoBjSrhakAobJw8BvkmJtDVxaeBtbt7rxWo1mg"
      }
    },
    "tokens": []
  },
  {
    "id": 7,
    "poolId": "AVs9TA4nWDzfPJE9gGVNJMVhcQy3V9PGazuz33BfG2RA",
    "programId": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    "source": "raydium",
    "name": "4k3D-So11 LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:46.004Z",
    "updatedAt": "2025-04-29T13:34:46.004Z",
    "metrics": {
      "id": 266,
      "poolId": 7,
      "timestamp": "2025-04-30T02:17:27.036Z",
      "tvl": 9500000,
      "fee": 0.0025,
      "apy24h": 14.665538,
      "apy7d": 13.198984,
      "apy30d": 12.465708,
      "volumeUsd": 1425000,
      "extraData": {
        "ammId": "AVs9TA4nWDzfPJE9gGVNJMVhcQy3V9PGazuz33BfG2RA"
      }
    },
    "tokens": []
  },
  {
    "id": 8,
    "poolId": "DVa7Qmb5ct9RCpaU7UTpSaf3GVMYz17vNVU67XpdCRut",
    "programId": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    "source": "raydium",
    "name": "9n4n-EPjF LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:46.040Z",
    "updatedAt": "2025-04-29T13:34:46.040Z",
    "metrics": {
      "id": 268,
      "poolId": 8,
      "timestamp": "2025-04-30T02:17:27.043Z",
      "tvl": 11000000,
      "fee": 0.0025,
      "apy24h": 14.665538,
      "apy7d": 13.198984,
      "apy30d": 12.465708,
      "volumeUsd": 1650000,
      "extraData": {
        "ammId": "DVa7Qmb5ct9RCpaU7UTpSaf3GVMYz17vNVU67XpdCRut"
      }
    },
    "tokens": []
  },
  {
    "id": 9,
    "poolId": "EoTcMgcDRTJVZDMZWBoU6rhYHZfkNTVEAfz3uUJRcYGj",
    "programId": "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",
    "source": "orca",
    "name": "7vfC-EPjF Orca LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:46.475Z",
    "updatedAt": "2025-04-29T13:34:46.475Z",
    "metrics": {
      "id": 270,
      "poolId": 9,
      "timestamp": "2025-04-30T02:17:27.234Z",
      "tvl": 5000000,
      "fee": 0.0005,
      "apy24h": 2.2140894,
      "apy7d": 2.103385,
      "apy30d": 1.9926804,
      "volumeUsd": 600000,
      "extraData": {
        "tickSpacing": "64",
        "whirlpoolId": "EoTcMgcDRTJVZDMZWBoU6rhYHZfkNTVEAfz3uUJRcYGj"
      }
    },
    "tokens": []
  },
  {
    "id": 10,
    "poolId": "HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ",
    "programId": "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",
    "source": "orca",
    "name": "So11-EPjF Orca LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:46.480Z",
    "updatedAt": "2025-04-29T13:34:46.480Z",
    "metrics": {
      "id": 269,
      "poolId": 10,
      "timestamp": "2025-04-30T02:17:27.234Z",
      "tvl": 3000000,
      "fee": 0.0001,
      "apy24h": 0.438958,
      "apy7d": 0.41701007,
      "apy30d": 0.39506218,
      "volumeUsd": 360000,
      "extraData": {
        "tickSpacing": "64",
        "whirlpoolId": "HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ"
      }
    },
    "tokens": []
  },
  {
    "id": 11,
    "poolId": "7qbRF6YsyGuLUVs6Y1q64bdVrfe4ZcUUz1JRdoVNUJnm",
    "programId": "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",
    "source": "orca",
    "name": "So11-Es9v Orca LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:46.500Z",
    "updatedAt": "2025-04-29T13:34:46.500Z",
    "metrics": {
      "id": 271,
      "poolId": 11,
      "timestamp": "2025-04-30T02:17:27.237Z",
      "tvl": 7000000,
      "fee": 0.001,
      "apy24h": 4.477063,
      "apy7d": 4.25321,
      "apy30d": 4.029357,
      "volumeUsd": 840000,
      "extraData": {
        "tickSpacing": "64",
        "whirlpoolId": "7qbRF6YsyGuLUVs6Y1q64bdVrfe4ZcUUz1JRdoVNUJnm"
      }
    },
    "tokens": []
  },
  {
    "id": 12,
    "poolId": "7FHzquT9yCMzD5hVQm8PzD46LuAnRZXGBv91o3YJ3xxy",
    "programId": "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",
    "source": "orca",
    "name": "9n4n-EPjF Orca LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:46.513Z",
    "updatedAt": "2025-04-29T13:34:46.513Z",
    "metrics": {
      "id": 272,
      "poolId": 12,
      "timestamp": "2025-04-30T02:17:27.241Z",
      "tvl": 9000000,
      "fee": 0.002,
      "apy24h": 9.153995,
      "apy7d": 8.696295,
      "apy30d": 8.238595,
      "volumeUsd": 1080000,
      "extraData": {
        "tickSpacing": "64",
        "whirlpoolId": "7FHzquT9yCMzD5hVQm8PzD46LuAnRZXGBv91o3YJ3xxy"
      }
    },
    "tokens": []
  },
  {
    "id": 13,
    "poolId": "3kTA1SQ9gL7oXy7pMqJ3MtU28cQQjPCYjhwBpRKVLVJw",
    "programId": "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",
    "source": "orca",
    "name": "DezX-So11 Orca LP",
    "active": true,
    "createdAt": "2025-04-29T13:34:46.520Z",
    "updatedAt": "2025-04-29T13:34:46.520Z",
    "metrics": {
      "id": 273,
      "poolId": 13,
      "timestamp": "2025-04-30T02:17:27.250Z",
      "tvl": 11000000,
      "fee": 0.003,
      "apy24h": 14.039689,
      "apy7d": 13.337705,
      "apy30d": 12.63572,
      "volumeUsd": 1320000,
      "extraData": {
        "tickSpacing": "64",
        "whirlpoolId": "3kTA1SQ9gL7oXy7pMqJ3MtU28cQQjPCYjhwBpRKVLVJw"
      }
    },
    "tokens": []
  }
]
```

### Pool 1: mSoL-EPjF Meteora LP

- ID: 1

- Pool ID: 8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj

- Source: meteora

- Token count: 0

- No tokens found in response

### Pool 2: So11-EPjF Meteora LP

- ID: 2

- Pool ID: FLKdMRsS7dvj9mMsEBxciUJc3TKJcEmVUbQRf5UnYEpw

- Source: meteora

- Token count: 0

- No tokens found in response

### Pool 3: 9n4n-EPjF Meteora LP

- ID: 3

- Pool ID: Cv3YJJvCZQJKP5G6AHh77GVbewGZUVs4Y1WJ6Yc3HbQj

- Source: meteora

- Token count: 0

- No tokens found in response

### Pool 4: 4k3D-EPjF LP

- ID: 4

- Pool ID: 7UF3m8hDGZ6bNnHzaT2YHrhp7A7n9qFfBj6QEpHPv5S8

- Source: raydium

- Token count: 0

- No tokens found in response

### Pool 5: So11-EPjF LP

- ID: 5

- Pool ID: 58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2

- Source: raydium

- Token count: 0

- No tokens found in response

### Pool 6: So11-Es9v LP

- ID: 6

- Pool ID: 6UmmUiYoBjSrhakAobJw8BvkmJtDVxaeBtbt7rxWo1mg

- Source: raydium

- Token count: 0

- No tokens found in response

### Pool 7: 4k3D-So11 LP

- ID: 7

- Pool ID: AVs9TA4nWDzfPJE9gGVNJMVhcQy3V9PGazuz33BfG2RA

- Source: raydium

- Token count: 0

- No tokens found in response

### Pool 8: 9n4n-EPjF LP

- ID: 8

- Pool ID: DVa7Qmb5ct9RCpaU7UTpSaf3GVMYz17vNVU67XpdCRut

- Source: raydium

- Token count: 0

- No tokens found in response

### Pool 9: 7vfC-EPjF Orca LP

- ID: 9

- Pool ID: EoTcMgcDRTJVZDMZWBoU6rhYHZfkNTVEAfz3uUJRcYGj

- Source: orca

- Token count: 0

- No tokens found in response

### Pool 10: So11-EPjF Orca LP

- ID: 10

- Pool ID: HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ

- Source: orca

- Token count: 0

- No tokens found in response

### Pool 11: So11-Es9v Orca LP

- ID: 11

- Pool ID: 7qbRF6YsyGuLUVs6Y1q64bdVrfe4ZcUUz1JRdoVNUJnm

- Source: orca

- Token count: 0

- No tokens found in response

### Pool 12: 9n4n-EPjF Orca LP

- ID: 12

- Pool ID: 7FHzquT9yCMzD5hVQm8PzD46LuAnRZXGBv91o3YJ3xxy

- Source: orca

- Token count: 0

- No tokens found in response

### Pool 13: DezX-So11 Orca LP

- ID: 13

- Pool ID: 3kTA1SQ9gL7oXy7pMqJ3MtU28cQQjPCYjhwBpRKVLVJw

- Source: orca

- Token count: 0

- No tokens found in response

## Test 3: Get Specific Pool


Fetching pool with ID: 8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj

```json
{
  "id": 1,
  "poolId": "8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj",
  "programId": "LbVRzDTJEz6PA1RBxMK2B2CalYJgF8C9tYMszWkRESZ",
  "source": "meteora",
  "name": "mSoL-EPjF Meteora LP",
  "active": true,
  "createdAt": "2025-04-29T13:34:45.530Z",
  "updatedAt": "2025-04-29T13:34:45.530Z",
  "metrics": {
    "id": 261,
    "poolId": 1,
    "timestamp": "2025-04-30T02:17:27.005Z",
    "tvl": 2000000,
    "fee": 0.003,
    "apy24h": 11.570175,
    "apy7d": 10.644561,
    "apy30d": 9.834649,
    "volumeUsd": 200000,
    "extraData": {
      "bins": "120",
      "concentrationBounds": "medium"
    }
  },
  "tokens": []
}
```

## Test 4: Get Pool with Hard-coded ID


Fetching pool with hard-coded ID: 7UF3m8hDGZ6bNnHzaT2YHrhp7A7n9qFfBj6QEpHPv5S8

```json
{
  "id": 4,
  "poolId": "7UF3m8hDGZ6bNnHzaT2YHrhp7A7n9qFfBj6QEpHPv5S8",
  "programId": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
  "source": "raydium",
  "name": "4k3D-EPjF LP",
  "active": true,
  "createdAt": "2025-04-29T13:34:45.768Z",
  "updatedAt": "2025-04-29T13:34:45.768Z",
  "metrics": {
    "id": 264,
    "poolId": 4,
    "timestamp": "2025-04-30T02:17:27.021Z",
    "tvl": 5000000,
    "fee": 0.0025,
    "apy24h": 14.665538,
    "apy7d": 13.198984,
    "apy30d": 12.465708,
    "volumeUsd": 750000,
    "extraData": {
      "ammId": "7UF3m8hDGZ6bNnHzaT2YHrhp7A7n9qFfBj6QEpHPv5S8"
    }
  },
  "tokens": []
}
```

