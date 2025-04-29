# DeFi API Data Test Results

Data retrieved at: 2025-04-29 00:36:11

## Summary
Successfully retrieved data for 19 liquidity pools from the DeFi API.

## Sample Pool Data

### Pool 1: RAY/USDC

#### Basic Pool Information
- **ID**: `RAYUSDC`
- **Name**: RAY/USDC
- **DEX**: raydium
- **Token 1**: RAY (`4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R`)
- **Token 2**: USDC (`EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v`)
- **Category**: DeFi
- **Version**: v1

#### Financial Metrics
- **Liquidity**: $5,234,789.50
- **24h Volume**: $1,245,678.90
- **APR**: 12.45%
- **Fee**: 0.25%

#### Historical Trend Data
- **APR Change (24h)**: 4.89%
- **APR Change (7d)**: 8.70%
- **APR Change (30d)**: 0.00%
- **TVL Change (24h)**: 0.00%
- **TVL Change (7d)**: 0.00%
- **TVL Change (30d)**: 0.00%

#### Prediction Data
- **Prediction Score**: 72.22

### Pool 2: SOL/USDC

#### Basic Pool Information
- **ID**: `SOLUSDC`
- **Name**: SOL/USDC
- **DEX**: raydium
- **Token 1**: SOL (`So11111111111111111111111111111111111111112`)
- **Token 2**: USDC (`EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v`)
- **Category**: Major
- **Version**: v1

#### Financial Metrics
- **Liquidity**: $15,789,456.00
- **24h Volume**: $3,456,789.00
- **APR**: 8.32%
- **Fee**: 0.25%

#### Historical Trend Data
- **APR Change (24h)**: 4.65%
- **APR Change (7d)**: 6.71%
- **APR Change (30d)**: 0.00%
- **TVL Change (24h)**: 0.00%
- **TVL Change (7d)**: 0.00%
- **TVL Change (30d)**: 0.00%

#### Prediction Data
- **Prediction Score**: 68.16

### Pool 3: mSOL/USDC

#### Basic Pool Information
- **ID**: `MSOLUSDCM`
- **Name**: mSOL/USDC
- **DEX**: meteora
- **Token 1**: mSOL (`mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So`)
- **Token 2**: USDC (`EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v`)
- **Category**: Other
- **Version**: v1

#### Financial Metrics
- **Liquidity**: $3,456,789.00
- **24h Volume**: $987,654.30
- **APR**: 14.75%
- **Fee**: 0.30%

#### Historical Trend Data
- **APR Change (24h)**: 5.96%
- **APR Change (7d)**: 8.24%
- **APR Change (30d)**: 0.00%
- **TVL Change (24h)**: 0.00%
- **TVL Change (7d)**: 0.00%
- **TVL Change (30d)**: 0.00%

#### Prediction Data
- **Prediction Score**: 68.38

## All Retrieved Pools

| Name | DEX | Liquidity | APR | Volume 24h | Prediction Score |
|------|-----|-----------|-----|------------|------------------|
| RAY/USDC | raydium | $5,234,789.50 | 12.45% | $1,245,678.90 | 72.22 |
| SOL/USDC | raydium | $15,789,456.00 | 8.32% | $3,456,789.00 | 68.16 |
| mSOL/USDC | meteora | $3,456,789.00 | 14.75% | $987,654.30 | 68.38 |
| BTC/USDC | meteora | $7,896,543.00 | 9.54% | $2,345,679.00 | 68.77 |
| SOL/USDC | orca | $8,765,432.00 | 11.23% | $4,567,890.00 | 71.62 |
| ETH/USDC | orca | $6,543,211.00 | 10.45% | $3,456,789.00 | 71.22 |
| SOL/USDC | orca | $3,000,000.00 | 0.44% | $360,000.00 | 61.22 |
| ETH/USDC | orca | $5,000,000.00 | 2.21% | $600,000.00 | 62.11 |
| RAY/USDC | raydium | $5,000,000.00 | 14.67% | $750,000.00 | 70.33 |
| mSOL/USDC | meteora | $2,000,000.00 | 11.57% | $200,000.00 | 63.79 |
| BTC/USDC | meteora | $3,000,000.00 | 11.57% | $300,000.00 | 66.79 |
| SOL/USDC | raydium | $6,500,000.00 | 14.67% | $975,000.00 | 68.33 |
| / | meteora | $4,000,000.00 | 11.57% | $400,000.00 | 63.79 |
| / | orca | $7,000,000.00 | 4.48% | $840,000.00 | 60.24 |
| / | orca | $9,000,000.00 | 9.15% | $1,080,000.00 | 62.58 |
| / | orca | $11,000,000.00 | 14.04% | $1,320,000.00 | 65.02 |
| / | raydium | $8,000,000.00 | 14.67% | $1,200,000.00 | 65.33 |
| / | raydium | $9,500,000.00 | 14.67% | $1,425,000.00 | 65.33 |
| / | raydium | $11,000,000.00 | 14.67% | $1,650,000.00 | 65.33 |


## Data Fields for Prediction Models

The following data fields were successfully retrieved for use in prediction models:

### Basic Pool Information
- `id`
- `name`
- `dex`
- `token1_symbol`
- `token2_symbol`
- `token1_address`
- `token2_address`

### Financial Metrics
- `liquidity`
- `volume_24h`
- `apr`
- `fee`

### Historical Trend Data
- `apr_change_24h`
- `apr_change_7d`
- `apr_change_30d`
- `tvl_change_24h`
- `tvl_change_7d`
- `tvl_change_30d`

### Categorization
- `category`
- `version`

### Time-based Data
- `created_at`
- `updated_at`

### Prediction Results
- `prediction_score`

