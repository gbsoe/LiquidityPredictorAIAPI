# Helius API Reference Guide

This document outlines the Solana RPC methods available through the Helius API endpoint, providing a reference for future use in our application.

## Supported RPC Methods

### Account and Program Information
- `getAccountInfo` - Get details about a specific account
- `getProgramAccounts` - Get accounts owned by a specific program (limited use - may time out for large programs)

### Token Information
- `getTokenAccountBalance` - Get the balance of a specific token account
- `getTokenLargestAccounts` - Get the largest accounts holding a specific token
- `getTokenSupply` - Get the total supply of a specific token

### Block and Transaction Information
- `getBlock` - Get a specific block
- `getBlocks` - Get blocks in a range
- `getBlockTime` - Get the estimated time for a block
- `getTransaction` - Get transaction details by transaction signature
- `getSignaturesForAddress` - Get transaction signatures for a specific address
- `getTransactionCount` - Get the total transaction count

### Network Status
- `getVersion` - Get the current Solana version
- `getSlot` - Get the current slot
- `getSlotLeader` - Get the current slot leader
- `getLatestBlockhash` - Get the latest blockhash
- `getHealth` - Get the health status of the node
- `getBalance` - Get the SOL balance of an account
- `getBlockHeight` - Get the current block height
- `isBlockhashValid` - Check if a blockhash is still valid
- `getGenesisHash` - Get the genesis hash
- `getFeeForMessage` - Calculate the fee for a message
- `getFirstAvailableBlock` - Get the first available block
- `getIdentity` - Get the identity pubkey of the node

### Cluster Information
- `getClusterNodes` - Get the list of nodes in the cluster
- `getEpochInfo` - Get information about the current epoch 
- `getEpochSchedule` - Get epoch schedule information
- `getRecentPerformanceSamples` - Get recent performance samples
- `minimumLedgerSlot` - Get the minimum ledger slot
- `getVoteAccounts` - Get information about vote accounts

### Stake and Inflation
- `getStakeActivation` - Get stake activation information
- `getStakeMinimumDelegation` - Get minimum stake delegation
- `getInflationGovernor` - Get inflation governor configuration
- `getInflationRate` - Get current inflation rate
- `getSupply` - Get information about circulating supply

### Special Methods
- `getMaxRetransmitSlot` - Get the maximum retransmit slot
- `getMaxShredInsertSlot` - Get the maximum shred insert slot
- `getMinimumBalanceForRentExemption` - Calculate minimum balance for rent exemption
- `getHighestSnapshotSlot` - Get the highest available snapshot slot
- `getBlockProduction` - Get block production information
- `getBlockCommitment` - Get the commitment for a block
- `getLargestAccounts` - Get the largest accounts in the network

### Asset-Related Methods
- `getAsset` - Get details about a specific asset
- `getAssetProof` - Get proof for a specific asset
- `getAssetsByGroup` - Get assets by group
- `getAssetsByCreator` - Get assets created by a specific creator
- `getAssetsByOwner` - Get assets owned by a specific address
- `getAssetsByAuthority` - Get assets by authority
- `getSignaturesForAsset` - Get transaction signatures for a specific asset

## Usage Considerations

1. The `getProgramAccounts` method may time out for large programs. Limit the response with filters and limits.
2. Use `getAccountInfo` for specific pool accounts when `getProgramAccounts` times out.
3. Some methods may have usage limits or restrictions based on your Helius API plan.
4. For best performance, cache results when appropriate to minimize API calls.

## Example Usage

```python
# Example of fetching token information
def get_token_info(token_address):
    response = make_rpc_request(
        "getAccountInfo",
        [token_address, {"encoding": "jsonParsed"}]
    )
    return response.get("result", {})

# Example of fetching a known liquidity pool
def get_liquidity_pool(pool_address):
    response = make_rpc_request(
        "getAccountInfo",
        [pool_address, {"encoding": "base64"}]
    )
    return response.get("result", {})
```

## Alternative Approaches for Liquidity Pools

Since `getProgramAccounts` may be limited, consider:

1. Using a predefined list of known pool addresses
2. Using `getAccountInfo` to fetch specific pools directly
3. Implementing pagination or time-delayed batch requests
4. Using token account queries to identify potential pool candidates