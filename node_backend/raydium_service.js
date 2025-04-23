const { Connection, PublicKey } = require('@solana/web3.js');
const axios = require('axios');

// Solana connection
const connection = new Connection(
    process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com',
    'confirmed'
);

// Raydium API endpoints
const RAYDIUM_API_BASE = 'https://api.raydium.io/v2';

/**
 * Fetches all liquidity pools from Raydium
 * @returns {Promise<Array>} Array of pool data
 */
async function getAllPools() {
    try {
        const response = await axios.get(`${RAYDIUM_API_BASE}/main/pairs`);
        return response.data || [];
    } catch (error) {
        console.error('Error fetching Raydium pools:', error);
        throw new Error(`Failed to fetch pools: ${error.message}`);
    }
}

/**
 * Fetches specific pool data by ID
 * @param {string} poolId - Pool ID
 * @returns {Promise<Object>} Pool data object
 */
async function getPoolById(poolId) {
    try {
        // Try to locate the pool in the complete list
        const allPools = await getAllPools();
        const pool = allPools.find(p => p.ammId === poolId || p.id === poolId);
        
        if (!pool) {
            throw new Error('Pool not found');
        }
        
        // Fetch additional pool details if available
        try {
            const poolDetail = await axios.get(`${RAYDIUM_API_BASE}/pool/${poolId}`);
            return { ...pool, ...poolDetail.data };
        } catch (detailError) {
            // Return basic pool info if detailed info is unavailable
            console.warn(`Could not fetch detailed info for pool ${poolId}:`, detailError.message);
            return pool;
        }
    } catch (error) {
        console.error(`Error fetching pool ${poolId}:`, error);
        throw new Error(`Failed to fetch pool: ${error.message}`);
    }
}

/**
 * Fetches metrics for a specific pool
 * @param {string} poolId - Pool ID
 * @returns {Promise<Object>} Pool metrics
 */
async function getPoolMetrics(poolId) {
    try {
        const pool = await getPoolById(poolId);
        if (!pool) {
            throw new Error('Pool not found');
        }
        
        // Get historical data if available
        let historicalData = [];
        try {
            const history = await axios.get(`${RAYDIUM_API_BASE}/pool/${poolId}/history`);
            historicalData = history.data || [];
        } catch (historyError) {
            console.warn(`Could not fetch history for pool ${poolId}:`, historyError.message);
        }
        
        // Calculate additional metrics
        const volume24h = pool.volume24h || 0;
        const volumeChange24h = pool.volumeChange24h || 0;
        const liquidity = pool.liquidity || 0;
        const liquidityChange24h = pool.liquidityChange24h || 0;
        const apr = pool.apr || 0;
        
        // Get current token prices if available
        const tokenPrices = {};
        if (pool.token0?.symbol) {
            try {
                const token0Price = await fetchTokenPrice(pool.token0.symbol);
                tokenPrices[pool.token0.symbol] = token0Price;
            } catch (priceError) {
                console.warn(`Could not fetch price for ${pool.token0.symbol}:`, priceError.message);
            }
        }
        
        if (pool.token1?.symbol) {
            try {
                const token1Price = await fetchTokenPrice(pool.token1.symbol);
                tokenPrices[pool.token1.symbol] = token1Price;
            } catch (priceError) {
                console.warn(`Could not fetch price for ${pool.token1.symbol}:`, priceError.message);
            }
        }
        
        return {
            poolId,
            name: `${pool.token0?.symbol || 'Unknown'}/${pool.token1?.symbol || 'Unknown'}`,
            liquidity,
            liquidityChange24h,
            volume24h,
            volumeChange24h,
            apr,
            tokenPrices,
            historicalData,
            updatedAt: new Date().toISOString()
        };
    } catch (error) {
        console.error(`Error fetching metrics for pool ${poolId}:`, error);
        throw new Error(`Failed to fetch pool metrics: ${error.message}`);
    }
}

/**
 * Helper function to fetch token price
 * @param {string} symbol - Token symbol
 * @returns {Promise<number>} Token price in USD
 */
async function fetchTokenPrice(symbol) {
    try {
        // Try using CoinGecko API first (this would be replaced with proper implementation)
        // This is a stub - in production, you'd use an SDK or direct API call
        return 1.0; // Placeholder price
    } catch (error) {
        console.error(`Error fetching price for ${symbol}:`, error);
        return null;
    }
}

module.exports = {
    getAllPools,
    getPoolById,
    getPoolMetrics
};
