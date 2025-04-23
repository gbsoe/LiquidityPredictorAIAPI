const { Connection, PublicKey, LAMPORTS_PER_SOL } = require('@solana/web3.js');
const axios = require('axios');

// Solana connection
const connection = new Connection(
    process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com',
    'confirmed'
);

/**
 * Gets current Solana blockchain statistics
 * @returns {Promise<Object>} Blockchain statistics
 */
async function getBlockchainStats() {
    try {
        // Fetch current block height
        const slot = await connection.getSlot();
        
        // Fetch transaction count for recent finalized block
        const blockHeight = await connection.getBlockHeight();
        
        // Get recent performance samples
        const performanceSamples = await connection.getRecentPerformanceSamples(10);
        
        // Calculate average TPS
        const avgTps = performanceSamples.length > 0 
            ? performanceSamples.reduce((acc, sample) => acc + sample.numTransactions / sample.samplePeriodSecs, 0) / performanceSamples.length 
            : 0;
        
        // Get SOL price if possible
        let solPrice = null;
        try {
            const response = await axios.get('https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd');
            solPrice = response.data?.solana?.usd || null;
        } catch (priceError) {
            console.warn('Could not fetch SOL price:', priceError.message);
        }
        
        return {
            slot,
            blockHeight,
            currentTimestamp: new Date().toISOString(),
            averageTps: avgTps,
            solPrice,
            cluster: process.env.SOLANA_CLUSTER || 'mainnet-beta'
        };
    } catch (error) {
        console.error('Error fetching blockchain stats:', error);
        throw new Error(`Failed to fetch blockchain stats: ${error.message}`);
    }
}

/**
 * Fetches information about a specific Solana token
 * @param {string} mintAddress - Token mint address
 * @returns {Promise<Object>} Token information
 */
async function getTokenInfo(mintAddress) {
    try {
        // Validate mint address
        let publicKey;
        try {
            publicKey = new PublicKey(mintAddress);
        } catch (error) {
            throw new Error('Invalid mint address');
        }
        
        // In a full implementation, you'd use the Solana Token Registry or another service
        // to get comprehensive token information. This is a simplified version.
        
        // Fetch basic token account data
        const accountInfo = await connection.getAccountInfo(publicKey);
        if (!accountInfo) {
            throw new Error('Token account not found');
        }
        
        // Try to get token metadata if available
        // This is a simplified approach - in production you'd use proper token metadata programs
        let metadata = null;
        try {
            const metadataPDA = await getTokenMetadataPDA(publicKey);
            const metadataInfo = await connection.getAccountInfo(metadataPDA);
            if (metadataInfo) {
                metadata = parseTokenMetadata(metadataInfo.data);
            }
        } catch (metadataError) {
            console.warn('Could not fetch token metadata:', metadataError.message);
        }
        
        return {
            mintAddress: mintAddress,
            name: metadata?.name || 'Unknown Token',
            symbol: metadata?.symbol || 'UNKNOWN',
            decimals: metadata?.decimals || 0,
            logoURI: metadata?.image || null,
            lastUpdated: new Date().toISOString()
        };
    } catch (error) {
        console.error(`Error fetching token info for ${mintAddress}:`, error);
        throw new Error(`Failed to fetch token info: ${error.message}`);
    }
}

// Helper functions (simplified for this implementation)
async function getTokenMetadataPDA(mintAddress) {
    // This is a placeholder - in production you'd derive the proper PDA
    return new PublicKey(mintAddress);
}

function parseTokenMetadata(data) {
    // This is a placeholder - in production you'd properly decode the metadata
    return { name: 'Unknown', symbol: 'UNKNOWN', decimals: 0 };
}

module.exports = {
    getBlockchainStats,
    getTokenInfo
};
