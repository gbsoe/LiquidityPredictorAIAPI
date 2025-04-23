const express = require('express');
const cors = require('cors');
const raydiumService = require('./raydium_service');
const solanaService = require('./solana_service');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.status(200).json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Raydium pool endpoints
app.get('/api/pools', async (req, res) => {
    try {
        const pools = await raydiumService.getAllPools();
        res.status(200).json(pools);
    } catch (error) {
        console.error('Error fetching pools:', error);
        res.status(500).json({ error: 'Failed to fetch pools', details: error.message });
    }
});

app.get('/api/pools/:poolId', async (req, res) => {
    try {
        const { poolId } = req.params;
        const poolData = await raydiumService.getPoolById(poolId);
        if (!poolData) {
            return res.status(404).json({ error: 'Pool not found' });
        }
        res.status(200).json(poolData);
    } catch (error) {
        console.error(`Error fetching pool ${req.params.poolId}:`, error);
        res.status(500).json({ error: 'Failed to fetch pool data', details: error.message });
    }
});

app.get('/api/pools/:poolId/metrics', async (req, res) => {
    try {
        const { poolId } = req.params;
        const metrics = await raydiumService.getPoolMetrics(poolId);
        if (!metrics) {
            return res.status(404).json({ error: 'Pool metrics not found' });
        }
        res.status(200).json(metrics);
    } catch (error) {
        console.error(`Error fetching pool metrics for ${req.params.poolId}:`, error);
        res.status(500).json({ error: 'Failed to fetch pool metrics', details: error.message });
    }
});

// Solana blockchain endpoints
app.get('/api/blockchain/stats', async (req, res) => {
    try {
        const stats = await solanaService.getBlockchainStats();
        res.status(200).json(stats);
    } catch (error) {
        console.error('Error fetching blockchain stats:', error);
        res.status(500).json({ error: 'Failed to fetch blockchain stats', details: error.message });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Unhandled error:', err);
    res.status(500).json({ 
        error: 'Internal server error', 
        message: err.message || 'An unexpected error occurred',
        timestamp: new Date().toISOString()
    });
});

// Start the server
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully');
    // Close any connections here
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('SIGINT received, shutting down gracefully');
    // Close any connections here
    process.exit(0);
});

module.exports = app;
