# SolPool Insight: Advanced Solana Liquidity Pool Analytics Platform

![SolPool Insight Logo](./generated-icon.png)

## Overview

SolPool Insight is a cutting-edge analytics platform designed to provide comprehensive data, powerful predictions, and actionable insights on Solana liquidity pools. By leveraging advanced machine learning and blockchain data integration, SolPool Insight helps crypto investors make better decisions regarding liquidity pool investments.

### Key Features

- **Comprehensive Pool Coverage**: Access data from thousands of liquidity pools across all major DEXes, including Raydium, Orca, Jupiter, Meteora, Saber, and more.
- **Advanced Prediction Technology**: Self-evolving machine learning models predict which pools will increase in TVL or APR.
- **Real-time Metrics**: Track APR, TVL, volume, and historical trends (24h, 7d, 30d).
- **Meme Coin Analytics**: Specialized tracking of meme token pools like BONK, SAMO, DOGWIFHAT, and more.
- **Mobile-Friendly Design**: Access all features from any device with a responsive design.
- **Robust API**: Integrate SolPool data into your own applications with our documented REST API.

## Technology Stack

- **Frontend**: Streamlit for interactive dashboards and data visualization
- **Data Processing**: Python data science stack (Pandas, NumPy, Scikit-learn)
- **Machine Learning**: TensorFlow, XGBoost, self-evolving ensemble models
- **Blockchain Integration**: Direct Solana on-chain data extraction
- **Database**: PostgreSQL for historical data storage
- **API**: RESTful API with Flask

## Prediction Technology

Our platform implements a comprehensive prediction system that leverages continuous data collection and advanced machine learning:

### Data Collection & Processing

- **Hourly Data Collection**: Automated collection of pool metrics (APR, TVL, volume) every hour
- **Token Price Updates**: Regular updates of token prices every 30 minutes
- **Historical Database**: Complete time-series data storage in PostgreSQL
- **Feature Engineering**: Over 40 derived features from raw metrics

### Multi-Model Architecture

Our prediction system uses multiple specialized models working together:

1. **APR Prediction Model**: Random Forest Regressor/XGBoost predicts future APR changes
2. **Pool Performance Classifier**: Categorizes pools into performance classes (Excellent, Good, Average, Poor)
3. **Risk Assessment Model**: Quantifies risk levels through multi-factor analysis

### Advanced ML Techniques

1. **Bayesian Optimization**: Automatically fine-tunes hyperparameters for optimal performance
2. **Time-Series Analysis**: Specialized algorithms for temporal pattern recognition
3. **Reinforcement Learning**: Optimizes prediction weights through continuous validation
4. **Ensemble Methods**: Combines multiple prediction signals for greater accuracy

### Continuous Improvement

- **Daily Model Training**: Models automatically retrain daily with fresh data
- **Prediction Validation**: Historical predictions are validated against actual outcomes
- **Adaptive Weighting**: Self-adjusting weight allocation to the most accurate model components

For more detailed information about our prediction technology, please see our [Prediction Technology Documentation](./prediction_technology.md).

## Getting Started

### Prerequisites

- Python 3.11+
- Required Python packages (see pyproject.toml)
- Solana RPC endpoint (optional but recommended for live data)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/solpool-insight.git
   cd solpool-insight
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with the following variables:
   ```
   SOLANA_RPC_ENDPOINT=https://your-rpc-endpoint.com
   DATABASE_URL=postgresql://username:password@host:port/database
   ```

4. Run the application:
   ```bash
   streamlit run minimal_pool_data.py
   ```

## Project Structure

```
├── .streamlit/              # Streamlit configuration
├── data_ingestion/          # Data collection modules
├── database/                # Database schemas and operations
├── docs/                    # Documentation
├── eda/                     # Exploratory data analysis
├── models/                  # Saved ML models
├── pages/                   # Streamlit multi-page app
├── utils/                   # Utility functions
├── advanced_filtering.py    # Advanced pool filtering capabilities
├── advanced_prediction_engine.py # Self-evolving prediction system
├── api_documentation.md     # API documentation
├── api_server.py            # API implementation
├── minimal_pool_data.py     # Main Streamlit application
├── onchain_extractor.py     # On-chain data extraction
├── pool_retrieval_system.py # Pool data retrieval system
```

## Usage

### Web Interface

The web interface is divided into several tabs:

1. **Overview**: Summary metrics and pool distribution statistics
2. **Pool Explorer**: Comprehensive pool search and filtering
3. **Insights & Predictions**: Machine learning-based predictions and trend analysis

### API

SolPool Insight provides a RESTful API for data access. See [API Documentation](./api_documentation.md) for details.

Example API request:
```bash
curl -X GET "https://api.solpool-insight.com/pools/top?limit=10&sort=apr"
```

## Mobile-Friendly Design

SolPool Insight is designed to work seamlessly on mobile devices:

- Responsive layout that adapts to screen size
- Optimized data tables for small screens
- Touch-friendly UI elements
- Simplified navigation on mobile devices

## Future Development

- **Enhanced Social Signals**: Integration of social media sentiment analysis
- **Portfolio Optimization**: Smart recommendations for optimal pool allocation
- **Advanced Risk Metrics**: Deeper risk analysis and volatility prediction
- **Customizable Alerts**: Notifications for significant pool changes
- **Additional Blockchains**: Expanding coverage to other DeFi ecosystems

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, suggestions, or support, please contact us at:
- Email: support@solpool-insight.com
- Twitter: [@SolPoolInsight](https://twitter.com/SolPoolInsight)
- Discord: [SolPool Insight Community](https://discord.gg/solpoolinsight)