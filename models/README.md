# ML Models Directory

This directory stores trained machine learning models for the Solana Liquidity Pool Analysis System.

## Model Types

### APR Prediction Model
- **Filename**: `apr_prediction_model.pkl`
- **Algorithm**: Random Forest Regressor
- **Purpose**: Predicts future APR values for liquidity pools

### Pool Performance Classifier
- **Filename**: `pool_performance_classifier.pkl`
- **Algorithm**: XGBoost Classifier  
- **Purpose**: Classifies pools into performance categories (High, Medium, Low)

### Risk Assessment Model
- **Filename**: `risk_assessment_model.pkl`
- **Algorithm**: LSTM Neural Network
- **Purpose**: Evaluates risk scores for liquidity pools

## Model Training Schedule

Models are retrained on the following schedule:

- **APR Prediction Model**: Daily
- **Pool Performance Classifier**: Every 3 days
- **Risk Assessment Model**: Weekly

## Feature Importance

The key features used by these models include:

1. Historical APR volatility
2. Token price correlation
3. Liquidity depth
4. Volume trends
5. Blockchain network metrics
6. Pool age and stability metrics

## Model Performance

Performance metrics are stored in the database and can be viewed in the Predictions and Risk Assessment pages of the dashboard.