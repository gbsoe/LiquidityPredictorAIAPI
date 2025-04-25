"""
Advanced Prediction Engine for Solana Liquidity Pools

This module provides a comprehensive machine learning system that can predict future
performance of Solana liquidity pools and automatically evolve over time by:

1. Combining multiple ML models (ensemble approach)
2. Automatically retraining on new data
3. Adjusting to changing market conditions
4. Self-evaluating prediction accuracy
5. Optimizing feature selection and hyperparameters

The system uses a combination of:
- Gradient Boosting (XGBoost/LightGBM)
- LSTM Neural Networks
- Reinforcement Learning
- Bayesian Optimization
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import joblib
import random
from copy import deepcopy

# Import advanced self-evolving modules
try:
    from bayesian_optimization import BayesianOptimizer, HyperparameterOptimizer
    HAS_BAYESIAN_OPT = True
except ImportError:
    HAS_BAYESIAN_OPT = False

try:
    from multi_agent_system import MultiAgentSystem
    HAS_MULTI_AGENT = True
except ImportError:
    HAS_MULTI_AGENT = False

try:
    from neural_architecture_search import ArchitectureSearch, NeuralArchitecture
    HAS_NEURAL_SEARCH = True
except ImportError:
    HAS_NEURAL_SEARCH = False

# Advanced ML libraries (would need to be installed)
# Define a global HAS_TENSORFLOW variable
HAS_TENSORFLOW = False

# Try to import TensorFlow but handle it gracefully if it fails
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.utils import plot_model
    HAS_TENSORFLOW = True
except (ImportError, AttributeError, TypeError) as e:
    # Log the error but continue - we'll handle the absence of TensorFlow
    # in the appropriate places
    print(f"TensorFlow import error: {e}. Advanced neural network features will be disabled.")

try:
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectFromModel, RFE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("prediction_engine")

# Constants
MODEL_DIR = "models"
HISTORY_DIR = "history"
FEATURE_IMPORTANCE_THRESHOLD = 0.01
MIN_TRAINING_SAMPLES = 100
MAX_HISTORY_DAYS = 365
DEFAULT_PREDICTION_HORIZON = 7  # days

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

@dataclass
class ModelMetadata:
    """Metadata for tracking model performance and evolution"""
    model_type: str
    version: int
    created_at: datetime
    trained_on: int  # number of samples
    features: List[str]
    metrics: Dict[str, float]
    hyperparams: Dict[str, Any]
    feature_importance: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "trained_on": self.trained_on,
            "features": self.features,
            "metrics": self.metrics,
            "hyperparams": self.hyperparams,
            "feature_importance": self.feature_importance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        return cls(
            model_type=data["model_type"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            trained_on=data["trained_on"],
            features=data["features"],
            metrics=data["metrics"],
            hyperparams=data["hyperparams"],
            feature_importance=data["feature_importance"]
        )

@dataclass
class PredictionResult:
    """Results from a prediction"""
    pool_id: str
    prediction_date: datetime
    prediction_horizon: int  # days
    target_metrics: Dict[str, Any]  # what was predicted
    prediction_values: Dict[str, Any]  # predicted values
    confidence_intervals: Dict[str, Tuple[float, float]]  # lower/upper bounds
    prediction_score: float  # 0-100 score of confidence
    contributing_factors: List[str]  # key factors for prediction
    model_version: Dict[str, int]  # versions of models used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "pool_id": self.pool_id,
            "prediction_date": self.prediction_date.isoformat(),
            "prediction_horizon": self.prediction_horizon,
            "target_metrics": self.target_metrics,
            "prediction_values": self.prediction_values,
            "confidence_intervals": {k: list(v) for k, v in self.confidence_intervals.items()},
            "prediction_score": self.prediction_score,
            "contributing_factors": self.contributing_factors,
            "model_version": self.model_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """Create from dictionary"""
        return cls(
            pool_id=data["pool_id"],
            prediction_date=datetime.fromisoformat(data["prediction_date"]),
            prediction_horizon=data["prediction_horizon"],
            target_metrics=data["target_metrics"],
            prediction_values=data["prediction_values"],
            confidence_intervals={k: tuple(v) for k, v in data["confidence_intervals"].items()},
            prediction_score=data["prediction_score"],
            contributing_factors=data["contributing_factors"],
            model_version=data["model_version"]
        )

class FeatureEngineer:
    """
    Handles feature engineering for prediction models
    """
    
    def __init__(self):
        """Initialize the feature engineer"""
        self.scalers = {}
        self.feature_selectors = {}
        self.best_features = {}
    
    def create_features(self, pool_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create features from raw pool data
        
        Args:
            pool_data: List of pool data dictionaries with historical metrics
            
        Returns:
            DataFrame with engineered features
        """
        # Convert to DataFrame
        df = pd.DataFrame(pool_data)
        
        # Ensure we have the necessary columns
        required_columns = ['id', 'name', 'dex', 'category', 'liquidity', 'volume_24h', 'apr']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create categorical features (one-hot encoded)
        if 'dex' in df.columns:
            dex_dummies = pd.get_dummies(df['dex'], prefix='dex')
            df = pd.concat([df, dex_dummies], axis=1)
            
        if 'category' in df.columns:
            category_dummies = pd.get_dummies(df['category'], prefix='category')
            df = pd.concat([df, category_dummies], axis=1)
            
        # Create token-based features
        if 'token1_symbol' in df.columns and 'token2_symbol' in df.columns:
            # One-hot encode token symbols (this could get large with many tokens)
            # In a real implementation, we'd want to limit to most common tokens
            token1_dummies = pd.get_dummies(df['token1_symbol'], prefix='token1')
            token2_dummies = pd.get_dummies(df['token2_symbol'], prefix='token2')
            
            # Add flag for whether the pool contains a stablecoin
            stablecoins = ['USDC', 'USDT', 'DAI', 'BUSD', 'USDH', 'TUSD']
            df['has_stablecoin'] = (
                df['token1_symbol'].isin(stablecoins) | 
                df['token2_symbol'].isin(stablecoins)
            ).astype(int)
            
            # Add flag for whether the pool contains SOL
            df['has_sol'] = (
                (df['token1_symbol'] == 'SOL') | 
                (df['token2_symbol'] == 'SOL')
            ).astype(int)
            
            # Add flag for whether the pool contains a meme token
            meme_tokens = ['BONK', 'SAMO', 'DOGWIFHAT', 'FLOKI', 'POPCAT']
            df['has_meme_token'] = (
                df['token1_symbol'].isin(meme_tokens) | 
                df['token2_symbol'].isin(meme_tokens)
            ).astype(int)
            
            df = pd.concat([df, token1_dummies, token2_dummies], axis=1)
        
        # Create derived metrics
        if 'liquidity' in df.columns and 'volume_24h' in df.columns:
            # Volume to liquidity ratio (higher = more active pool)
            df['volume_liquidity_ratio'] = df['volume_24h'] / df['liquidity'].replace(0, np.nan)
            df['volume_liquidity_ratio'] = df['volume_liquidity_ratio'].fillna(0)
            
            # Logarithmic transformations for skewed data
            df['log_liquidity'] = np.log1p(df['liquidity'])
            df['log_volume_24h'] = np.log1p(df['volume_24h'])
        
        # Create features from historical data if available
        if 'apr_change_24h' in df.columns and 'apr_change_7d' in df.columns:
            # Moving average APR change
            df['avg_apr_change_rate'] = (df['apr_change_24h'] + df['apr_change_7d']) / 2
            
            # Acceleration of APR change
            df['apr_change_acceleration'] = df['apr_change_24h'] - (df['apr_change_7d'] / 7)
            
            # APR trend direction (1 = increasing, 0 = stable, -1 = decreasing)
            df['apr_trend'] = np.sign(df['apr_change_7d'])
            
        if 'tvl_change_24h' in df.columns and 'tvl_change_7d' in df.columns:
            # Moving average TVL change
            df['avg_tvl_change_rate'] = (df['tvl_change_24h'] + df['tvl_change_7d']) / 2
            
            # Acceleration of TVL change
            df['tvl_change_acceleration'] = df['tvl_change_24h'] - (df['tvl_change_7d'] / 7)
            
            # TVL trend direction
            df['tvl_trend'] = np.sign(df['tvl_change_7d'])
            
        # Calculate volatility if we have enough historical data
        if 'apr_change_24h' in df.columns and 'apr_change_7d' in df.columns and 'apr_change_30d' in df.columns:
            # Simple volatility measure based on standard deviation of changes
            df['apr_volatility'] = np.sqrt(
                df['apr_change_24h']**2 + 
                df['apr_change_7d']**2 + 
                df['apr_change_30d']**2
            ) / 3
            
        if 'tvl_change_24h' in df.columns and 'tvl_change_7d' in df.columns and 'tvl_change_30d' in df.columns:
            df['tvl_volatility'] = np.sqrt(
                df['tvl_change_24h']**2 + 
                df['tvl_change_7d']**2 + 
                df['tvl_change_30d']**2
            ) / 3
        
        # Fee-based features
        if 'fee' in df.columns:
            # Expected daily fee income
            df['daily_fee_income'] = df['fee'] * df['volume_24h']
            
            # Fee APR component (fees / liquidity * 365)
            df['fee_apr_component'] = (df['fee'] * df['volume_24h'] * 365) / df['liquidity'].replace(0, np.nan)
            df['fee_apr_component'] = df['fee_apr_component'].fillna(0)
        
        # Create protocol age/maturity feature if timestamps are available
        if 'created_at' in df.columns:
            # Convert string timestamps to datetime if needed
            if isinstance(df['created_at'].iloc[0], str):
                df['created_at'] = pd.to_datetime(df['created_at'])
                
            # Calculate age in days
            df['age_days'] = (datetime.now() - df['created_at']).dt.total_seconds() / (3600 * 24)
            
            # Nonlinear transformation of age (newer pools might behave differently)
            df['log_age'] = np.log1p(df['age_days'])
        
        # Drop non-feature columns that are not needed for modeling
        drop_columns = ['name', 'id', 'token1_symbol', 'token2_symbol', 
                        'token1_address', 'token2_address', 'created_at', 'updated_at']
        
        # Only drop columns that exist
        drop_columns = [col for col in drop_columns if col in df.columns]
        feature_df = df.drop(columns=drop_columns)
        
        # Replace any remaining NaN values
        feature_df = feature_df.fillna(0)
        
        return feature_df
    
    def scale_features(self, feature_df: pd.DataFrame, target: str = 'apr_prediction') -> pd.DataFrame:
        """
        Scale features for machine learning
        
        Args:
            feature_df: DataFrame with features
            target: Name of target column to exclude from scaling
            
        Returns:
            DataFrame with scaled features
        """
        if not HAS_SKLEARN:
            logger.warning("Scikit-learn not available, skipping feature scaling")
            return feature_df
        
        # Create a copy to avoid modifying the original
        scaled_df = feature_df.copy()
        
        # Separate numeric columns for scaling
        numeric_columns = scaled_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Remove target from scaling if it exists in the DataFrame
        if target in numeric_columns:
            numeric_columns.remove(target)
        
        # Create or reuse scaler
        if target not in self.scalers:
            self.scalers[target] = StandardScaler()
            scaled_values = self.scalers[target].fit_transform(scaled_df[numeric_columns])
        else:
            scaled_values = self.scalers[target].transform(scaled_df[numeric_columns])
        
        # Replace values in DataFrame
        scaled_df[numeric_columns] = scaled_values
        
        return scaled_df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, target: str = 'apr_prediction') -> pd.DataFrame:
        """
        Select most important features
        
        Args:
            X: Feature DataFrame
            y: Target Series
            target: Target name for tracking
            
        Returns:
            DataFrame with selected features
        """
        if not HAS_SKLEARN:
            logger.warning("Scikit-learn not available, skipping feature selection")
            return X
        
        # If we already have a feature selector for this target, use it
        if target in self.feature_selectors:
            selector = self.feature_selectors[target]
            return X[self.best_features[target]]
        
        # Use Random Forest to select important features
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = SelectFromModel(rf, threshold=FEATURE_IMPORTANCE_THRESHOLD)
        
        # Fit the selector
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store selector and features for future use
        self.feature_selectors[target] = selector
        self.best_features[target] = selected_features
        
        logger.info(f"Selected {len(selected_features)} features for {target}")
        
        return X[selected_features]
    
    def save_state(self, filepath: str = 'feature_engineer.pkl') -> None:
        """Save the state of the feature engineer"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'scalers': self.scalers,
                    'feature_selectors': self.feature_selectors,
                    'best_features': self.best_features
                }, f)
            logger.info(f"Saved feature engineer state to {filepath}")
        except Exception as e:
            logger.error(f"Error saving feature engineer state: {e}")
    
    def load_state(self, filepath: str = 'feature_engineer.pkl') -> bool:
        """Load the state of the feature engineer"""
        if not os.path.exists(filepath):
            logger.warning(f"No saved state found at {filepath}")
            return False
            
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                self.scalers = state['scalers']
                self.feature_selectors = state['feature_selectors']
                self.best_features = state['best_features']
            logger.info(f"Loaded feature engineer state from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading feature engineer state: {e}")
            return False

class APRPredictionModel:
    """
    Model specifically trained to predict future APR changes
    """
    
    def __init__(self, time_horizon: int = 7):
        """
        Initialize APR prediction model
        
        Args:
            time_horizon: Number of days into the future to predict
        """
        self.time_horizon = time_horizon
        self.model = None
        self.lstm_model = None
        self.feature_names = None
        self.metadata = None
        self.version = 1
        self.feature_engineer = FeatureEngineer()
        
    def _create_xgboost_model(self) -> xgb.XGBRegressor:
        """Create an XGBoost model for APR prediction"""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is required but not installed")
            
        # Model hyperparameters - in a real implementation these would be tuned
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42
        }
        
        return XGBRegressor(**params)
    
    def _create_lstm_model(self, input_shape: int) -> tf.keras.Model:
        """Create an LSTM model for time series prediction"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required but not installed")
            
        model = Sequential([
            LSTM(64, input_shape=(input_shape, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # Output layer for regression
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def prepare_training_data(self, historical_data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from historical pool data
        
        Args:
            historical_data: Historical pool data with known future outcomes
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Create features
        features_df = self.feature_engineer.create_features(historical_data)
        
        # The target is the future APR change
        if f'apr_change_{self.time_horizon}d' in df.columns:
            # We already have the future change in the data
            y = df[f'apr_change_{self.time_horizon}d']
        else:
            # We need to calculate the target from historical data
            # This would require time series of pool data
            # For the example, we'll use a placeholder
            y = df['apr_change_7d'] if 'apr_change_7d' in df.columns else df['apr'].diff()
            y = y.fillna(0)
        
        # Scale features
        X = self.feature_engineer.scale_features(features_df, 'apr_prediction')
        
        # Select important features
        if len(X) > MIN_TRAINING_SAMPLES:
            X = self.feature_engineer.select_features(X, y, 'apr_prediction')
        
        # Store feature names for inference
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, historical_data: List[Dict[str, Any]]) -> ModelMetadata:
        """
        Train APR prediction model on historical data
        
        Args:
            historical_data: Historical pool data with known future outcomes
            
        Returns:
            ModelMetadata with training info
        """
        # Prepare data
        X, y = self.prepare_training_data(historical_data)
        
        if len(X) < MIN_TRAINING_SAMPLES:
            logger.warning(f"Not enough training samples: {len(X)} < {MIN_TRAINING_SAMPLES}")
            
            # Create dummy metadata for tracking
            self.metadata = ModelMetadata(
                model_type="APRPrediction",
                version=self.version,
                created_at=datetime.now(),
                trained_on=len(X),
                features=self.feature_names,
                metrics={"error": "Insufficient training data"},
                hyperparams={},
                feature_importance={}
            )
            
            return self.metadata
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize models
        try:
            # XGBoost model
            if HAS_XGBOOST:
                self.model = self._create_xgboost_model()
                
                # Train model
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=20,
                    verbose=False
                )
                
                # Evaluate model
                y_pred = self.model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Get feature importance
                importance = self.model.feature_importances_
                feature_importance = dict(zip(self.feature_names, importance))
                
                # Sort by importance
                feature_importance = {k: v for k, v in sorted(
                    feature_importance.items(), 
                    key=lambda item: item[1], 
                    reverse=True
                )}
                
                # Create metadata
                self.metadata = ModelMetadata(
                    model_type="APRPrediction",
                    version=self.version,
                    created_at=datetime.now(),
                    trained_on=len(X_train),
                    features=self.feature_names,
                    metrics={
                        "mse": float(mse),
                        "mae": float(mae),
                        "r2": float(r2)
                    },
                    hyperparams=self.model.get_params(),
                    feature_importance=feature_importance
                )
                
                logger.info(f"Trained APR prediction model v{self.version}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
                
                # LSTM model for time series (if we have enough sequential data)
                # This would require reshaping data for sequence input
                # Placeholder for the concept
                
                # Save model
                self._save_model()
                
                # Increment version for next training
                self.version += 1
                
                return self.metadata
                
            else:
                logger.warning("XGBoost not available, skipping model training")
                
                # Create dummy metadata
                self.metadata = ModelMetadata(
                    model_type="APRPrediction",
                    version=self.version,
                    created_at=datetime.now(),
                    trained_on=len(X_train),
                    features=self.feature_names,
                    metrics={"error": "XGBoost not available"},
                    hyperparams={},
                    feature_importance={}
                )
                
                return self.metadata
                
        except Exception as e:
            logger.error(f"Error training APR prediction model: {e}")
            
            # Create error metadata
            self.metadata = ModelMetadata(
                model_type="APRPrediction",
                version=self.version,
                created_at=datetime.now(),
                trained_on=len(X) if 'X_train' not in locals() else len(X_train),
                features=self.feature_names,
                metrics={"error": str(e)},
                hyperparams={},
                feature_importance={}
            )
            
            return self.metadata
    
    def predict(self, pool_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict future APR change for a pool
        
        Args:
            pool_data: Pool data dictionary
            
        Returns:
            Dictionary with prediction details
        """
        # Check if model is trained
        if self.model is None:
            logger.error("Model not trained")
            return {
                "error": "Model not trained",
                "apr_prediction": 0.0,
                "confidence": 0.0
            }
        
        try:
            # Convert to DataFrame with a single row
            pool_df = pd.DataFrame([pool_data])
            
            # Create features
            features_df = self.feature_engineer.create_features([pool_data])
            
            # Scale features
            X = self.feature_engineer.scale_features(features_df, 'apr_prediction')
            
            # Select only the features used during training
            if self.feature_names:
                # Only keep columns that exist in both X and feature_names
                common_features = [f for f in self.feature_names if f in X.columns]
                X = X[common_features]
                
                # Add missing features with zeros
                missing_features = [f for f in self.feature_names if f not in X.columns]
                for feature in missing_features:
                    X[feature] = 0
                
                # Ensure correct order
                X = X[self.feature_names]
            
            # Make prediction
            apr_change_prediction = float(self.model.predict(X)[0])
            
            # Calculate confidence
            # This is a simplified approach - in a real model, we'd use prediction intervals
            confidence = min(100, max(0, 100 - (abs(apr_change_prediction) * 10)))
            
            # Get current APR
            current_apr = pool_data.get('apr', 0)
            
            # Calculate future APR
            future_apr = current_apr + apr_change_prediction
            
            # Calculate confidence interval
            # For XGBoost, we'd typically use quantile regression or bootstrapping
            # This is a simplified approach
            confidence_margin = abs(apr_change_prediction) * (1 - (confidence / 100))
            lower_bound = future_apr - confidence_margin
            upper_bound = future_apr + confidence_margin
            
            # Create response
            result = {
                "current_apr": current_apr,
                "predicted_apr_change": apr_change_prediction,
                "predicted_future_apr": future_apr,
                "confidence": confidence,
                "confidence_interval": (lower_bound, upper_bound),
                "prediction_horizon_days": self.time_horizon
            }
            
            # Add feature contribution if available
            if HAS_XGBOOST and hasattr(self.model, 'get_booster'):
                pass  # In a real implementation, we would add SHAP values here
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting APR: {e}")
            return {
                "error": str(e),
                "apr_prediction": 0.0,
                "confidence": 0.0
            }
    
    def _save_model(self) -> None:
        """Save the model and metadata"""
        try:
            # Create model directory if it doesn't exist
            model_dir = os.path.join(MODEL_DIR, f"apr_prediction_v{self.version}")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save XGBoost model
            if self.model is not None:
                model_path = os.path.join(model_dir, "xgboost_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                logger.info(f"Saved XGBoost model to {model_path}")
            
            # Save LSTM model
            if self.lstm_model is not None and HAS_TENSORFLOW:
                model_path = os.path.join(model_dir, "lstm_model.h5")
                self.lstm_model.save(model_path)
                logger.info(f"Saved LSTM model to {model_path}")
            
            # Save feature names
            if self.feature_names:
                feature_path = os.path.join(model_dir, "feature_names.json")
                with open(feature_path, 'w') as f:
                    json.dump(self.feature_names, f)
                logger.info(f"Saved feature names to {feature_path}")
            
            # Save metadata
            if self.metadata:
                metadata_path = os.path.join(model_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(self.metadata.to_dict(), f, indent=2)
                logger.info(f"Saved model metadata to {metadata_path}")
            
            # Save feature engineer state
            fe_path = os.path.join(model_dir, "feature_engineer.pkl")
            self.feature_engineer.save_state(fe_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_latest_model(self) -> bool:
        """Load the latest trained model"""
        try:
            # Find the latest model version
            model_dirs = [d for d in os.listdir(MODEL_DIR) if d.startswith("apr_prediction_v")]
            
            if not model_dirs:
                logger.warning("No trained models found")
                return False
            
            # Sort by version number
            model_dirs.sort(key=lambda x: int(x.split('v')[1]), reverse=True)
            latest_dir = model_dirs[0]
            
            # Get version number
            self.version = int(latest_dir.split('v')[1])
            
            model_dir = os.path.join(MODEL_DIR, latest_dir)
            
            # Load XGBoost model
            model_path = os.path.join(model_dir, "xgboost_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded XGBoost model from {model_path}")
            
            # Load LSTM model
            lstm_path = os.path.join(model_dir, "lstm_model.h5")
            if os.path.exists(lstm_path) and HAS_TENSORFLOW:
                self.lstm_model = load_model(lstm_path)
                logger.info(f"Loaded LSTM model from {lstm_path}")
            
            # Load feature names
            feature_path = os.path.join(model_dir, "feature_names.json")
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded feature names from {feature_path}")
            
            # Load metadata
            metadata_path = os.path.join(model_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                    self.metadata = ModelMetadata.from_dict(metadata_dict)
                logger.info(f"Loaded model metadata from {metadata_path}")
            
            # Load feature engineer state
            fe_path = os.path.join(model_dir, "feature_engineer.pkl")
            self.feature_engineer.load_state(fe_path)
            
            logger.info(f"Successfully loaded model version {self.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def evaluate_model_accuracy(self, validation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate model accuracy on validation data
        
        Args:
            validation_data: Historical data with known outcomes
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained")
            return {"error": "Model not trained"}
        
        try:
            # Prepare validation data
            val_df = pd.DataFrame(validation_data)
            
            # We need actual outcomes to compare against
            target_col = f'apr_change_{self.time_horizon}d'
            if target_col not in val_df.columns:
                logger.error(f"Validation data missing target column: {target_col}")
                return {"error": f"Missing target column: {target_col}"}
            
            # Extract actual values
            y_true = val_df[target_col].values
            
            # Generate predictions
            predictions = []
            for pool_data in validation_data:
                pred = self.predict(pool_data)
                predictions.append(pred["predicted_apr_change"])
            
            y_pred = np.array(predictions)
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Calculate directional accuracy (did we predict the direction correctly?)
            direction_true = np.sign(y_true)
            direction_pred = np.sign(y_pred)
            directional_accuracy = np.mean(direction_true == direction_pred)
            
            metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "directional_accuracy": float(directional_accuracy)
            }
            
            logger.info(f"Model evaluation: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"error": str(e)}

class TVLPredictionModel:
    """
    Model specifically trained to predict future TVL changes
    Similar structure to APRPredictionModel but focused on TVL
    """
    
    def __init__(self, time_horizon: int = 7):
        """
        Initialize TVL prediction model
        
        Args:
            time_horizon: Number of days into the future to predict
        """
        self.time_horizon = time_horizon
        self.model = None
        self.lstm_model = None
        self.feature_names = None
        self.metadata = None
        self.version = 1
        self.feature_engineer = FeatureEngineer()
        
    # Implementation similar to APRPredictionModel
    # Methods would be adapted for TVL prediction
    
    def predict(self, pool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified placeholder for TVL prediction"""
        # In a real implementation, this would use a trained model
        # For demonstration, return a reasonable placeholder prediction
        
        current_tvl = pool_data.get('liquidity', 0)
        
        # Use recent TVL changes if available
        tvl_change_7d = pool_data.get('tvl_change_7d', None)
        
        if tvl_change_7d is not None:
            # Predict continued trend but with decay
            predicted_change = tvl_change_7d * 0.8
        else:
            # Default to slight growth based on category
            category = pool_data.get('category', 'Other')
            if category == 'Meme':
                predicted_change = 10.0  # Higher volatility, higher potential growth
            elif category == 'Major':
                predicted_change = 2.0   # Stable growth
            elif category == 'DeFi':
                predicted_change = 5.0   # Moderate growth
            else:
                predicted_change = 3.0   # Default growth
        
        # Add some randomness to the prediction
        predicted_change *= (0.8 + 0.4 * random.random())
        
        # Calculate future TVL
        future_tvl = current_tvl * (1 + (predicted_change / 100))
        
        # Create confidence interval
        confidence = 70  # Default confidence
        margin = abs(predicted_change) * (1 - (confidence / 100)) * 2
        lower_bound = current_tvl * (1 + ((predicted_change - margin) / 100))
        upper_bound = current_tvl * (1 + ((predicted_change + margin) / 100))
        
        return {
            "current_tvl": current_tvl,
            "predicted_tvl_change_percent": predicted_change,
            "predicted_future_tvl": future_tvl,
            "confidence": confidence,
            "confidence_interval": (lower_bound, upper_bound),
            "prediction_horizon_days": self.time_horizon
        }

class RiskAssessmentModel:
    """
    Model for predicting risk level and volatility
    """
    
    def __init__(self):
        """Initialize risk assessment model"""
        self.model = None
        self.metadata = None
        self.version = 1
        
    def predict(self, pool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified risk assessment"""
        # In a real implementation, this would use a trained model
        # For demonstration, calculate risk based on key factors
        
        risk_score = 50  # Default medium risk
        
        # Adjust risk based on pool category
        category = pool_data.get('category', 'Other')
        if category == 'Meme':
            risk_score += 30  # Meme tokens are high risk
        elif category == 'Stablecoin':
            risk_score -= 30  # Stablecoin pairs are low risk
        elif category == 'Major':
            risk_score -= 15  # Major pairs are lower risk
        elif category == 'DeFi':
            risk_score += 10  # DeFi tokens are slightly higher risk
        
        # Adjust for liquidity (higher liquidity = lower risk)
        liquidity = pool_data.get('liquidity', 0)
        if liquidity > 50_000_000:
            risk_score -= 20
        elif liquidity > 10_000_000:
            risk_score -= 10
        elif liquidity < 1_000_000:
            risk_score += 15
        
        # Adjust for volatility if available
        if 'apr_volatility' in pool_data:
            volatility = pool_data['apr_volatility']
            risk_score += min(25, volatility * 10)  # Higher volatility = higher risk
        
        # Calculate implied volatility
        tvl_change_7d = abs(pool_data.get('tvl_change_7d', 0))
        apr_change_7d = abs(pool_data.get('apr_change_7d', 0))
        
        implied_volatility = (tvl_change_7d + apr_change_7d) / 2
        
        # Cap risk score between a valid range
        risk_score = min(100, max(0, risk_score))
        
        # Define risk categories
        if risk_score < 20:
            risk_category = "Very Low"
        elif risk_score < 40:
            risk_category = "Low"
        elif risk_score < 60:
            risk_category = "Medium"
        elif risk_score < 80:
            risk_category = "High"
        else:
            risk_category = "Very High"
        
        return {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "implied_volatility": implied_volatility,
            "key_risk_factors": self._get_risk_factors(pool_data, risk_score)
        }
    
    def _get_risk_factors(self, pool_data: Dict[str, Any], risk_score: float) -> List[str]:
        """Generate explanations of key risk factors"""
        factors = []
        
        # Category-based factors
        category = pool_data.get('category', 'Other')
        if category == 'Meme':
            factors.append("Meme token volatility")
        elif category == 'DeFi':
            factors.append("DeFi protocol risk")
        
        # Liquidity factors
        liquidity = pool_data.get('liquidity', 0)
        if liquidity < 1_000_000:
            factors.append("Low liquidity")
        
        # Volatility factors
        if 'apr_volatility' in pool_data and pool_data['apr_volatility'] > 5:
            factors.append("High APR volatility")
        
        if 'tvl_change_7d' in pool_data:
            tvl_change = pool_data['tvl_change_7d']
            if abs(tvl_change) > 10:
                direction = "increase" if tvl_change > 0 else "decrease"
                factors.append(f"Rapid TVL {direction}")
        
        # Token factors
        token1 = pool_data.get('token1_symbol', '')
        token2 = pool_data.get('token2_symbol', '')
        
        stablecoins = ['USDC', 'USDT', 'DAI', 'BUSD']
        has_stablecoin = token1 in stablecoins or token2 in stablecoins
        
        if not has_stablecoin:
            factors.append("No stablecoin pairing")
        
        # Return top factors based on risk level
        if risk_score >= 80:
            return factors  # Return all factors for high risk
        else:
            return factors[:2]  # Return top factors for lower risk

class PoolPerformanceClassifier:
    """
    Classifier model that categorizes pools into performance classes
    """
    
    def __init__(self):
        """Initialize pool performance classifier"""
        self.model = None
        self.metadata = None
        self.version = 1
        
    def predict(self, pool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict performance class for a pool"""
        # In a real implementation, this would use a trained classifier model
        # For demonstration, classify based on key metrics
        
        # Calculate a performance score based on multiple factors
        performance_score = 0
        
        # APR component
        apr = pool_data.get('apr', 0)
        if apr > 20:
            performance_score += 30
        elif apr > 10:
            performance_score += 20
        elif apr > 5:
            performance_score += 10
        
        # Volume/Liquidity ratio component
        liquidity = pool_data.get('liquidity', 1)
        volume = pool_data.get('volume_24h', 0)
        volume_liquidity_ratio = volume / liquidity
        
        if volume_liquidity_ratio > 0.2:
            performance_score += 25  # Very high activity
        elif volume_liquidity_ratio > 0.1:
            performance_score += 15  # High activity
        elif volume_liquidity_ratio > 0.05:
            performance_score += 10  # Good activity
        
        # TVL growth component
        tvl_change_7d = pool_data.get('tvl_change_7d', 0)
        if tvl_change_7d > 10:
            performance_score += 20  # Strong growth
        elif tvl_change_7d > 5:
            performance_score += 15  # Good growth
        elif tvl_change_7d > 0:
            performance_score += 10  # Positive growth
        elif tvl_change_7d < -10:
            performance_score -= 15  # Strong decline
        
        # APR stability component
        apr_change_7d = pool_data.get('apr_change_7d', 0)
        if abs(apr_change_7d) < 2:
            performance_score += 15  # Very stable APR
        elif abs(apr_change_7d) < 5:
            performance_score += 5   # Moderately stable APR
        
        # Determine performance class
        if performance_score >= 70:
            performance_class = "Excellent"
        elif performance_score >= 50:
            performance_class = "Good"
        elif performance_score >= 30:
            performance_class = "Average"
        elif performance_score >= 10:
            performance_class = "Below Average"
        else:
            performance_class = "Poor"
        
        # Get key performance factors
        performance_factors = self._get_performance_factors(pool_data, performance_score)
        
        return {
            "performance_score": performance_score,
            "performance_class": performance_class,
            "key_factors": performance_factors
        }
    
    def _get_performance_factors(self, pool_data: Dict[str, Any], score: float) -> List[str]:
        """Generate explanations of key performance factors"""
        factors = []
        
        # APR factor
        apr = pool_data.get('apr', 0)
        if apr > 20:
            factors.append("Very high APR")
        elif apr > 10:
            factors.append("Strong APR")
        
        # Activity factor
        liquidity = pool_data.get('liquidity', 1)
        volume = pool_data.get('volume_24h', 0)
        volume_liquidity_ratio = volume / liquidity
        
        if volume_liquidity_ratio > 0.1:
            factors.append("High trading activity")
        
        # Growth factor
        tvl_change_7d = pool_data.get('tvl_change_7d', 0)
        if tvl_change_7d > 5:
            factors.append("Strong liquidity growth")
        elif tvl_change_7d < -5:
            factors.append("Declining liquidity")
        
        # Stability factor
        apr_change_7d = pool_data.get('apr_change_7d', 0)
        if abs(apr_change_7d) < 2:
            factors.append("Stable APR")
        elif abs(apr_change_7d) > 10:
            factors.append("Volatile APR")
        
        # Return the top factors
        return factors[:3]

class MarketTrendAnalyzer:
    """
    Analyzes overall market trends to improve predictions
    """
    
    def __init__(self, history_days: int = 30):
        """
        Initialize market trend analyzer
        
        Args:
            history_days: Number of days of history to analyze
        """
        self.history_days = history_days
        self.market_data = []
        self.last_update = None
        
    def update_market_data(self, pool_data_list: List[Dict[str, Any]]) -> None:
        """
        Update market data with the latest pool information
        
        Args:
            pool_data_list: List of current pool data
        """
        now = datetime.now()
        
        # Create a market snapshot
        avg_apr = np.mean([p.get('apr', 0) for p in pool_data_list])
        avg_tvl = np.mean([p.get('liquidity', 0) for p in pool_data_list])
        total_tvl = np.sum([p.get('liquidity', 0) for p in pool_data_list])
        total_volume = np.sum([p.get('volume_24h', 0) for p in pool_data_list])
        
        # Calculate metrics by category
        categories = {}
        for pool in pool_data_list:
            category = pool.get('category', 'Other')
            if category not in categories:
                categories[category] = {
                    'count': 0,
                    'apr_sum': 0,
                    'tvl_sum': 0,
                    'volume_sum': 0
                }
            
            categories[category]['count'] += 1
            categories[category]['apr_sum'] += pool.get('apr', 0)
            categories[category]['tvl_sum'] += pool.get('liquidity', 0)
            categories[category]['volume_sum'] += pool.get('volume_24h', 0)
        
        # Calculate averages
        category_metrics = {}
        for category, data in categories.items():
            count = data['count']
            if count > 0:
                category_metrics[category] = {
                    'avg_apr': data['apr_sum'] / count,
                    'avg_tvl': data['tvl_sum'] / count,
                    'total_tvl': data['tvl_sum'],
                    'total_volume': data['volume_sum']
                }
        
        # Create the snapshot
        snapshot = {
            'timestamp': now.isoformat(),
            'avg_apr': avg_apr,
            'avg_tvl': avg_tvl,
            'total_tvl': total_tvl,
            'total_volume': total_volume,
            'category_metrics': category_metrics,
            'pool_count': len(pool_data_list)
        }
        
        # Add to market data
        self.market_data.append(snapshot)
        
        # Remove old data
        cutoff_date = now - timedelta(days=self.history_days)
        self.market_data = [
            data for data in self.market_data 
            if datetime.fromisoformat(data['timestamp']) >= cutoff_date
        ]
        
        # Update last update timestamp
        self.last_update = now
        
        # Save market data
        self._save_market_data()
        
        logger.info(f"Updated market data with {len(pool_data_list)} pools")
    
    def get_market_trends(self) -> Dict[str, Any]:
        """
        Calculate market trends from historical data
        
        Returns:
            Dictionary with market trend information
        """
        if not self.market_data:
            return {
                "error": "No market data available"
            }
        
        # Get latest and oldest data points
        latest = self.market_data[-1]
        oldest = self.market_data[0] if len(self.market_data) > 1 else latest
        
        # Calculate overall changes
        tvl_change = ((latest['total_tvl'] - oldest['total_tvl']) / oldest['total_tvl']) * 100 if oldest['total_tvl'] > 0 else 0
        apr_change = latest['avg_apr'] - oldest['avg_apr']
        volume_change = ((latest['total_volume'] - oldest['total_volume']) / oldest['total_volume']) * 100 if oldest['total_volume'] > 0 else 0
        
        # Calculate category trends
        category_trends = {}
        
        for category, metrics in latest['category_metrics'].items():
            if category in oldest['category_metrics']:
                old_metrics = oldest['category_metrics'][category]
                
                category_trends[category] = {
                    'apr_change': metrics['avg_apr'] - old_metrics['avg_apr'],
                    'tvl_change_percent': ((metrics['total_tvl'] - old_metrics['total_tvl']) / old_metrics['total_tvl']) * 100 if old_metrics['total_tvl'] > 0 else 0,
                    'current_avg_apr': metrics['avg_apr'],
                    'current_total_tvl': metrics['total_tvl']
                }
            else:
                # New category
                category_trends[category] = {
                    'apr_change': 0,
                    'tvl_change_percent': 100,  # New, so 100% growth
                    'current_avg_apr': metrics['avg_apr'],
                    'current_total_tvl': metrics['total_tvl']
                }
        
        # Calculate which categories are trending up or down
        trending_up = sorted(
            [(c, trends['tvl_change_percent']) for c, trends in category_trends.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        trending_down = sorted(
            [(c, trends['tvl_change_percent']) for c, trends in category_trends.items()],
            key=lambda x: x[1]
        )[:3]
        
        # Create market state assessment
        market_state = "Neutral"
        if tvl_change > 10:
            market_state = "Strongly Bullish"
        elif tvl_change > 5:
            market_state = "Bullish"
        elif tvl_change < -10:
            market_state = "Strongly Bearish"
        elif tvl_change < -5:
            market_state = "Bearish"
        
        return {
            "market_state": market_state,
            "total_tvl_change_percent": tvl_change,
            "avg_apr_change": apr_change,
            "volume_change_percent": volume_change,
            "trending_categories_up": trending_up,
            "trending_categories_down": trending_down,
            "category_trends": category_trends,
            "data_period_days": self.history_days,
            "last_update": self.last_update.isoformat() if self.last_update else None
        }
    
    def adjust_prediction_for_market(self, prediction: Dict[str, Any], pool_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust a pool prediction based on market trends
        
        Args:
            prediction: Original prediction
            pool_data: Pool data
            
        Returns:
            Adjusted prediction
        """
        if not self.market_data:
            return prediction
        
        # Get market trends
        trends = self.get_market_trends()
        
        # Get pool category
        category = pool_data.get('category', 'Other')
        
        # Check if we have trends for this category
        category_trend = trends.get('category_trends', {}).get(category, None)
        
        if not category_trend:
            return prediction
        
        # Adjust APR prediction if it exists
        if 'predicted_apr_change' in prediction:
            # Get original prediction
            original_apr_change = prediction['predicted_apr_change']
            
            # Calculate adjustment factor based on category trend
            category_apr_trend = category_trend['apr_change']
            
            # If category APR is trending the same direction as our prediction, amplify it
            # If opposite, dampen it
            if (original_apr_change > 0 and category_apr_trend > 0) or (original_apr_change < 0 and category_apr_trend < 0):
                # Same direction, amplify
                adjustment_factor = 1.0 + (abs(category_apr_trend) / 10)  # Up to 10% amplification
            else:
                # Opposite direction, dampen
                adjustment_factor = 1.0 - (abs(category_apr_trend) / 20)  # Up to 5% dampening
            
            # Apply adjustment
            adjusted_apr_change = original_apr_change * adjustment_factor
            
            # Update prediction
            prediction['predicted_apr_change'] = adjusted_apr_change
            prediction['predicted_future_apr'] = prediction['current_apr'] + adjusted_apr_change
            
            # Update confidence interval
            if 'confidence_interval' in prediction:
                lower, upper = prediction['confidence_interval']
                lower_change = lower - prediction['current_apr']
                upper_change = upper - prediction['current_apr']
                
                # Adjust interval
                adjusted_lower = prediction['current_apr'] + (lower_change * adjustment_factor)
                adjusted_upper = prediction['current_apr'] + (upper_change * adjustment_factor)
                
                prediction['confidence_interval'] = (adjusted_lower, adjusted_upper)
            
            # Note the adjustment in the prediction
            prediction['market_adjusted'] = True
            prediction['adjustment_factor'] = adjustment_factor
        
        # Adjust TVL prediction if it exists
        if 'predicted_tvl_change_percent' in prediction:
            # Similar logic for TVL
            original_tvl_change = prediction['predicted_tvl_change_percent']
            category_tvl_trend = category_trend['tvl_change_percent']
            
            if (original_tvl_change > 0 and category_tvl_trend > 0) or (original_tvl_change < 0 and category_tvl_trend < 0):
                adjustment_factor = 1.0 + (abs(category_tvl_trend) / 100)  # Up to 10% amplification
            else:
                adjustment_factor = 1.0 - (abs(category_tvl_trend) / 200)  # Up to 5% dampening
            
            adjusted_tvl_change = original_tvl_change * adjustment_factor
            
            prediction['predicted_tvl_change_percent'] = adjusted_tvl_change
            prediction['predicted_future_tvl'] = prediction['current_tvl'] * (1 + (adjusted_tvl_change / 100))
            
            # Update confidence interval
            if 'confidence_interval' in prediction:
                lower, upper = prediction['confidence_interval']
                lower_pct = ((lower / prediction['current_tvl']) - 1) * 100
                upper_pct = ((upper / prediction['current_tvl']) - 1) * 100
                
                adjusted_lower = prediction['current_tvl'] * (1 + ((lower_pct * adjustment_factor) / 100))
                adjusted_upper = prediction['current_tvl'] * (1 + ((upper_pct * adjustment_factor) / 100))
                
                prediction['confidence_interval'] = (adjusted_lower, adjusted_upper)
            
            # Note the adjustment
            prediction['market_adjusted'] = True
            prediction['adjustment_factor'] = adjustment_factor
        
        return prediction
    
    def _save_market_data(self) -> None:
        """Save market data to a file"""
        try:
            with open(os.path.join(HISTORY_DIR, 'market_data.json'), 'w') as f:
                json.dump(self.market_data, f)
            logger.info("Saved market data")
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
    
    def load_market_data(self) -> bool:
        """Load market data from a file"""
        file_path = os.path.join(HISTORY_DIR, 'market_data.json')
        if not os.path.exists(file_path):
            logger.warning("No market data file found")
            return False
        
        try:
            with open(file_path, 'r') as f:
                self.market_data = json.load(f)
            
            if self.market_data:
                self.last_update = datetime.fromisoformat(self.market_data[-1]['timestamp'])
            
            logger.info(f"Loaded {len(self.market_data)} market data points")
            return True
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return False

class ReinforcementLearningOptimizer:
    """
    Uses reinforcement learning to optimize prediction weights and hyperparameters
    """
    
    def __init__(self):
        """Initialize the RL optimizer"""
        self.model_weights = {
            'apr_prediction': 0.4,
            'tvl_prediction': 0.3,
            'risk_assessment': 0.2,
            'performance_classifier': 0.1
        }
        self.reward_history = []
        self.exploration_rate = 0.2  # Initial exploration rate
        self.min_exploration = 0.05  # Minimum exploration rate
        self.exploration_decay = 0.995  # Decay rate
        self.learning_rate = 0.01  # Learning rate
        self.steps = 0
        
    def optimize_weights(self, prediction_errors: Dict[str, float]) -> Dict[str, float]:
        """
        Update model weights based on prediction errors
        
        Args:
            prediction_errors: Errors for each model type
            
        Returns:
            Updated weights
        """
        # Increase steps
        self.steps += 1
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
        
        # Calculate normalized errors (lower error = better performance)
        total_error = sum(prediction_errors.values())
        
        if total_error == 0:
            # No errors, no need to update
            return self.model_weights
        
        normalized_errors = {
            model: error / total_error
            for model, error in prediction_errors.items()
        }
        
        # Calculate inverse errors (higher = better)
        inverse_errors = {
            model: 1.0 - error
            for model, error in normalized_errors.items()
        }
        
        # Normalize to ensure they sum to 1
        total_inverse = sum(inverse_errors.values())
        if total_inverse > 0:
            target_weights = {
                model: inv / total_inverse
                for model, inv in inverse_errors.items()
            }
        else:
            # Fallback to equal weights
            target_weights = {
                model: 1.0 / len(prediction_errors)
                for model in prediction_errors.keys()
            }
        
        # Explore or exploit
        if random.random() < self.exploration_rate:
            # Exploration: add random noise to weights
            noise_scale = 0.1  # Scale of noise
            for model in self.model_weights:
                if model in target_weights:
                    noise = (random.random() * 2 - 1) * noise_scale
                    target_weights[model] = max(0.01, min(0.7, target_weights[model] + noise))
            
            # Renormalize
            total_weight = sum(target_weights.values())
            target_weights = {
                model: weight / total_weight
                for model, weight in target_weights.items()
            }
        
        # Update weights gradually (learning rate)
        for model in self.model_weights:
            if model in target_weights:
                self.model_weights[model] += self.learning_rate * (target_weights[model] - self.model_weights[model])
        
        # Ensure weights sum to 1
        total_weight = sum(self.model_weights.values())
        self.model_weights = {
            model: weight / total_weight
            for model, weight in self.model_weights.items()
        }
        
        logger.info(f"Updated model weights: {self.model_weights}")
        return self.model_weights
    
    def save_state(self, filepath: str = 'rl_optimizer.json') -> None:
        """Save optimizer state"""
        try:
            state = {
                'model_weights': self.model_weights,
                'exploration_rate': self.exploration_rate,
                'steps': self.steps,
                'reward_history': self.reward_history
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved RL optimizer state to {filepath}")
        except Exception as e:
            logger.error(f"Error saving RL optimizer state: {e}")
    
    def load_state(self, filepath: str = 'rl_optimizer.json') -> bool:
        """Load optimizer state"""
        if not os.path.exists(filepath):
            logger.warning(f"No saved state found at {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.model_weights = state['model_weights']
            self.exploration_rate = state['exploration_rate']
            self.steps = state['steps']
            self.reward_history = state['reward_history']
            
            logger.info(f"Loaded RL optimizer state from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading RL optimizer state: {e}")
            return False

class SelfEvolvingPredictionEngine:
    """
    Main prediction engine that combines multiple models and evolves over time
    
    This advanced self-evolving system integrates:
    1. Ensemble learning with multiple model types
    2. Reinforcement learning for weight optimization
    3. Bayesian optimization for hyperparameter tuning 
    4. Multi-agent system for collaborative intelligence
    5. Neural architecture search for optimal network design
    6. Market trend analysis for contextual predictions
    """
    
    def __init__(self, prediction_horizon: int = 7):
        """
        Initialize the prediction engine
        
        Args:
            prediction_horizon: Default prediction horizon in days
        """
        self.prediction_horizon = prediction_horizon
        self.apr_model = APRPredictionModel(prediction_horizon)
        self.tvl_model = TVLPredictionModel(prediction_horizon)
        self.risk_model = RiskAssessmentModel()
        self.performance_model = PoolPerformanceClassifier()
        self.market_analyzer = MarketTrendAnalyzer()
        self.rl_optimizer = ReinforcementLearningOptimizer()
        self.training_history = []
        self.prediction_history = []
        self.validation_results = []
        self.version = 1
        
        # Advanced self-evolving components (initialized on demand)
        self._bayes_optimizer = None
        self._multi_agent_system = None
        self._neural_architecture_search = None
        
        # Evolvability tracking metrics
        self.evolution_metrics = {
            "hyperparameter_optimizations": 0,
            "architecture_searches": 0,
            "multi_agent_cycles": 0,
            "total_evolution_steps": 0,
            "improvement_rates": [],
            "model_complexity_history": [],
            "feature_importance_history": []
        }
    
    @property
    def bayes_optimizer(self):
        """Lazy initialization of Bayesian optimizer"""
        if not self._bayes_optimizer and HAS_BAYESIAN_OPT:
            try:
                self._bayes_optimizer = HyperparameterOptimizer("xgboost")
                logger.info("Initialized Bayesian optimizer for hyperparameter tuning")
            except Exception as e:
                logger.error(f"Error initializing Bayesian optimizer: {e}")
        return self._bayes_optimizer
    
    @property
    def multi_agent_system(self):
        """Lazy initialization of multi-agent system"""
        if not self._multi_agent_system and HAS_MULTI_AGENT:
            try:
                self._multi_agent_system = MultiAgentSystem()
                self._multi_agent_system.create_default_agents()
                logger.info("Initialized multi-agent system with default agents")
            except Exception as e:
                logger.error(f"Error initializing multi-agent system: {e}")
        return self._multi_agent_system
    
    @property
    def neural_architecture_search(self):
        """Lazy initialization of neural architecture search"""
        if not self._neural_architecture_search and HAS_NEURAL_SEARCH:
            try:
                # Default to a simple time series input shape
                # This would be adjusted based on actual input data
                input_shape = (10, 1)  # 10 time steps, 1 feature
                self._neural_architecture_search = ArchitectureSearch(input_shape)
                logger.info("Initialized neural architecture search")
            except Exception as e:
                logger.error(f"Error initializing neural architecture search: {e}")
        return self._neural_architecture_search
        
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train all prediction models
        
        Args:
            training_data: Historical pool data with known outcomes
            
        Returns:
            Training results
        """
        training_start = datetime.now()
        
        # Create a training record
        training_record = {
            "start_time": training_start.isoformat(),
            "data_points": len(training_data),
            "models": {}
        }
        
        # Train APR prediction model
        try:
            apr_metadata = self.apr_model.train(training_data)
            training_record["models"]["apr_prediction"] = apr_metadata.to_dict()
            logger.info(f"Trained APR prediction model: {apr_metadata.metrics}")
        except Exception as e:
            logger.error(f"Error training APR prediction model: {e}")
            training_record["models"]["apr_prediction"] = {"error": str(e)}
        
        # Would train TVL model in a similar way
        # self.tvl_model.train(training_data)
        
        # Update market analyzer with the latest data
        self.market_analyzer.update_market_data(training_data)
        
        # Complete training record
        training_record["end_time"] = datetime.now().isoformat()
        training_record["duration_seconds"] = (datetime.now() - training_start).total_seconds()
        
        # Add to history
        self.training_history.append(training_record)
        
        # Save training history
        self._save_training_history()
        
        self.version += 1
        
        return training_record
    
    def validate(self, validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate models on holdout data
        
        Args:
            validation_data: Validation data with known outcomes
            
        Returns:
            Validation results
        """
        validation_start = datetime.now()
        
        # Create validation record
        validation_record = {
            "timestamp": validation_start.isoformat(),
            "data_points": len(validation_data),
            "models": {}
        }
        
        # Validate APR prediction model
        try:
            apr_metrics = self.apr_model.evaluate_model_accuracy(validation_data)
            validation_record["models"]["apr_prediction"] = apr_metrics
            logger.info(f"Validated APR prediction model: {apr_metrics}")
        except Exception as e:
            logger.error(f"Error validating APR prediction model: {e}")
            validation_record["models"]["apr_prediction"] = {"error": str(e)}
        
        # Would validate other models in a similar way
        
        # Calculate overall accuracy
        model_errors = {}
        for model_name, metrics in validation_record["models"].items():
            if "error" not in metrics:
                if "mae" in metrics:
                    model_errors[model_name] = metrics["mae"]
                elif "mse" in metrics:
                    model_errors[model_name] = metrics["mse"]
        
        # Use RL optimizer to update model weights
        if model_errors:
            updated_weights = self.rl_optimizer.optimize_weights(model_errors)
            validation_record["model_weights"] = updated_weights
        
        # Add to validation history
        self.validation_results.append(validation_record)
        
        # Save validation history
        self._save_validation_history()
        
        return validation_record
    
    def predict(self, pool_data: Dict[str, Any]) -> PredictionResult:
        """
        Generate a comprehensive prediction for a pool
        
        Args:
            pool_data: Pool data dictionary
            
        Returns:
            PredictionResult with combined predictions
        """
        prediction_time = datetime.now()
        
        # Get predictions from individual models
        apr_prediction = self.apr_model.predict(pool_data)
        tvl_prediction = self.tvl_model.predict(pool_data)
        risk_assessment = self.risk_model.predict(pool_data)
        performance_class = self.performance_model.predict(pool_data)
        
        # Adjust predictions based on market trends
        apr_prediction = self.market_analyzer.adjust_prediction_for_market(apr_prediction, pool_data)
        tvl_prediction = self.market_analyzer.adjust_prediction_for_market(tvl_prediction, pool_data)
        
        # Calculate overall prediction score
        # Weight the individual model confidences according to the RL optimizer weights
        weights = self.rl_optimizer.model_weights
        
        apr_confidence = apr_prediction.get('confidence', 0)
        tvl_confidence = tvl_prediction.get('confidence', 0)
        
        # Invert risk score to get a "confidence" (lower risk = higher confidence)
        risk_confidence = max(0, 100 - risk_assessment.get('risk_score', 50))
        
        # Map performance class to a score
        performance_score = {
            "Excellent": 90,
            "Good": 75,
            "Average": 50,
            "Below Average": 30,
            "Poor": 10
        }.get(performance_class.get('performance_class', "Average"), 50)
        
        # Calculate weighted score
        prediction_score = (
            weights.get('apr_prediction', 0.4) * apr_confidence +
            weights.get('tvl_prediction', 0.3) * tvl_confidence +
            weights.get('risk_assessment', 0.2) * risk_confidence +
            weights.get('performance_classifier', 0.1) * performance_score
        )
        
        # Gather contributing factors
        contributing_factors = []
        
        # Add APR factors
        if 'predicted_apr_change' in apr_prediction:
            change = apr_prediction['predicted_apr_change']
            if change > 2:
                contributing_factors.append(f"APR predicted to increase by {change:.2f}%")
            elif change < -2:
                contributing_factors.append(f"APR predicted to decrease by {abs(change):.2f}%")
        
        # Add TVL factors
        if 'predicted_tvl_change_percent' in tvl_prediction:
            change = tvl_prediction['predicted_tvl_change_percent']
            if change > 5:
                contributing_factors.append(f"TVL predicted to increase by {change:.2f}%")
            elif change < -5:
                contributing_factors.append(f"TVL predicted to decrease by {abs(change):.2f}%")
        
        # Add risk factors
        risk_factors = risk_assessment.get('key_risk_factors', [])
        contributing_factors.extend(risk_factors)
        
        # Add performance factors
        performance_factors = performance_class.get('key_factors', [])
        contributing_factors.extend(performance_factors)
        
        # Deduplicate factors
        contributing_factors = list(set(contributing_factors))
        
        # Create prediction result
        result = PredictionResult(
            pool_id=pool_data.get('id', 'unknown'),
            prediction_date=prediction_time,
            prediction_horizon=self.prediction_horizon,
            target_metrics={
                'apr': apr_prediction.get('current_apr'),
                'tvl': tvl_prediction.get('current_tvl'),
                'risk_score': risk_assessment.get('risk_score'),
                'performance_class': performance_class.get('performance_class')
            },
            prediction_values={
                'future_apr': apr_prediction.get('predicted_future_apr'),
                'apr_change': apr_prediction.get('predicted_apr_change'),
                'future_tvl': tvl_prediction.get('predicted_future_tvl'),
                'tvl_change_percent': tvl_prediction.get('predicted_tvl_change_percent')
            },
            confidence_intervals={
                'apr': apr_prediction.get('confidence_interval', (0, 0)),
                'tvl': tvl_prediction.get('confidence_interval', (0, 0))
            },
            prediction_score=prediction_score,
            contributing_factors=contributing_factors,
            model_version={
                'apr_model': self.apr_model.version,
                'engine': self.version
            }
        )
        
        # Add to prediction history
        self.prediction_history.append(result.to_dict())
        
        # Save prediction history periodically (every 10 predictions)
        if len(self.prediction_history) % 10 == 0:
            self._save_prediction_history()
        
        return result
    
    def evolve(self, historical_data: List[Dict[str, Any]], validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Evolve the prediction engine by training and validating models.
        This advanced method integrates multiple evolutionary approaches:
        
        1. Bayesian hyperparameter optimization
        2. Neural architecture search
        3. Multi-agent collaborative learning
        4. Reinforcement learning optimization
        5. Feature selection evolution
        
        Args:
            historical_data: Historical pool data
            validation_split: Portion of data to use for validation
            
        Returns:
            Evolution results with comprehensive metrics
        """
        evolution_start = datetime.now()
        self.evolution_metrics["total_evolution_steps"] += 1
        
        # Split data into training and validation
        split_idx = int(len(historical_data) * (1 - validation_split))
        training_data = historical_data[:split_idx]
        validation_data = historical_data[split_idx:]
        
        # Basic model training and validation (baseline approach)
        training_results = self.train(training_data)
        validation_results = self.validate(validation_data)
        
        # Track initial performance metrics for improvement measurement
        initial_metrics = self._extract_performance_metrics(validation_results)
        
        # ===== Advanced Evolution Steps =====
        evolution_steps_results = {}
        
        # Step 1: Bayesian hyperparameter optimization (if available)
        if self.bayes_optimizer:
            try:
                evolution_steps_results["bayesian_optimization"] = self._evolve_with_bayesian_optimization(
                    training_data, validation_data
                )
                self.evolution_metrics["hyperparameter_optimizations"] += 1
            except Exception as e:
                logger.error(f"Error in Bayesian optimization step: {e}")
                evolution_steps_results["bayesian_optimization"] = {"error": str(e)}
        
        # Step 2: Neural architecture search (if available and appropriate)
        if self.neural_architecture_search and len(training_data) >= 500:  # Only with sufficient data
            try:
                evolution_steps_results["neural_architecture_search"] = self._evolve_with_neural_architecture_search(
                    training_data, validation_data
                )
                self.evolution_metrics["architecture_searches"] += 1
            except Exception as e:
                logger.error(f"Error in neural architecture search step: {e}")
                evolution_steps_results["neural_architecture_search"] = {"error": str(e)}
        
        # Step 3: Multi-agent collective intelligence (if available)
        if self.multi_agent_system:
            try:
                evolution_steps_results["multi_agent_learning"] = self._evolve_with_multi_agent_system(
                    training_data, validation_data
                )
                self.evolution_metrics["multi_agent_cycles"] += 1
            except Exception as e:
                logger.error(f"Error in multi-agent evolution step: {e}")
                evolution_steps_results["multi_agent_learning"] = {"error": str(e)}
        
        # Calculate performance improvement
        final_validation = self.validate(validation_data)
        final_metrics = self._extract_performance_metrics(final_validation)
        
        improvement = self._calculate_improvement(initial_metrics, final_metrics)
        self.evolution_metrics["improvement_rates"].append(improvement)
        
        # Track model complexity
        complexity = {
            "feature_count": len(self.apr_model.feature_names) if hasattr(self.apr_model, 'feature_names') else 0,
            "parameter_count": self._estimate_model_parameters(),
            "version": self.version
        }
        self.evolution_metrics["model_complexity_history"].append(complexity)
        
        # Save state of all components
        self._save_evolution_state()
        
        # Format results
        evolution_results = {
            "training_results": training_results,
            "initial_validation": validation_results,
            "final_validation": final_validation,
            "evolution_steps": evolution_steps_results,
            "improvement": improvement,
            "model_weights": self.rl_optimizer.model_weights,
            "version": self.version,
            "duration_seconds": (datetime.now() - evolution_start).total_seconds()
        }
        
        logger.info(f"Evolution cycle complete. Improvement: {improvement:.2f}%, Version: {self.version}")
        return evolution_results
    
    def _evolve_with_bayesian_optimization(self, 
                                         training_data: List[Dict[str, Any]], 
                                         validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evolve models using Bayesian hyperparameter optimization
        
        Args:
            training_data: Training data
            validation_data: Validation data
            
        Returns:
            Optimization results
        """
        logger.info("Starting Bayesian hyperparameter optimization")
        
        # Define evaluation function for hyperparameter optimization
        def evaluate_hyperparams(params: Dict[str, Any]) -> float:
            # Create a temporary model with these hyperparameters
            temp_model = APRPredictionModel(self.prediction_horizon)
            
            # Apply hyperparameters
            if HAS_XGBOOST and temp_model._create_xgboost_model:
                xgb_params = {
                    "learning_rate": params.get("learning_rate", 0.1),
                    "max_depth": params.get("max_depth", 6),
                    "subsample": params.get("subsample", 0.8),
                    "colsample_bytree": params.get("colsample_bytree", 0.8),
                    "reg_alpha": params.get("reg_alpha", 0.1),
                    "reg_lambda": params.get("reg_lambda", 1.0)
                }
                temp_model.model = temp_model._create_xgboost_model()
                temp_model.model.set_params(**xgb_params)
            
            # Train on training data
            X_train, y_train = temp_model.prepare_training_data(training_data)
            
            # We can't do a full train since it's slow, so let's simulate
            if temp_model.model is not None:
                try:
                    temp_model.model.fit(X_train, y_train)
                    
                    # Evaluate on validation data
                    X_val, y_val = temp_model.prepare_training_data(validation_data)
                    y_pred = temp_model.model.predict(X_val)
                    
                    # Calculate error (negated for maximization)
                    mae = mean_absolute_error(y_val, y_pred)
                    return -mae  # Negative because we want to maximize score (minimize error)
                except Exception as e:
                    logger.error(f"Error evaluating hyperparameters: {e}")
                    return -float('inf')  # Worst possible score
            else:
                return -float('inf')
        
        # Define hyperparameter space based on model type
        if self.bayes_optimizer and HAS_BAYESIAN_OPT:
            # Run optimization
            results = self.bayes_optimizer.optimize(
                evaluation_function=evaluate_hyperparams,
                n_iterations=20,
                maximize=True
            )
            
            # Apply best parameters to actual model
            if 'best_params' in results and results['best_params']:
                best_params = results['best_params']
                
                # Update model parameters if possible
                if HAS_XGBOOST and hasattr(self.apr_model, 'model') and self.apr_model.model:
                    xgb_params = {k: v for k, v in best_params.items() 
                                  if k in ['learning_rate', 'max_depth', 'subsample', 
                                          'colsample_bytree', 'reg_alpha', 'reg_lambda']}
                    self.apr_model.model.set_params(**xgb_params)
                    logger.info(f"Applied best hyperparameters to APR model: {xgb_params}")
            
            return results
        else:
            return {"error": "Bayesian optimizer not available"}
    
    def _evolve_with_neural_architecture_search(self, 
                                              training_data: List[Dict[str, Any]], 
                                              validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evolve models using neural architecture search
        
        Args:
            training_data: Training data
            validation_data: Validation data
            
        Returns:
            Architecture search results
        """
        logger.info("Starting neural architecture search")
        
        if not HAS_NEURAL_SEARCH or not self.neural_architecture_search:
            return {"error": "Neural architecture search not available"}
        
        if not HAS_TENSORFLOW:
            return {"error": "TensorFlow not available for neural architecture search"}
        
        try:
            # Prepare data for neural architecture search
            # We need to convert pool data to numpy arrays for neural networks
            feature_engineer = FeatureEngineer()
            X_train_df = feature_engineer.create_features(training_data)
            
            # Scale features
            X_train_df = feature_engineer.scale_features(X_train_df)
            
            # Extract target variable (APR change)
            target_col = f'apr_change_{self.prediction_horizon}d'
            if target_col in X_train_df.columns:
                y_train = X_train_df[target_col].values
                X_train_df = X_train_df.drop(columns=[target_col])
            else:
                # Fallback to apr_change_7d if specific horizon not available
                y_train = X_train_df['apr_change_7d'].values if 'apr_change_7d' in X_train_df.columns else np.zeros(len(X_train_df))
                if 'apr_change_7d' in X_train_df.columns:
                    X_train_df = X_train_df.drop(columns=['apr_change_7d'])
            
            # Convert to numpy arrays
            X_train = X_train_df.values
            
            # Do the same for validation data
            X_val_df = feature_engineer.create_features(validation_data)
            X_val_df = feature_engineer.scale_features(X_val_df)
            
            if target_col in X_val_df.columns:
                y_val = X_val_df[target_col].values
                X_val_df = X_val_df.drop(columns=[target_col])
            else:
                y_val = X_val_df['apr_change_7d'].values if 'apr_change_7d' in X_val_df.columns else np.zeros(len(X_val_df))
                if 'apr_change_7d' in X_val_df.columns:
                    X_val_df = X_val_df.drop(columns=['apr_change_7d'])
            
            X_val = X_val_df.values
            
            # Update neural architecture search input shape
            self._neural_architecture_search = ArchitectureSearch(
                input_shape=(X_train.shape[1],),  # Feature dimension
                output_shape=1  # Regression target
            )
            
            # Run a simplified search (for demonstration)
            best_arch = self.neural_architecture_search.run_search(
                train_data=(X_train, y_train),
                val_data=(X_val, y_val),
                generations=2,  # Limit for demonstration
                population_size=5,
                epochs=5
            )
            
            if best_arch:
                # We could create and save the best model here
                # For demonstration, we'll just return the architecture details
                return {
                    "best_architecture_id": best_arch.architecture_id,
                    "layers": len(best_arch.layers),
                    "fitness": best_arch.calculate_fitness(),
                    "estimated_parameters": best_arch.estimate_complexity()
                }
            else:
                return {"error": "No architecture found"}
                
        except Exception as e:
            logger.error(f"Error in neural architecture search: {e}")
            return {"error": str(e)}
    
    def _evolve_with_multi_agent_system(self, 
                                       training_data: List[Dict[str, Any]], 
                                       validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evolve prediction capabilities using multi-agent system
        
        Args:
            training_data: Training data
            validation_data: Validation data
            
        Returns:
            Multi-agent evolution results
        """
        logger.info("Starting multi-agent collective intelligence evolution")
        
        if not HAS_MULTI_AGENT or not self.multi_agent_system:
            return {"error": "Multi-agent system not available"}
        
        try:
            # Process some training examples through the multi-agent system
            results = []
            
            # Take a subset of data for efficiency
            sample_size = min(100, len(training_data))
            training_sample = random.sample(training_data, sample_size)
            
            for pool_data in training_sample:
                # Get target variable based on prediction horizon
                target_col = f'apr_change_{self.prediction_horizon}d'
                target_variable = "apr_change"  # Default
                
                # Use multi-agent prediction
                prediction = self.multi_agent_system.predict(pool_data, target_variable)
                
                # If we have the true outcome, update the system
                true_value = pool_data.get(target_col, pool_data.get('apr_change_7d'))
                if true_value is not None:
                    self.multi_agent_system.update_with_outcome(prediction['task_id'], true_value)
                
                results.append({
                    "pool_id": pool_data.get('id', 'unknown'),
                    "prediction": prediction['value'],
                    "confidence": prediction['confidence']
                })
            
            # Trigger an evolutionary step in the multi-agent system
            self.multi_agent_system._evolutionary_step()
            
            # Get system metrics
            system_state = {
                "agent_count": len(self.multi_agent_system.agents),
                "tasks_completed": self.multi_agent_system.system_metrics['tasks_completed'],
                "knowledge_items": self.multi_agent_system.system_metrics['knowledge_items'],
                "evolution_cycles": self.multi_agent_system.system_metrics['evolution_cycles'],
                "avg_prediction_error": self.multi_agent_system.system_metrics['avg_prediction_error']
            }
            
            return {
                "sample_predictions": results[:5],  # Show just a few
                "system_state": system_state
            }
            
        except Exception as e:
            logger.error(f"Error in multi-agent evolution: {e}")
            return {"error": str(e)}
    
    def _extract_performance_metrics(self, validation_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract key performance metrics from validation results"""
        metrics = {}
        
        # Extract metrics from various models
        if 'models' in validation_results:
            for model_name, model_metrics in validation_results['models'].items():
                if 'error' not in model_metrics:
                    for metric_name, value in model_metrics.items():
                        if isinstance(value, (int, float)):
                            metrics[f"{model_name}_{metric_name}"] = value
        
        return metrics
    
    def _calculate_improvement(self, initial_metrics: Dict[str, float], 
                              final_metrics: Dict[str, float]) -> float:
        """Calculate overall improvement percentage"""
        if not initial_metrics or not final_metrics:
            return 0.0
            
        # Find common metrics
        common_metrics = set(initial_metrics.keys()) & set(final_metrics.keys())
        if not common_metrics:
            return 0.0
            
        # Calculate improvement for each metric
        improvements = []
        for metric in common_metrics:
            if 'mae' in metric or 'mse' in metric or 'error' in metric:
                # For error metrics, lower is better
                if initial_metrics[metric] > 0:  # Avoid division by zero
                    improvement = (initial_metrics[metric] - final_metrics[metric]) / initial_metrics[metric] * 100
                    improvements.append(improvement)
            elif 'accuracy' in metric or 'score' in metric:
                # For accuracy metrics, higher is better
                if initial_metrics[metric] > 0:  # Avoid division by zero
                    improvement = (final_metrics[metric] - initial_metrics[metric]) / initial_metrics[metric] * 100
                    improvements.append(improvement)
        
        # Average improvement
        if improvements:
            return sum(improvements) / len(improvements)
        return 0.0
    
    def _estimate_model_parameters(self) -> int:
        """Estimate total number of parameters in all models"""
        total_params = 0
        
        # XGBoost model parameters
        if hasattr(self.apr_model, 'model') and self.apr_model.model is not None:
            if hasattr(self.apr_model.model, 'get_booster'):
                # Get number of trees and their depth
                booster = self.apr_model.model.get_booster()
                trees = booster.get_dump()
                for tree in trees:
                    # Rough estimate: count nodes in the tree dump
                    total_params += tree.count('\n') + 1
            else:
                # Rough estimate based on hyperparameters
                if hasattr(self.apr_model.model, 'n_estimators'):
                    n_trees = getattr(self.apr_model.model, 'n_estimators', 100)
                    max_depth = getattr(self.apr_model.model, 'max_depth', 6)
                    # Roughly 2^depth - 1 nodes per tree
                    tree_params = 2 ** max_depth - 1
                    total_params += n_trees * tree_params
        
        # LSTM model parameters (if any)
        if hasattr(self.apr_model, 'lstm_model') and self.apr_model.lstm_model is not None:
            # Keras models have a useful method for this
            total_params += self.apr_model.lstm_model.count_params()
        
        return total_params
    
    def _save_evolution_state(self) -> None:
        """Save state of all evolutionary components"""
        # Save RL optimizer
        self.rl_optimizer.save_state(os.path.join(MODEL_DIR, f"rl_optimizer_v{self.version}.json"))
        
        # Save multi-agent system (if available)
        if self.multi_agent_system:
            try:
                self.multi_agent_system.save_state(os.path.join(MODEL_DIR, f"multi_agent_v{self.version}.pkl"))
            except Exception as e:
                logger.error(f"Error saving multi-agent system state: {e}")
        
        # Save evolution metrics
        try:
            metrics_path = os.path.join(HISTORY_DIR, "evolution_metrics.json")
            with open(metrics_path, 'w') as f:
                # Convert any non-serializable values
                metrics_copy = deepcopy(self.evolution_metrics)
                json.dump(metrics_copy, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving evolution metrics: {e}")
    
    def auto_evolve(self, historical_data: List[Dict[str, Any]], min_improvement: float = 0.05) -> Dict[str, Any]:
        """
        Automatically evolve the model until improvement is below threshold
        
        Args:
            historical_data: Historical pool data
            min_improvement: Minimum improvement to continue evolution
            
        Returns:
            Final evolution results
        """
        max_iterations = 5
        prev_metrics = None
        results = {}
        
        for i in range(max_iterations):
            logger.info(f"Starting evolution iteration {i+1}/{max_iterations}")
            
            # Evolve once
            iteration_results = self.evolve(historical_data)
            
            # Get validation metrics
            current_metrics = iteration_results["validation_results"]["models"].get("apr_prediction", {})
            
            if "error" in current_metrics:
                logger.error(f"Error in evolution iteration {i+1}: {current_metrics['error']}")
                break
                
            if prev_metrics is not None:
                # Calculate improvement
                if "mae" in prev_metrics and "mae" in current_metrics:
                    improvement = (prev_metrics["mae"] - current_metrics["mae"]) / prev_metrics["mae"]
                    
                    logger.info(f"Iteration {i+1} improvement: {improvement:.4f}")
                    
                    # Check if improvement is below threshold
                    if improvement < min_improvement:
                        logger.info(f"Improvement below threshold ({improvement:.4f} < {min_improvement}), stopping evolution")
                        break
                        
            prev_metrics = current_metrics
            results = iteration_results
        
        logger.info(f"Auto-evolution completed after {i+1} iterations")
        return results
    
    def load_latest_models(self) -> bool:
        """Load the latest versions of all models"""
        try:
            # Load APR model
            apr_loaded = self.apr_model.load_latest_model()
            
            # Load market analyzer data
            market_loaded = self.market_analyzer.load_market_data()
            
            # Load RL optimizer state
            optimizer_loaded = self.rl_optimizer.load_state()
            
            # Load training and validation history
            self._load_training_history()
            self._load_validation_history()
            self._load_prediction_history()
            
            return apr_loaded
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _save_training_history(self) -> None:
        """Save training history to a file"""
        try:
            with open(os.path.join(HISTORY_DIR, 'training_history.json'), 'w') as f:
                json.dump(self.training_history, f)
            logger.info("Saved training history")
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
    
    def _load_training_history(self) -> bool:
        """Load training history from a file"""
        file_path = os.path.join(HISTORY_DIR, 'training_history.json')
        if not os.path.exists(file_path):
            logger.warning("No training history file found")
            return False
        
        try:
            with open(file_path, 'r') as f:
                self.training_history = json.load(f)
            logger.info(f"Loaded {len(self.training_history)} training records")
            return True
        except Exception as e:
            logger.error(f"Error loading training history: {e}")
            return False
    
    def _save_validation_history(self) -> None:
        """Save validation history to a file"""
        try:
            with open(os.path.join(HISTORY_DIR, 'validation_history.json'), 'w') as f:
                json.dump(self.validation_results, f)
            logger.info("Saved validation history")
        except Exception as e:
            logger.error(f"Error saving validation history: {e}")
    
    def _load_validation_history(self) -> bool:
        """Load validation history from a file"""
        file_path = os.path.join(HISTORY_DIR, 'validation_history.json')
        if not os.path.exists(file_path):
            logger.warning("No validation history file found")
            return False
        
        try:
            with open(file_path, 'r') as f:
                self.validation_results = json.load(f)
            logger.info(f"Loaded {len(self.validation_results)} validation records")
            return True
        except Exception as e:
            logger.error(f"Error loading validation history: {e}")
            return False
    
    def _save_prediction_history(self) -> None:
        """Save prediction history to a file"""
        try:
            with open(os.path.join(HISTORY_DIR, 'prediction_history.json'), 'w') as f:
                json.dump(self.prediction_history, f)
            logger.info(f"Saved {len(self.prediction_history)} prediction records")
        except Exception as e:
            logger.error(f"Error saving prediction history: {e}")
    
    def _load_prediction_history(self) -> bool:
        """Load prediction history from a file"""
        file_path = os.path.join(HISTORY_DIR, 'prediction_history.json')
        if not os.path.exists(file_path):
            logger.warning("No prediction history file found")
            return False
        
        try:
            with open(file_path, 'r') as f:
                self.prediction_history = json.load(f)
            logger.info(f"Loaded {len(self.prediction_history)} prediction records")
            return True
        except Exception as e:
            logger.error(f"Error loading prediction history: {e}")
            return False

def run_prediction_evolution(data_path: str) -> None:
    """
    Run a full evolution cycle on historical data
    
    Args:
        data_path: Path to historical data file
    """
    try:
        # Load historical data
        with open(data_path, 'r') as f:
            historical_data = json.load(f)
        
        logger.info(f"Loaded {len(historical_data)} historical data points")
        
        # Initialize prediction engine
        engine = SelfEvolvingPredictionEngine()
        
        # Run auto-evolution
        results = engine.auto_evolve(historical_data)
        
        # Save results
        with open(os.path.join(HISTORY_DIR, 'evolution_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Prediction evolution completed and results saved")
        
    except Exception as e:
        logger.error(f"Error in prediction evolution: {e}")

def predict_pool_performance(pool_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run prediction on a single pool
    
    Args:
        pool_data: Pool data dictionary
        
    Returns:
        Prediction results
    """
    try:
        # Initialize prediction engine
        engine = SelfEvolvingPredictionEngine()
        
        # Load latest models
        engine.load_latest_models()
        
        # Generate prediction
        result = engine.predict(pool_data)
        
        # Return as dictionary
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error in pool prediction: {e}")
        return {
            "error": str(e),
            "pool_id": pool_data.get('id', 'unknown')
        }

def predict_multiple_pools(pool_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run predictions on multiple pools
    
    Args:
        pool_data_list: List of pool data dictionaries
        
    Returns:
        List of prediction results
    """
    try:
        # Initialize prediction engine
        engine = SelfEvolvingPredictionEngine()
        
        # Load latest models
        engine.load_latest_models()
        
        # Generate predictions
        results = []
        
        for pool_data in pool_data_list:
            result = engine.predict(pool_data)
            results.append(result.to_dict())
        
        return results
        
    except Exception as e:
        logger.error(f"Error in multi-pool prediction: {e}")
        return [{"error": str(e)}]

# Example usage of the module
if __name__ == "__main__":
    # Load sample data
    sample_path = "sample_pool_data.json"
    
    if os.path.exists(sample_path):
        try:
            with open(sample_path, 'r') as f:
                sample_data = json.load(f)
            
            # Check if we have any data
            if sample_data:
                # Run evolution on a subset of the data
                run_prediction_evolution(sample_path)
                
                # Generate some example predictions
                if len(sample_data) > 0:
                    example_pool = sample_data[0]
                    prediction = predict_pool_performance(example_pool)
                    
                    print("\nSample prediction:")
                    print(f"Pool: {example_pool.get('name')}")
                    print(f"Current APR: {example_pool.get('apr', 0):.2f}%")
                    print(f"Predicted APR in {prediction.get('prediction_horizon', 7)} days: {prediction.get('prediction_values', {}).get('future_apr', 0):.2f}%")
                    print(f"Prediction score: {prediction.get('prediction_score', 0):.1f}/100")
                    print(f"Key factors: {prediction.get('contributing_factors', [])}")
            else:
                print("Sample data is empty")
        except Exception as e:
            print(f"Error processing sample data: {e}")
    else:
        print(f"Sample data file not found at {sample_path}")
        
        # Create a dummy sample
        sample_pool = {
            "id": "sample_pool_1",
            "name": "SOL/USDC",
            "dex": "Raydium",
            "category": "Major",
            "token1_symbol": "SOL",
            "token2_symbol": "USDC",
            "liquidity": 24532890.45,
            "volume_24h": 8763021.32,
            "apr": 12.87,
            "apr_change_24h": 0.42,
            "apr_change_7d": 1.2,
            "apr_change_30d": -2.1,
            "tvl_change_24h": 1.1,
            "tvl_change_7d": 3.5,
            "tvl_change_30d": -2.1
        }
        
        # Run prediction on the dummy sample
        prediction = predict_pool_performance(sample_pool)
        
        print("\nSample prediction (from dummy data):")
        print(f"Pool: {sample_pool.get('name')}")
        print(f"Current APR: {sample_pool.get('apr', 0):.2f}%")
        print(f"Predicted APR change: {prediction.get('prediction_values', {}).get('apr_change', 0):.2f}%")
        print(f"Predicted future APR: {prediction.get('prediction_values', {}).get('future_apr', 0):.2f}%")
        print(f"Prediction score: {prediction.get('prediction_score', 0):.1f}/100")
        print(f"Key factors: {prediction.get('contributing_factors', [])}")