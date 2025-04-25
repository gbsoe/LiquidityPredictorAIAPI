import pandas as pd
import numpy as np
import logging
import pickle
import os
import sys
import time
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Handle potential import errors gracefully
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available. Using RandomForest as fallback.")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not available. Using RandomForest as fallback.")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_operations import DBManager
from eda.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('ml_models')

# Define constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

class APRPredictionModel:
    """
    Random Forest Regressor for predicting APR
    """
    
    def __init__(self):
        """Initialize the APR prediction model"""
        self.model = None
        self.feature_names = None
        self.model_path = os.path.join(MODELS_DIR, 'apr_prediction_model.pkl')
        self.version = datetime.now().strftime("%Y%m%d")
        self.scaler = StandardScaler()
    
    def get_feature_names(self, X):
        """Get feature names from the dataset"""
        return list(X.columns)
    
    def preprocess_data(self, df):
        """Preprocess data for training/prediction"""
        try:
            # Drop non-feature columns
            drop_cols = ['id', 'pool_id', 'timestamp', 'next_apr', 'performance_class', 
                        'risk_score', 'pool_name']
            feature_cols = [col for col in df.columns if col not in drop_cols]
            
            # Handle categorical features if any
            # (In this case, we'll just drop them for simplicity)
            categorical_cols = []
            feature_cols = [col for col in feature_cols if col not in categorical_cols]
            
            # Create feature matrix and target vector
            X = df[feature_cols]
            y = df['next_apr']
            
            # Save feature names
            self.feature_names = feature_cols
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def train(self, df):
        """Train the model with the provided data"""
        logger.info("Training APR prediction model")
        
        try:
            # Preprocess data
            X, y = self.preprocess_data(df)
            
            if X.empty or y.empty:
                logger.error("Empty dataset after preprocessing")
                return False
            
            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Create and train the model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Define the pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Parameters for grid search
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [5, 10, 15]
            }
            
            # Perform grid search with time series CV
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                verbose=1
            )
            
            # Train the model
            start_time = time.time()
            grid_search.fit(X, y)
            training_time = time.time() - start_time
            
            # Get the best model
            self.model = grid_search.best_estimator_
            
            # Save the model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Log results
            logger.info(f"APR prediction model trained in {training_time:.2f} seconds")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Model saved to {self.model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training APR prediction model: {e}")
            return False
    
    def evaluate(self, df):
        """Evaluate the model on test data"""
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Preprocess data
            X, y = self.preprocess_data(df)
            
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            
            results = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            logger.info(f"APR prediction model evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating APR prediction model: {e}")
            return None
    
    def predict(self, df):
        """Make predictions using the trained model"""
        if self.model is None:
            # Try to load the model
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except:
                logger.error("Model not trained yet and could not load from file")
                return None
        
        try:
            # If this is a single pool dataframe, we need to preprocess
            if 'next_apr' not in df.columns:
                # Use only the features that were used in training
                if self.feature_names is not None:
                    feature_cols = [col for col in self.feature_names if col in df.columns]
                    missing_cols = [col for col in self.feature_names if col not in df.columns]
                    
                    if missing_cols:
                        logger.warning(f"Missing feature columns: {missing_cols}")
                        # Fill missing columns with zeros
                        for col in missing_cols:
                            df[col] = 0
                    
                    X = df[self.feature_names]
                else:
                    # If feature names not saved, use all numeric features
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    X = df[numeric_cols]
            else:
                # Preprocess data
                X, _ = self.preprocess_data(df)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Add predictions to the dataframe
            result_df = df.copy()
            result_df['predicted_apr'] = predictions
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None

class PoolPerformanceClassifier:
    """
    XGBoost Classifier for pool performance classification
    """
    
    def __init__(self):
        """Initialize the pool performance classifier"""
        self.model = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.model_path = os.path.join(MODELS_DIR, 'pool_performance_classifier.pkl')
        self.version = datetime.now().strftime("%Y%m%d")
    
    def preprocess_data(self, df):
        """Preprocess data for training/prediction"""
        try:
            # Drop non-feature columns
            drop_cols = ['id', 'pool_id', 'timestamp', 'next_apr', 'performance_class', 
                        'risk_score', 'pool_name']
            feature_cols = [col for col in df.columns if col not in drop_cols]
            
            # Create feature matrix and target vector
            X = df[feature_cols]
            
            # Check if performance_class exists (for training)
            if 'performance_class' in df.columns:
                # Encode the target variable
                y = self.label_encoder.fit_transform(df['performance_class'])
            else:
                y = None
            
            # Save feature names
            self.feature_names = feature_cols
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def train(self, df):
        """Train the model with the provided data"""
        logger.info("Training pool performance classifier")
        
        try:
            # Preprocess data
            X, y = self.preprocess_data(df)
            
            if X.empty or y is None:
                logger.error("Empty dataset after preprocessing")
                return False
            
            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Create classifier based on available packages
            if HAS_XGBOOST:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
            else:
                # Fallback to RandomForest if XGBoost is not available
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Define the pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Parameters for grid search
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [3, 5, 7]
            }
            
            # Perform grid search with time series CV
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=tscv,
                scoring='accuracy',
                verbose=1
            )
            
            # Train the model
            start_time = time.time()
            grid_search.fit(X, y)
            training_time = time.time() - start_time
            
            # Get the best model
            self.model = grid_search.best_estimator_
            
            # Save the model and label encoder
            with open(self.model_path, 'wb') as f:
                pickle.dump({'model': self.model, 'label_encoder': self.label_encoder}, f)
            
            # Log results
            logger.info(f"Pool performance classifier trained in {training_time:.2f} seconds")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Model saved to {self.model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training pool performance classifier: {e}")
            return False
    
    def evaluate(self, df):
        """Evaluate the model on test data"""
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Preprocess data
            X, y = self.preprocess_data(df)
            
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, classification_report
            
            accuracy = accuracy_score(y, y_pred)
            report = classification_report(y, y_pred, 
                                         target_names=self.label_encoder.classes_, 
                                         output_dict=True)
            
            results = {
                'accuracy': accuracy,
                'classification_report': report
            }
            
            logger.info(f"Pool performance classifier evaluation: Accuracy={accuracy:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating pool performance classifier: {e}")
            return None
    
    def predict(self, df):
        """Make predictions using the trained model"""
        if self.model is None:
            # Try to load the model
            try:
                with open(self.model_path, 'rb') as f:
                    loaded = pickle.load(f)
                    self.model = loaded['model']
                    self.label_encoder = loaded['label_encoder']
            except:
                logger.error("Model not trained yet and could not load from file")
                return None
        
        try:
            # If this is a single pool dataframe, we need to preprocess
            if 'performance_class' not in df.columns:
                # Use only the features that were used in training
                if self.feature_names is not None:
                    feature_cols = [col for col in self.feature_names if col in df.columns]
                    missing_cols = [col for col in self.feature_names if col not in df.columns]
                    
                    if missing_cols:
                        logger.warning(f"Missing feature columns: {missing_cols}")
                        # Fill missing columns with zeros
                        for col in missing_cols:
                            df[col] = 0
                    
                    X = df[self.feature_names]
                else:
                    # If feature names not saved, use all numeric features
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    X = df[numeric_cols]
            else:
                # Preprocess data
                X, _ = self.preprocess_data(df)
            
            # Make predictions
            prediction_indices = self.model.predict(X)
            predictions = self.label_encoder.inverse_transform(prediction_indices)
            
            # Get probabilities for each class
            probabilities = self.model.predict_proba(X)
            
            # Add predictions to the dataframe
            result_df = df.copy()
            result_df['predicted_performance'] = predictions
            
            # Add probabilities for each class
            for i, class_name in enumerate(self.label_encoder.classes_):
                result_df[f'prob_{class_name}'] = probabilities[:, i]
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None

class RiskAssessmentModel:
    """
    LSTM Neural Network for risk assessment
    """
    
    def __init__(self):
        """Initialize the risk assessment model"""
        self.model = None
        self.feature_names = None
        self.lookback = 24  # 24 hours of data
        self.scaler = StandardScaler()
        self.model_path = os.path.join(MODELS_DIR, 'risk_assessment_model')
        self.version = datetime.now().strftime("%Y%m%d")
    
    def create_sequences(self, X, y=None):
        """Create sequences for LSTM input"""
        sequences_X = []
        sequences_y = []
        
        for i in range(len(X) - self.lookback):
            sequences_X.append(X[i:i+self.lookback])
            if y is not None:
                sequences_y.append(y[i+self.lookback])
        
        return np.array(sequences_X), None if y is None else np.array(sequences_y)
    
    def preprocess_data(self, df):
        """Preprocess data for training/prediction"""
        try:
            # Drop non-feature columns
            drop_cols = ['id', 'pool_id', 'timestamp', 'next_apr', 'performance_class', 
                         'risk_score', 'pool_name']
            feature_cols = [col for col in df.columns if col not in drop_cols 
                            and col != 'risk_score']
            
            # Create feature matrix and target vector
            X = df[feature_cols].values
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            # Check if risk_score exists (for training)
            if 'risk_score' in df.columns:
                y = df['risk_score'].values
            else:
                y = None
            
            # Save feature names
            self.feature_names = feature_cols
            
            # Create sequences for LSTM
            X_seq, y_seq = self.create_sequences(X, y)
            
            return X_seq, y_seq
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def build_model(self, input_shape):
        """Build the model (LSTM if TensorFlow is available, otherwise RandomForest)"""
        if HAS_TENSORFLOW:
            model = Sequential()
            
            # LSTM layers
            model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(50))
            model.add(Dropout(0.2))
            
            # Output layer
            model.add(Dense(1))
            
            # Compile the model
            model.compile(optimizer='adam', loss='mse')
            
            return model
        else:
            # Fallback to RandomForest if TensorFlow is not available
            logger.warning("TensorFlow not available. Using RandomForest as fallback for risk assessment.")
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
    
    def train(self, df):
        """Train the model with the provided data"""
        logger.info("Training risk assessment model")
        
        try:
            # Preprocess data
            X, y = self.preprocess_data(df)
            
            if X is None or y is None:
                logger.error("Empty dataset after preprocessing")
                return False
                
            if HAS_TENSORFLOW:
                # For TensorFlow model
                if X.shape[0] == 0:
                    logger.error("Empty dataset after preprocessing")
                    return False
                
                # Build the model
                input_shape = (X.shape[1], X.shape[2])
                self.model = self.build_model(input_shape)
                
                # Early stopping to prevent overfitting
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                # Train the model
                start_time = time.time()
                history = self.model.fit(
                    X, y,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1
                )
                training_time = time.time() - start_time
                
                # Save the model
                self.model.save(self.model_path)
                
                # Log results
                logger.info(f"Risk assessment model trained in {training_time:.2f} seconds")
                logger.info(f"Final loss: {history.history['loss'][-1]:.4f}")
                logger.info(f"Model saved to {self.model_path}")
            else:
                # For RandomForest fallback
                logger.info("Using RandomForest as fallback for risk assessment model")
                
                # Create feature matrix and target vector differently for RandomForest
                if isinstance(X, np.ndarray) and len(X.shape) == 3:
                    # Flatten the 3D array to 2D for RandomForest
                    n_samples, n_timesteps, n_features = X.shape
                    X_flat = X.reshape(n_samples, n_timesteps * n_features)
                else:
                    X_flat = X
                
                # Build the model
                self.model = self.build_model(None)  # input_shape not needed for RandomForest
                
                # Train the model
                start_time = time.time()
                self.model.fit(X_flat, y)
                training_time = time.time() - start_time
                
                # Save the model using pickle instead of TensorFlow's save method
                with open(f"{self.model_path}.pkl", 'wb') as f:
                    pickle.dump(self.model, f)
                
                # Log results
                logger.info(f"RandomForest risk assessment model trained in {training_time:.2f} seconds")
                logger.info(f"Model saved to {self.model_path}.pkl")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training risk assessment model: {e}")
            return False
    
    def evaluate(self, df):
        """Evaluate the model on test data"""
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Preprocess data
            X, y = self.preprocess_data(df)
            
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            results = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse
            }
            
            logger.info(f"Risk assessment model evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating risk assessment model: {e}")
            return None
    
    def predict(self, df):
        """Make predictions using the trained model"""
        if self.model is None:
            # Try to load the model based on available packages
            try:
                if HAS_TENSORFLOW:
                    from tensorflow.keras.models import load_model
                    self.model = load_model(self.model_path)
                else:
                    # Try to load RandomForest model from pickle
                    try:
                        with open(f"{self.model_path}.pkl", 'rb') as f:
                            self.model = pickle.load(f)
                    except:
                        logger.error("RandomForest model not found")
                        return None
            except Exception as e:
                logger.error(f"Model not trained yet and could not load from file: {e}")
                return None
        
        try:
            # Preprocess data differently based on the model type
            if HAS_TENSORFLOW:
                # We need a sequence of data for the LSTM
                # First make sure we have enough data points
                if len(df) < self.lookback:
                    logger.error(f"Not enough data points for prediction. Need at least {self.lookback} records.")
                    return None
                
                # Preprocess data (sequences will be created)
                X, _ = self.preprocess_data(df)
                
                # Make predictions
                predictions = self.model.predict(X)
                
                # Create a result dataframe
                # The predictions have fewer rows than the input due to sequence creation
                result_df = df.iloc[self.lookback:].copy()
                result_df['predicted_risk_score'] = predictions
            else:
                # For RandomForest, we need to flatten the input data
                # Preprocess data
                X, _ = self.preprocess_data(df)
                
                # Flatten if it's a 3D array
                if isinstance(X, np.ndarray) and len(X.shape) == 3:
                    n_samples, n_timesteps, n_features = X.shape
                    X_flat = X.reshape(n_samples, n_timesteps * n_features)
                else:
                    X_flat = X
                
                # Make predictions
                predictions = self.model.predict(X_flat)
                
                # Create a result dataframe (no sequence offset for RandomForest)
                result_df = df.copy()
                result_df['predicted_risk_score'] = predictions
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None

def train_all_models():
    """Train all models using the latest data"""
    try:
        # Get data for training
        fe = FeatureEngineer()
        db = DBManager()
        
        # Get top pools by liquidity for better data quality
        top_pools = db.get_top_pools_by_liquidity(limit=50)
        
        all_features = []
        
        # Process each pool
        for _, pool in top_pools.iterrows():
            pool_id = pool['pool_id']
            try:
                # Prepare features for this pool
                pool_features = fe.prepare_pool_features(pool_id, days=60)  # Use 60 days of data
                
                if pool_features is not None and not pool_features.empty:
                    # Add pool identifier
                    pool_features['pool_id'] = pool_id
                    pool_features['pool_name'] = pool['name']
                    
                    # Prepare target variables
                    pool_features = fe.prepare_target_variables(pool_features)
                    
                    if pool_features is not None and not pool_features.empty:
                        all_features.append(pool_features)
            except Exception as e:
                logger.error(f"Error processing pool {pool_id} for training: {e}")
        
        if not all_features:
            logger.error("No valid data available for training")
            return False
        
        # Combine all pools' features
        training_data = pd.concat(all_features, ignore_index=True)
        
        # Train APR prediction model
        apr_model = APRPredictionModel()
        apr_model.train(training_data)
        
        # Train pool performance classifier
        perf_model = PoolPerformanceClassifier()
        perf_model.train(training_data)
        
        # Train risk assessment model
        risk_model = RiskAssessmentModel()
        risk_model.train(training_data)
        
        logger.info("All models trained successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return False

def save_predictions_to_db(predictions):
    """Save model predictions to the database"""
    try:
        db = DBManager()
        
        for idx, row in predictions.iterrows():
            db.save_prediction(
                pool_id=row['pool_id'],
                predicted_apr=row['predicted_apr'],
                performance_class=row['predicted_performance'],
                risk_score=row['predicted_risk_score'],
                model_version=datetime.now().strftime("%Y%m%d")
            )
        
        logger.info(f"Saved {len(predictions)} predictions to the database")
        return True
        
    except Exception as e:
        logger.error(f"Error saving predictions to database: {e}")
        return False

def generate_predictions():
    """Generate predictions using trained models"""
    try:
        # Load models
        apr_model = APRPredictionModel()
        perf_model = PoolPerformanceClassifier()
        risk_model = RiskAssessmentModel()
        
        # Get data for prediction
        fe = FeatureEngineer()
        db = DBManager()
        
        # Get top pools
        top_pools = db.get_top_pools_by_liquidity(limit=50)
        
        all_predictions = []
        
        # Process each pool
        for _, pool in top_pools.iterrows():
            pool_id = pool['pool_id']
            try:
                # Prepare features for this pool
                pool_features = fe.prepare_pool_features(pool_id, days=30)
                
                if pool_features is not None and not pool_features.empty:
                    # Add pool identifier
                    pool_features['pool_id'] = pool_id
                    pool_features['pool_name'] = pool['name']
                    
                    # Make predictions with each model
                    apr_predictions = apr_model.predict(pool_features)
                    
                    if apr_predictions is not None:
                        perf_predictions = perf_model.predict(apr_predictions)
                        
                        if perf_predictions is not None:
                            risk_predictions = risk_model.predict(perf_predictions)
                            
                            if risk_predictions is not None:
                                # Keep the most recent prediction for each pool
                                latest_prediction = risk_predictions.iloc[-1]
                                all_predictions.append(latest_prediction)
            except Exception as e:
                logger.error(f"Error generating predictions for pool {pool_id}: {e}")
        
        if not all_predictions:
            logger.error("No valid predictions generated")
            return None
        
        # Combine all predictions
        predictions_df = pd.DataFrame(all_predictions)
        
        # Save predictions to database
        save_predictions_to_db(predictions_df)
        
        return predictions_df
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return None

if __name__ == "__main__":
    # Train all models
    train_all_models()
    
    # Generate predictions
    predictions = generate_predictions()
    
    if predictions is not None:
        print(f"Generated predictions for {len(predictions)} pools")
