import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('model_utils')

# Define constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def load_model(model_path):
    """
    Load a saved model from disk
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None

def evaluate_regression_model(y_true, y_pred, model_name="Regression Model"):
    """
    Evaluate a regression model and print metrics
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        logger.info(f"{model_name} Evaluation:")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R²: {r2:.4f}")
        
        results = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        return results
    except Exception as e:
        logger.error(f"Error evaluating regression model: {e}")
        return None

def evaluate_classification_model(y_true, y_pred, labels=None, model_name="Classification Model"):
    """
    Evaluate a classification model and print metrics
    """
    try:
        # Generate report
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        
        logger.info(f"{model_name} Evaluation:")
        logger.info(f"Accuracy: {report['accuracy']:.4f}")
        
        # Log class-specific metrics
        for cls in report:
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                logger.info(f"Class {cls} - Precision: {report[cls]['precision']:.4f}, "
                           f"Recall: {report[cls]['recall']:.4f}, "
                           f"F1-Score: {report[cls]['f1-score']:.4f}")
        
        return report
    except Exception as e:
        logger.error(f"Error evaluating classification model: {e}")
        return None

def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    """
    Plot feature importance for a tree-based model
    """
    try:
        # Extract feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            logger.error("Model doesn't have feature_importances_ or coef_ attribute")
            return None
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance and take top N
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(title)
        plt.tight_layout()
        
        # Return the DataFrame for further analysis
        return importance_df
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")
        return None

def plot_prediction_vs_actual(y_true, y_pred, model_name="Model", max_points=500):
    """
    Plot predicted vs actual values for regression models
    """
    try:
        # Create DataFrame for plotting
        if len(y_true) > max_points:
            # Sample points for large datasets
            indices = np.random.choice(len(y_true), max_points, replace=False)
            y_true_sample = y_true[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred
        
        plot_df = pd.DataFrame({
            'Actual': y_true_sample,
            'Predicted': y_pred_sample
        })
        
        # Calculate metrics for the plot
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Create the scatter plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Actual', y='Predicted', data=plot_df, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
        max_val = max(plot_df['Actual'].max(), plot_df['Predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f"{model_name}: Predicted vs Actual\nRMSE={rmse:.4f}, R²={r2:.4f}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.tight_layout()
        
        return plot_df
    except Exception as e:
        logger.error(f"Error plotting prediction vs actual: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred, labels=None, model_name="Classification Model"):
    """
    Plot confusion matrix for classification models
    """
    try:
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Create DataFrame for nicer display
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))
        
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{model_name}: Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        
        return cm_df
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        return None

def plot_time_series_prediction(timestamps, y_true, y_pred, model_name="Time Series Model"):
    """
    Plot time series data with predictions
    """
    try:
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        # Sort by timestamp
        plot_df = plot_df.sort_values('Timestamp')
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(plot_df['Timestamp'], plot_df['Actual'], 'b-', label='Actual')
        plt.plot(plot_df['Timestamp'], plot_df['Predicted'], 'r--', label='Predicted')
        plt.title(f"{model_name}: Time Series Prediction")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plot_df
    except Exception as e:
        logger.error(f"Error plotting time series prediction: {e}")
        return None

def prepare_prediction_report(pool_id, predictions, actual=None):
    """
    Prepare a prediction report for a specific pool
    """
    try:
        report = {
            'pool_id': pool_id,
            'prediction_time': datetime.now().isoformat(),
            'predictions': {
                'apr': float(predictions.get('predicted_apr', 0)),
                'performance_class': predictions.get('predicted_performance', 'unknown'),
                'risk_score': float(predictions.get('predicted_risk_score', 0))
            },
            'actual_values': {}
        }
        
        # Add actual values if available
        if actual is not None:
            report['actual_values'] = {
                'apr': float(actual.get('apr', 0)),
                'performance_class': actual.get('performance_class', 'unknown'),
                'risk_score': float(actual.get('risk_score', 0))
            }
        
        return report
    except Exception as e:
        logger.error(f"Error preparing prediction report: {e}")
        return None

def analyze_model_performance_over_time(predictions_df, target_col, pred_col):
    """
    Analyze how model performance changes over time
    """
    try:
        # Ensure DataFrame has timestamp column
        if 'timestamp' not in predictions_df.columns:
            logger.error("DataFrame must have a 'timestamp' column")
            return None
        
        # Convert to datetime if needed
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        
        # Group by time period (e.g., day)
        predictions_df['date'] = predictions_df['timestamp'].dt.date
        
        # Calculate error metrics for each day
        daily_metrics = []
        
        for date, group in predictions_df.groupby('date'):
            if group[target_col].count() < 5:  # Skip days with few samples
                continue
                
            mae = mean_absolute_error(group[target_col], group[pred_col])
            rmse = np.sqrt(mean_squared_error(group[target_col], group[pred_col]))
            
            daily_metrics.append({
                'date': date,
                'sample_count': len(group),
                'mae': mae,
                'rmse': rmse
            })
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(daily_metrics)
        
        # Plot metrics over time
        if not metrics_df.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(metrics_df['date'], metrics_df['rmse'], 'b-', label='RMSE')
            plt.plot(metrics_df['date'], metrics_df['mae'], 'r--', label='MAE')
            plt.title("Model Performance Over Time")
            plt.xlabel("Date")
            plt.ylabel("Error")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
        
        return metrics_df
    except Exception as e:
        logger.error(f"Error analyzing model performance over time: {e}")
        return None

def get_model_version_info(model_path):
    """
    Get version information for a saved model
    """
    try:
        # Check if the file exists
        if not os.path.exists(model_path):
            return {
                'exists': False,
                'message': 'Model file does not exist'
            }
        
        # Get file modification time
        mod_time = os.path.getmtime(model_path)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # Get file size
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        
        # Try to load the model to check if it's valid
        try:
            model = load_model(model_path)
            is_valid = model is not None
        except:
            is_valid = False
        
        return {
            'exists': True,
            'path': model_path,
            'last_modified': mod_time_str,
            'size_mb': f"{size_mb:.2f} MB",
            'is_valid': is_valid
        }
    except Exception as e:
        logger.error(f"Error getting model version info: {e}")
        return {
            'exists': False,
            'error': str(e)
        }
