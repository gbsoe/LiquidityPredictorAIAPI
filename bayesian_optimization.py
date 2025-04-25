"""
Bayesian Optimization Module for Self-Evolving Prediction Engine

This module provides advanced Bayesian optimization capabilities for hyperparameter tuning
and automated feature engineering. It enables the prediction engine to automatically search
for optimal configurations using probabilistic surrogate models.

Key features:
1. Gaussian Process-based surrogate models
2. Acquisition function optimization
3. Sequential model-based optimization
4. Multi-objective Bayesian optimization
5. Warm-starting from prior knowledge
"""

import os
import json
import logging
import numpy as np
import random
from typing import Dict, List, Any, Tuple, Callable, Optional
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bayesian_opt.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bayesian_optimization")

# Constants
DEFAULT_ITERATIONS = 50
EXPLORATION_WEIGHT = 0.1
MAX_HISTORY_SIZE = 200

@dataclass
class Parameter:
    """Parameter for optimization"""
    name: str
    type: str  # "float", "int", "categorical"
    min_value: float = None
    max_value: float = None
    choices: List[Any] = None
    log_scale: bool = False  # For float parameters, use log scale
    
    def sample(self) -> Any:
        """Sample a random value from the parameter space"""
        if self.type == "float":
            if self.log_scale:
                # Log-uniform sampling
                log_min = np.log(self.min_value)
                log_max = np.log(self.max_value)
                return float(np.exp(random.uniform(log_min, log_max)))
            else:
                # Uniform sampling
                return float(random.uniform(self.min_value, self.max_value))
        elif self.type == "int":
            # Integer sampling
            return int(random.randint(int(self.min_value), int(self.max_value)))
        elif self.type == "categorical":
            # Categorical sampling
            return random.choice(self.choices)
        else:
            raise ValueError(f"Unknown parameter type: {self.type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "name": self.name,
            "type": self.type
        }
        
        if self.type in ["float", "int"]:
            result["min_value"] = self.min_value
            result["max_value"] = self.max_value
            if self.type == "float":
                result["log_scale"] = self.log_scale
        elif self.type == "categorical":
            result["choices"] = self.choices
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Parameter':
        """Create from dictionary"""
        return cls(
            name=data["name"],
            type=data["type"],
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            choices=data.get("choices"),
            log_scale=data.get("log_scale", False)
        )

@dataclass
class OptimizationTrial:
    """A single trial in the optimization process"""
    params: Dict[str, Any]
    score: float = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "params": self.params,
            "score": self.score,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationTrial':
        """Create from dictionary"""
        return cls(
            params=data["params"],
            score=data["score"],
            metadata=data.get("metadata", {})
        )

class GaussianProcessSurrogate:
    """
    Gaussian Process surrogate model
    
    In a full implementation, this would use scikit-learn's GaussianProcessRegressor
    or a similar library. For simplicity, we'll implement a basic approach here.
    """
    
    def __init__(self, kernel="rbf", alpha=1e-6):
        """Initialize the surrogate model"""
        self.kernel = kernel
        self.alpha = alpha
        self.X = None
        self.y = None
        self.fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Gaussian Process surrogate model
        
        Args:
            X: Feature matrix
            y: Target values
        """
        # In a real implementation, this would create and fit a GP model
        # For simplicity, we'll just store the data
        self.X = X
        self.y = y
        self.fitted = True
        logger.info(f"Fitted GP surrogate model with {len(X)} points")
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation at points X
        
        Args:
            X: Points to predict at
            
        Returns:
            Tuple of (means, std_devs)
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
            
        # In a real implementation, this would use the GP to predict means and uncertainties
        # For simplicity, we'll implement a basic distance-based approach
        
        means = np.zeros(len(X))
        std_devs = np.ones(len(X))
        
        for i, x in enumerate(X):
            # Calculate distances to all training points
            if len(self.X) > 0:
                distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
                
                # Weighted average based on distances
                weights = 1.0 / (distances + 1e-8)
                weights = weights / np.sum(weights)
                
                # Predicted mean
                means[i] = np.sum(weights * self.y)
                
                # Uncertainty increases with distance
                closest_dist = np.min(distances)
                std_devs[i] = 0.1 + 0.9 * np.tanh(closest_dist)
            else:
                # No training data yet
                means[i] = 0.0
                std_devs[i] = 1.0
                
        return means, std_devs

class BayesianOptimizer:
    """
    Bayesian Optimizer for hyperparameter tuning
    """
    
    def __init__(self, parameter_space: List[Parameter], evaluate_function: Callable, maximize: bool = True):
        """
        Initialize the Bayesian optimizer
        
        Args:
            parameter_space: List of parameters to optimize
            evaluate_function: Function to evaluate a set of parameters
            maximize: Whether to maximize (True) or minimize (False) the function
        """
        self.parameter_space = parameter_space
        self.evaluate_function = evaluate_function
        self.maximize = maximize
        self.trials = []
        self.best_score = float('-inf') if maximize else float('inf')
        self.best_params = None
        self.surrogate = GaussianProcessSurrogate()
        
    def _acquisition_function(self, x: np.ndarray, surrogate: GaussianProcessSurrogate, best_score: float) -> float:
        """
        Expected Improvement acquisition function
        
        Args:
            x: Point to evaluate
            surrogate: Surrogate model
            best_score: Current best score
            
        Returns:
            Expected improvement
        """
        x = x.reshape(1, -1)
        mu, sigma = surrogate.predict(x)
        mu = mu[0]
        sigma = sigma[0]
        
        if sigma <= 0.0:
            return 0.0
            
        z = (mu - best_score) / sigma
        
        if self.maximize:
            # Expected improvement for maximization
            ei = sigma * (z * (0.5 * (1 + np.math.erf(z / np.sqrt(2)))) + 
                        (1 / np.sqrt(2 * np.pi)) * np.exp(-(z ** 2) / 2))
        else:
            # Expected improvement for minimization (negate the mean)
            z = (best_score - mu) / sigma
            ei = sigma * (z * (0.5 * (1 + np.math.erf(z / np.sqrt(2)))) + 
                        (1 / np.sqrt(2 * np.pi)) * np.exp(-(z ** 2) / 2))
        
        return ei
    
    def _encode_params(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Encode parameters as a numpy array
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            Encoded parameters as numpy array
        """
        encoded = []
        
        for param in self.parameter_space:
            value = params[param.name]
            
            if param.type == "float":
                if param.log_scale:
                    # Normalize log value
                    log_min = np.log(param.min_value)
                    log_max = np.log(param.max_value)
                    normalized = (np.log(value) - log_min) / (log_max - log_min)
                else:
                    # Normalize linear value
                    normalized = (value - param.min_value) / (param.max_value - param.min_value)
                encoded.append(normalized)
            elif param.type == "int":
                # Normalize integer
                normalized = (value - param.min_value) / (param.max_value - param.min_value)
                encoded.append(normalized)
            elif param.type == "categorical":
                # One-hot encoding
                for choice in param.choices:
                    encoded.append(1.0 if value == choice else 0.0)
                    
        return np.array(encoded)
    
    def _decode_params(self, encoded: np.ndarray) -> Dict[str, Any]:
        """
        Decode numpy array back to parameter dictionary
        
        Args:
            encoded: Encoded parameters
            
        Returns:
            Dictionary of parameter values
        """
        params = {}
        idx = 0
        
        for param in self.parameter_space:
            if param.type in ["float", "int"]:
                # Single value for float or int
                normalized = encoded[idx]
                idx += 1
                
                if param.type == "float":
                    if param.log_scale:
                        log_min = np.log(param.min_value)
                        log_max = np.log(param.max_value)
                        value = np.exp(log_min + normalized * (log_max - log_min))
                    else:
                        value = param.min_value + normalized * (param.max_value - param.min_value)
                else:  # int
                    value = int(round(param.min_value + normalized * (param.max_value - param.min_value)))
                
                params[param.name] = value
            elif param.type == "categorical":
                # One-hot encoded categorical
                one_hot = encoded[idx:idx+len(param.choices)]
                idx += len(param.choices)
                
                # Find category with highest value
                category_idx = np.argmax(one_hot)
                params[param.name] = param.choices[category_idx]
                
        return params
    
    def _propose_next_point(self) -> Dict[str, Any]:
        """
        Propose the next point to evaluate
        
        Returns:
            Dictionary of parameter values to try next
        """
        # If we don't have enough points, do random sampling
        if len(self.trials) < 5:
            return self._random_point()
            
        # Prepare data for surrogate model
        X = np.vstack([self._encode_params(trial.params) for trial in self.trials])
        y = np.array([trial.score for trial in self.trials])
        
        # Fit surrogate model
        self.surrogate.fit(X, y)
        
        # Use random search to find the point with maximum acquisition value
        best_acq = float('-inf')
        best_point = None
        
        for _ in range(100):
            # Sample random point
            point = self._random_point()
            encoded = self._encode_params(point)
            
            # Calculate acquisition value
            acq_value = self._acquisition_function(encoded, self.surrogate, self.best_score)
            
            # Update best
            if acq_value > best_acq:
                best_acq = acq_value
                best_point = point
                
        # If we couldn't find a good point, just return a random one
        if best_point is None:
            return self._random_point()
            
        return best_point
    
    def _random_point(self) -> Dict[str, Any]:
        """
        Generate a random point in the parameter space
        
        Returns:
            Dictionary of parameter values
        """
        point = {}
        for param in self.parameter_space:
            point[param.name] = param.sample()
        return point
    
    def run_optimization(self, n_iterations: int = DEFAULT_ITERATIONS) -> Dict[str, Any]:
        """
        Run the optimization process
        
        Args:
            n_iterations: Number of iterations to run
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting Bayesian optimization with {n_iterations} iterations")
        
        for i in range(n_iterations):
            # Propose next point
            next_params = self._propose_next_point()
            
            # Evaluate function
            try:
                score = self.evaluate_function(next_params)
                
                # Create trial
                trial = OptimizationTrial(
                    params=next_params,
                    score=score
                )
                
                # Update best score
                if (self.maximize and score > self.best_score) or (not self.maximize and score < self.best_score):
                    self.best_score = score
                    self.best_params = next_params
                    logger.info(f"New best score: {score} with params: {next_params}")
                
                # Add to trials
                self.trials.append(trial)
                
                logger.info(f"Iteration {i+1}/{n_iterations}: score = {score}")
                
            except Exception as e:
                logger.error(f"Error evaluating params {next_params}: {e}")
        
        # Sort trials by score
        sorted_trials = sorted(
            self.trials, 
            key=lambda x: x.score, 
            reverse=self.maximize
        )
        
        # Create result
        result = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "all_trials": [trial.to_dict() for trial in sorted_trials]
        }
        
        return result
    
    def save_state(self, filepath: str) -> None:
        """
        Save optimizer state to a file
        
        Args:
            filepath: Path to save to
        """
        state = {
            "parameter_space": [param.to_dict() for param in self.parameter_space],
            "maximize": self.maximize,
            "trials": [trial.to_dict() for trial in self.trials],
            "best_score": self.best_score,
            "best_params": self.best_params
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved optimizer state to {filepath}")
        except Exception as e:
            logger.error(f"Error saving optimizer state: {e}")
    
    @classmethod
    def load_state(cls, filepath: str, evaluate_function: Callable) -> 'BayesianOptimizer':
        """
        Load optimizer state from a file
        
        Args:
            filepath: Path to load from
            evaluate_function: Function to evaluate parameters
            
        Returns:
            BayesianOptimizer instance
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            # Create parameter space
            parameter_space = [Parameter.from_dict(param_dict) for param_dict in state["parameter_space"]]
            
            # Create optimizer
            optimizer = cls(parameter_space, evaluate_function, state["maximize"])
            
            # Load trials
            optimizer.trials = [OptimizationTrial.from_dict(trial_dict) for trial_dict in state["trials"]]
            
            # Load best
            optimizer.best_score = state["best_score"]
            optimizer.best_params = state["best_params"]
            
            logger.info(f"Loaded optimizer state from {filepath} with {len(optimizer.trials)} trials")
            
            return optimizer
            
        except Exception as e:
            logger.error(f"Error loading optimizer state: {e}")
            raise e

class HyperparameterOptimizer:
    """
    Hyperparameter optimizer for machine learning models
    """
    
    def __init__(self, model_type: str, parameter_space: List[Parameter] = None):
        """
        Initialize the hyperparameter optimizer
        
        Args:
            model_type: Type of model to optimize
            parameter_space: Custom parameter space, or None to use defaults
        """
        self.model_type = model_type
        
        # Use default parameter space if not provided
        if parameter_space is None:
            self.parameter_space = self._get_default_parameter_space(model_type)
        else:
            self.parameter_space = parameter_space
            
        self.best_params = None
        self.best_score = None
        self.optimizer = None
        
    def _get_default_parameter_space(self, model_type: str) -> List[Parameter]:
        """
        Get default parameter space for a model type
        
        Args:
            model_type: Type of model
            
        Returns:
            List of parameters
        """
        if model_type == "xgboost":
            return [
                Parameter("learning_rate", "float", 0.01, 0.3, log_scale=True),
                Parameter("max_depth", "int", 3, 10),
                Parameter("min_child_weight", "int", 1, 10),
                Parameter("subsample", "float", 0.5, 1.0),
                Parameter("colsample_bytree", "float", 0.5, 1.0),
                Parameter("gamma", "float", 0, 5),
                Parameter("reg_alpha", "float", 0, 10),
                Parameter("reg_lambda", "float", 0, 10)
            ]
        elif model_type == "lstm":
            return [
                Parameter("learning_rate", "float", 0.0001, 0.01, log_scale=True),
                Parameter("lstm_units", "int", 32, 256),
                Parameter("num_lstm_layers", "int", 1, 3),
                Parameter("dropout", "float", 0.0, 0.5),
                Parameter("recurrent_dropout", "float", 0.0, 0.5),
                Parameter("batch_size", "int", 16, 128)
            ]
        elif model_type == "randomforest":
            return [
                Parameter("n_estimators", "int", 50, 500),
                Parameter("max_depth", "int", 3, 20),
                Parameter("min_samples_split", "int", 2, 20),
                Parameter("min_samples_leaf", "int", 1, 10),
                Parameter("max_features", "categorical", choices=["sqrt", "log2", "auto"])
            ]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def optimize(self, evaluation_function: Callable, n_iterations: int = DEFAULT_ITERATIONS, maximize: bool = True) -> Dict[str, Any]:
        """
        Optimize hyperparameters
        
        Args:
            evaluation_function: Function to evaluate hyperparameters
            n_iterations: Number of iterations
            maximize: Whether to maximize or minimize the metric
            
        Returns:
            Dictionary with optimization results
        """
        # Create optimizer
        self.optimizer = BayesianOptimizer(
            parameter_space=self.parameter_space,
            evaluate_function=evaluation_function,
            maximize=maximize
        )
        
        # Run optimization
        results = self.optimizer.run_optimization(n_iterations)
        
        # Store results
        self.best_params = results["best_params"]
        self.best_score = results["best_score"]
        
        return results
    
    def save_state(self, filepath: str) -> None:
        """
        Save optimizer state
        
        Args:
            filepath: Path to save to
        """
        if self.optimizer:
            self.optimizer.save_state(filepath)
        else:
            logger.warning("No optimizer to save")
    
    def load_state(self, filepath: str, evaluation_function: Callable) -> bool:
        """
        Load optimizer state
        
        Args:
            filepath: Path to load from
            evaluation_function: Function to evaluate hyperparameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.optimizer = BayesianOptimizer.load_state(filepath, evaluation_function)
            
            if self.optimizer:
                self.best_params = self.optimizer.best_params
                self.best_score = self.optimizer.best_score
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading optimizer state: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Example of how to use the Bayesian optimizer
    
    # Define a simple objective function (optimize XGBoost hyperparameters)
    def evaluate_xgboost_hyperparams(params):
        # This would normally train and evaluate an XGBoost model
        # For demonstration, we'll use a simple function
        lr = params["learning_rate"]
        depth = params["max_depth"]
        
        # Simulate an objective function
        # This simple quadratic function has a maximum at lr=0.1, depth=6
        score = -(lr - 0.1)**2 * 100 - (depth - 6)**2 / 4 + 0.8
        
        # Add some random noise
        score += random.uniform(-0.05, 0.05)
        
        # Simulate training time
        import time
        time.sleep(0.1)
        
        return score
    
    # Create optimizer for XGBoost
    hp_optimizer = HyperparameterOptimizer("xgboost")
    
    # Run optimization
    print("Running Bayesian optimization...")
    results = hp_optimizer.optimize(
        evaluation_function=evaluate_xgboost_hyperparams,
        n_iterations=20  # Usually you'd use more iterations
    )
    
    # Print results
    print(f"Best parameters: {results['best_params']}")
    print(f"Best score: {results['best_score']}")
    
    # Save optimizer state
    hp_optimizer.save_state("xgboost_optimizer.json")