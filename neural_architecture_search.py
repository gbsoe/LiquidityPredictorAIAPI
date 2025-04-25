"""
Neural Architecture Search for Self-Evolving Prediction Engine

This module implements automated neural architecture search to discover optimal
neural network architectures for prediction tasks. Rather than relying on manually
designed networks, the system can automatically explore the architecture space and
discover designs that are specifically optimized for liquidity pool prediction tasks.

Key features:
1. Efficient neural architecture search using evolutionary algorithms
2. Parameter sharing across candidate architectures
3. Progressive architecture growth
4. Multi-objective optimization balancing accuracy and complexity
5. Transfer learning from previously discovered architectures
"""

import os
import json
import logging
import random
import pickle
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Callable, Optional, Union
from dataclasses import dataclass, field
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nas.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("neural_architecture_search")

# Constants
MAX_LAYERS = 10
MAX_UNITS_PER_LAYER = 512
MIN_UNITS_PER_LAYER = 4
MAX_POPULATION_SIZE = 50
DEFAULT_MUTATION_RATE = 0.2
MAX_ARCHITECTURE_HISTORY = 100
DEFAULT_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

@dataclass
class LayerSpec:
    """Specification for a neural network layer"""
    layer_type: str  # dense, lstm, gru, conv1d, etc.
    units: int = 64
    activation: str = "relu"
    dropout_rate: float = 0.0
    kernel_size: int = None  # For convolutional layers
    l2_regularization: float = 0.0
    batch_normalization: bool = False
    layer_normalization: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "layer_type": self.layer_type,
            "units": self.units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "kernel_size": self.kernel_size,
            "l2_regularization": self.l2_regularization,
            "batch_normalization": self.batch_normalization,
            "layer_normalization": self.layer_normalization
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerSpec':
        """Create from dictionary"""
        return cls(
            layer_type=data["layer_type"],
            units=data.get("units", 64),
            activation=data.get("activation", "relu"),
            dropout_rate=data.get("dropout_rate", 0.0),
            kernel_size=data.get("kernel_size"),
            l2_regularization=data.get("l2_regularization", 0.0),
            batch_normalization=data.get("batch_normalization", False),
            layer_normalization=data.get("layer_normalization", False)
        )
    
    @classmethod
    def random(cls, layer_type: str = None) -> 'LayerSpec':
        """Create a random layer specification"""
        # Choose random layer type if not provided
        if layer_type is None:
            layer_type = random.choice(["dense", "lstm", "gru"])
            
        # Choose random units (power of 2)
        units_power = random.randint(2, 9)  # 4 to 512
        units = 2 ** units_power
        
        # Choose random activation
        activation = random.choice(["relu", "tanh", "sigmoid", "elu", "selu"])
        
        # Choose random dropout
        dropout_rate = random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        
        # L2 regularization
        l2_options = [0.0, 0.001, 0.01, 0.1]
        l2_regularization = random.choice(l2_options)
        
        # Normalization
        batch_normalization = random.choice([True, False])
        layer_normalization = random.choice([True, False]) if not batch_normalization else False
        
        # Kernel size for convolutional layers
        kernel_size = None
        if layer_type.startswith("conv"):
            kernel_size = random.choice([3, 5, 7, 9])
            
        return cls(
            layer_type=layer_type,
            units=units,
            activation=activation,
            dropout_rate=dropout_rate,
            kernel_size=kernel_size,
            l2_regularization=l2_regularization,
            batch_normalization=batch_normalization,
            layer_normalization=layer_normalization
        )
    
    def mutate(self, mutation_rate: float = DEFAULT_MUTATION_RATE) -> 'LayerSpec':
        """
        Create a mutated copy of this layer spec
        
        Args:
            mutation_rate: Probability of mutating each parameter
            
        Returns:
            Mutated layer spec
        """
        layer_spec = LayerSpec(
            layer_type=self.layer_type,
            units=self.units,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            kernel_size=self.kernel_size,
            l2_regularization=self.l2_regularization,
            batch_normalization=self.batch_normalization,
            layer_normalization=self.layer_normalization
        )
        
        # Potentially mutate each parameter
        if random.random() < mutation_rate:
            # Mutate units (with 50% probability)
            if random.random() < 0.5:
                # Small adjustment
                factor = random.uniform(0.75, 1.5)
                new_units = int(self.units * factor)
                # Ensure within bounds and power of 2
                power = max(2, min(9, round(np.log2(new_units))))
                layer_spec.units = 2 ** power
            
        if random.random() < mutation_rate:
            # Mutate activation
            layer_spec.activation = random.choice(["relu", "tanh", "sigmoid", "elu", "selu"])
            
        if random.random() < mutation_rate:
            # Mutate dropout
            dropout_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            layer_spec.dropout_rate = random.choice(dropout_options)
            
        if random.random() < mutation_rate:
            # Mutate L2 regularization
            l2_options = [0.0, 0.001, 0.01, 0.1]
            layer_spec.l2_regularization = random.choice(l2_options)
            
        if random.random() < mutation_rate:
            # Mutate normalization
            if random.random() < 0.5:
                layer_spec.batch_normalization = not self.batch_normalization
                if layer_spec.batch_normalization:
                    layer_spec.layer_normalization = False
            else:
                layer_spec.layer_normalization = not self.layer_normalization
                if layer_spec.layer_normalization:
                    layer_spec.batch_normalization = False
                    
        if self.layer_type.startswith("conv") and random.random() < mutation_rate:
            # Mutate kernel size
            layer_spec.kernel_size = random.choice([3, 5, 7, 9])
            
        return layer_spec

@dataclass
class LearningConfig:
    """Configuration for learning algorithm"""
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "adam"
    loss_function: str = "mse"
    epochs: int = DEFAULT_EPOCHS
    early_stopping: bool = True
    patience: int = EARLY_STOPPING_PATIENCE
    learning_rate_schedule: str = None  # None, "reduce_on_plateau", "cosine_decay"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "loss_function": self.loss_function,
            "epochs": self.epochs,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "learning_rate_schedule": self.learning_rate_schedule
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningConfig':
        """Create from dictionary"""
        return cls(
            learning_rate=data.get("learning_rate", 0.001),
            batch_size=data.get("batch_size", 32),
            optimizer=data.get("optimizer", "adam"),
            loss_function=data.get("loss_function", "mse"),
            epochs=data.get("epochs", DEFAULT_EPOCHS),
            early_stopping=data.get("early_stopping", True),
            patience=data.get("patience", EARLY_STOPPING_PATIENCE),
            learning_rate_schedule=data.get("learning_rate_schedule")
        )
    
    @classmethod
    def random(cls) -> 'LearningConfig':
        """Create a random learning configuration"""
        learning_rate = 10 ** random.uniform(-4, -2)  # 0.0001 to 0.01
        batch_size = 2 ** random.randint(4, 8)  # 16 to 256
        optimizer = random.choice(["adam", "sgd", "rmsprop", "adamw"])
        loss_function = random.choice(["mse", "mae", "huber"])
        early_stopping = random.choice([True, False])
        patience = random.randint(3, 10)
        lr_schedule = random.choice([None, "reduce_on_plateau", "cosine_decay"])
        
        return cls(
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer=optimizer,
            loss_function=loss_function,
            early_stopping=early_stopping,
            patience=patience,
            learning_rate_schedule=lr_schedule
        )
    
    def mutate(self, mutation_rate: float = DEFAULT_MUTATION_RATE) -> 'LearningConfig':
        """
        Create a mutated copy of this learning config
        
        Args:
            mutation_rate: Probability of mutating each parameter
            
        Returns:
            Mutated learning config
        """
        config = LearningConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            loss_function=self.loss_function,
            epochs=self.epochs,
            early_stopping=self.early_stopping,
            patience=self.patience,
            learning_rate_schedule=self.learning_rate_schedule
        )
        
        # Potentially mutate each parameter
        if random.random() < mutation_rate:
            # Mutate learning rate (small adjustment)
            factor = random.uniform(0.5, 2.0)
            config.learning_rate = self.learning_rate * factor
            # Keep within reasonable bounds
            config.learning_rate = max(1e-5, min(1e-1, config.learning_rate))
            
        if random.random() < mutation_rate:
            # Mutate batch size
            factor = 2 ** random.choice([-1, 0, 1])  # Half, same, or double
            config.batch_size = int(self.batch_size * factor)
            # Keep within reasonable bounds
            config.batch_size = max(8, min(512, config.batch_size))
            
        if random.random() < mutation_rate:
            # Mutate optimizer
            config.optimizer = random.choice(["adam", "sgd", "rmsprop", "adamw"])
            
        if random.random() < mutation_rate:
            # Mutate loss function
            config.loss_function = random.choice(["mse", "mae", "huber"])
            
        if random.random() < mutation_rate:
            # Mutate early stopping
            config.early_stopping = not self.early_stopping
            
        if random.random() < mutation_rate:
            # Mutate patience
            config.patience = max(2, self.patience + random.choice([-2, -1, 1, 2]))
            
        if random.random() < mutation_rate:
            # Mutate learning rate schedule
            config.learning_rate_schedule = random.choice([None, "reduce_on_plateau", "cosine_decay"])
            
        return config

@dataclass
class NeuralArchitecture:
    """Specification for a complete neural network architecture"""
    input_shape: Tuple[int, ...] = None
    layers: List[LayerSpec] = field(default_factory=list)
    learning_config: LearningConfig = field(default_factory=LearningConfig)
    architecture_id: str = None
    created_at: datetime = field(default_factory=datetime.now)
    performance: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.architecture_id is None:
            self.architecture_id = f"arch_{int(time.time())}_{random.randint(1000, 9999)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "input_shape": self.input_shape,
            "layers": [layer.to_dict() for layer in self.layers],
            "learning_config": self.learning_config.to_dict(),
            "architecture_id": self.architecture_id,
            "created_at": self.created_at.isoformat(),
            "performance": self.performance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralArchitecture':
        """Create from dictionary"""
        arch = cls(
            input_shape=data.get("input_shape"),
            layers=[LayerSpec.from_dict(layer) for layer in data.get("layers", [])],
            learning_config=LearningConfig.from_dict(data.get("learning_config", {})),
            architecture_id=data.get("architecture_id")
        )
        
        arch.created_at = datetime.fromisoformat(data["created_at"])
        arch.performance = data.get("performance", {})
        
        return arch
    
    @classmethod
    def random(cls, input_shape: Tuple[int, ...], min_layers: int = 2, max_layers: int = 6) -> 'NeuralArchitecture':
        """
        Create a random neural architecture
        
        Args:
            input_shape: Input shape for the network
            min_layers: Minimum number of layers
            max_layers: Maximum number of layers
            
        Returns:
            Random neural architecture
        """
        # Decide number of layers
        num_layers = random.randint(min_layers, max_layers)
        
        # Create layers
        layers = []
        
        # First layer should be suited for the input shape
        if len(input_shape) > 1:
            # Sequential data or images - use appropriate layer
            if len(input_shape) == 2 and input_shape[1] > 1:
                # Time series data
                first_layer_type = random.choice(["lstm", "gru"])
            else:
                # Default to dense
                first_layer_type = "dense"
        else:
            # Simple vector input - use dense layer
            first_layer_type = "dense"
            
        layers.append(LayerSpec.random(first_layer_type))
        
        # Add remaining layers
        for i in range(1, num_layers):
            # Last layer should be dense
            if i == num_layers - 1:
                layer_type = "dense"
            else:
                # Middle layers can be various types
                layer_type = random.choice(["dense", "dense", "lstm", "gru"])
                
            layers.append(LayerSpec.random(layer_type))
        
        # Create learning config
        learning_config = LearningConfig.random()
        
        return cls(
            input_shape=input_shape,
            layers=layers,
            learning_config=learning_config
        )
    
    def mutate(self, mutation_rate: float = DEFAULT_MUTATION_RATE) -> 'NeuralArchitecture':
        """
        Create a mutated copy of this architecture
        
        Args:
            mutation_rate: Probability of mutating each component
            
        Returns:
            Mutated architecture
        """
        # Create a new instance
        arch = NeuralArchitecture(
            input_shape=self.input_shape,
            architecture_id=f"arch_{int(time.time())}_{random.randint(1000, 9999)}"
        )
        
        # Mutate learning config
        arch.learning_config = self.learning_config.mutate(mutation_rate)
        
        # Decide whether to add, remove, or keep layer count
        layer_op = random.random()
        
        if layer_op < 0.2 and len(self.layers) > 2:
            # Remove a layer (don't remove first or last layer)
            idx_to_remove = random.randint(1, len(self.layers) - 2)
            new_layers = self.layers.copy()
            new_layers.pop(idx_to_remove)
            arch.layers = [layer.mutate(mutation_rate) for layer in new_layers]
            
        elif layer_op < 0.4 and len(self.layers) < MAX_LAYERS:
            # Add a layer (not at first or last position)
            insert_pos = random.randint(1, len(self.layers) - 1)
            new_layer = LayerSpec.random()
            
            new_layers = self.layers.copy()
            new_layers.insert(insert_pos, new_layer)
            
            # Mutate existing layers
            arch.layers = [layer.mutate(mutation_rate) for layer in new_layers]
            
        else:
            # Keep same number of layers, just mutate them
            arch.layers = [layer.mutate(mutation_rate) for layer in self.layers]
        
        return arch
    
    def cross(self, other: 'NeuralArchitecture') -> 'NeuralArchitecture':
        """
        Create a child architecture by crossing with another architecture
        
        Args:
            other: Another neural architecture
            
        Returns:
            Child architecture
        """
        # Create a new instance
        child = NeuralArchitecture(
            input_shape=self.input_shape,
            architecture_id=f"arch_{int(time.time())}_{random.randint(1000, 9999)}"
        )
        
        # Learning config: choose from one parent or combine
        if random.random() < 0.5:
            child.learning_config = self.learning_config
        else:
            child.learning_config = other.learning_config
            
        # Maybe mutate learning config
        if random.random() < DEFAULT_MUTATION_RATE:
            child.learning_config = child.learning_config.mutate()
        
        # Layers: crossover
        max_layers = min(len(self.layers), len(other.layers))
        min_layers = 2
        
        # Decide number of layers for child
        if random.random() < 0.5:
            # Number of layers between parents
            num_layers = random.randint(min_layers, max_layers)
        else:
            # More diverse option: potentially more layers than either parent
            max_possible = min(MAX_LAYERS, max(len(self.layers), len(other.layers)) + 1)
            num_layers = random.randint(min_layers, max_possible)
        
        # Create layer list
        child.layers = []
        
        # First layer: take from one parent or create new
        if random.random() < 0.8:  # 80% chance to inherit
            first_layer = random.choice([self.layers[0], other.layers[0]])
            # Maybe mutate
            if random.random() < DEFAULT_MUTATION_RATE:
                first_layer = first_layer.mutate()
            child.layers.append(first_layer)
        else:
            # New first layer
            if len(self.input_shape) > 1:
                first_layer_type = random.choice(["lstm", "gru"])
            else:
                first_layer_type = "dense"
            child.layers.append(LayerSpec.random(first_layer_type))
        
        # Middle layers: mix from both parents
        for i in range(1, num_layers - 1):
            if random.random() < 0.7:  # 70% chance to inherit
                # Pick a layer from either parent
                parent = random.choice([self, other])
                
                if i < len(parent.layers) - 1:
                    # Use a middle layer
                    layer = parent.layers[i]
                else:
                    # Use a random middle layer
                    idx = random.randint(1, len(parent.layers) - 2)
                    layer = parent.layers[idx]
                
                # Maybe mutate
                if random.random() < DEFAULT_MUTATION_RATE:
                    layer = layer.mutate()
                
                child.layers.append(layer)
            else:
                # New random middle layer
                layer_type = random.choice(["dense", "dense", "lstm", "gru"])
                child.layers.append(LayerSpec.random(layer_type))
        
        # Last layer: take from one parent or create new
        if random.random() < 0.8:  # 80% chance to inherit
            last_idx_self = len(self.layers) - 1
            last_idx_other = len(other.layers) - 1
            
            last_layer = random.choice([
                self.layers[last_idx_self], 
                other.layers[last_idx_other]
            ])
            
            # Maybe mutate
            if random.random() < DEFAULT_MUTATION_RATE:
                last_layer = last_layer.mutate()
                
            child.layers.append(last_layer)
        else:
            # New last layer (always dense)
            child.layers.append(LayerSpec.random("dense"))
        
        return child
    
    def to_keras_model(self, output_units: int = 1) -> Any:
        """
        Convert to a Keras model
        
        Args:
            output_units: Number of output units
            
        Returns:
            Keras Model instance
        """
        try:
            # Import TensorFlow locally to avoid dependencies
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                Dense, LSTM, GRU, Dropout, BatchNormalization, 
                LayerNormalization, Conv1D, Input
            )
            from tensorflow.keras.regularizers import l2
            from tensorflow.keras.optimizers import Adam, SGD, RMSprop
            
            # Create model
            model = Sequential()
            
            # Add input layer
            input_shape = self.input_shape
            first_layer = self.layers[0]
            
            # Add first layer with input shape
            if first_layer.layer_type == "dense":
                model.add(Dense(
                    units=first_layer.units,
                    activation=first_layer.activation,
                    kernel_regularizer=l2(first_layer.l2_regularization) if first_layer.l2_regularization > 0 else None,
                    input_shape=input_shape
                ))
            elif first_layer.layer_type == "lstm":
                model.add(LSTM(
                    units=first_layer.units,
                    activation=first_layer.activation,
                    kernel_regularizer=l2(first_layer.l2_regularization) if first_layer.l2_regularization > 0 else None,
                    return_sequences=len(self.layers) > 1 and self.layers[1].layer_type in ["lstm", "gru"],
                    input_shape=input_shape
                ))
            elif first_layer.layer_type == "gru":
                model.add(GRU(
                    units=first_layer.units,
                    activation=first_layer.activation,
                    kernel_regularizer=l2(first_layer.l2_regularization) if first_layer.l2_regularization > 0 else None,
                    return_sequences=len(self.layers) > 1 and self.layers[1].layer_type in ["lstm", "gru"],
                    input_shape=input_shape
                ))
            elif first_layer.layer_type == "conv1d":
                model.add(Conv1D(
                    filters=first_layer.units,
                    kernel_size=first_layer.kernel_size or 3,
                    activation=first_layer.activation,
                    kernel_regularizer=l2(first_layer.l2_regularization) if first_layer.l2_regularization > 0 else None,
                    input_shape=input_shape
                ))
            
            # Add normalization if requested
            if first_layer.batch_normalization:
                model.add(BatchNormalization())
            elif first_layer.layer_normalization:
                model.add(LayerNormalization())
            
            # Add dropout if requested
            if first_layer.dropout_rate > 0:
                model.add(Dropout(first_layer.dropout_rate))
            
            # Add remaining layers
            for i, layer_spec in enumerate(self.layers[1:], 1):
                is_last_recurrent = i == len(self.layers) - 1 and layer_spec.layer_type in ["lstm", "gru"]
                next_is_recurrent = i < len(self.layers) - 1 and self.layers[i+1].layer_type in ["lstm", "gru"]
                
                if layer_spec.layer_type == "dense":
                    model.add(Dense(
                        units=layer_spec.units,
                        activation=layer_spec.activation,
                        kernel_regularizer=l2(layer_spec.l2_regularization) if layer_spec.l2_regularization > 0 else None
                    ))
                elif layer_spec.layer_type == "lstm":
                    model.add(LSTM(
                        units=layer_spec.units,
                        activation=layer_spec.activation,
                        kernel_regularizer=l2(layer_spec.l2_regularization) if layer_spec.l2_regularization > 0 else None,
                        return_sequences=next_is_recurrent
                    ))
                elif layer_spec.layer_type == "gru":
                    model.add(GRU(
                        units=layer_spec.units,
                        activation=layer_spec.activation,
                        kernel_regularizer=l2(layer_spec.l2_regularization) if layer_spec.l2_regularization > 0 else None,
                        return_sequences=next_is_recurrent
                    ))
                elif layer_spec.layer_type == "conv1d":
                    model.add(Conv1D(
                        filters=layer_spec.units,
                        kernel_size=layer_spec.kernel_size or 3,
                        activation=layer_spec.activation,
                        kernel_regularizer=l2(layer_spec.l2_regularization) if layer_spec.l2_regularization > 0 else None
                    ))
                
                # Add normalization if requested
                if layer_spec.batch_normalization:
                    model.add(BatchNormalization())
                elif layer_spec.layer_normalization:
                    model.add(LayerNormalization())
                
                # Add dropout if requested
                if layer_spec.dropout_rate > 0:
                    model.add(Dropout(layer_spec.dropout_rate))
            
            # Add output layer if needed
            last_layer = self.layers[-1]
            if last_layer.layer_type != "dense" or last_layer.units != output_units:
                model.add(Dense(output_units))
            
            # Configure optimizer
            if self.learning_config.optimizer == "adam":
                optimizer = Adam(learning_rate=self.learning_config.learning_rate)
            elif self.learning_config.optimizer == "sgd":
                optimizer = SGD(learning_rate=self.learning_config.learning_rate)
            elif self.learning_config.optimizer == "rmsprop":
                optimizer = RMSprop(learning_rate=self.learning_config.learning_rate)
            else:
                # Default to Adam
                optimizer = Adam(learning_rate=self.learning_config.learning_rate)
            
            # Configure loss
            if self.learning_config.loss_function == "mse":
                loss = "mean_squared_error"
            elif self.learning_config.loss_function == "mae":
                loss = "mean_absolute_error"
            elif self.learning_config.loss_function == "huber":
                loss = tf.keras.losses.Huber()
            else:
                # Default to MSE
                loss = "mean_squared_error"
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=["mae", "mse"]
            )
            
            return model
            
        except ImportError:
            logger.error("TensorFlow not installed, cannot create Keras model")
            return None
    
    def estimate_complexity(self) -> int:
        """
        Estimate model complexity as total number of parameters
        
        Returns:
            Estimated parameter count
        """
        total_params = 0
        
        # Simple estimation for dense layers
        prev_units = np.prod(self.input_shape)
        
        for layer in self.layers:
            if layer.layer_type == "dense":
                # Dense layer: weights + biases
                params = (prev_units * layer.units) + layer.units
                total_params += params
                prev_units = layer.units
            elif layer.layer_type in ["lstm", "gru"]:
                # Recurrent layers have more parameters
                if layer.layer_type == "lstm":
                    # LSTM has 4 gates with weights and biases
                    params = 4 * ((prev_units * layer.units) + (layer.units * layer.units) + layer.units)
                else:  # GRU
                    # GRU has 3 gates with weights and biases
                    params = 3 * ((prev_units * layer.units) + (layer.units * layer.units) + layer.units)
                
                total_params += params
                prev_units = layer.units
            elif layer.layer_type == "conv1d":
                # Convolutional layer
                kernel_size = layer.kernel_size or 3
                params = (kernel_size * prev_units * layer.units) + layer.units
                total_params += params
                prev_units = layer.units
            
            # Add parameters for batch/layer normalization
            if layer.batch_normalization or layer.layer_normalization:
                # Each unit needs 2 parameters (scale and shift)
                total_params += 2 * layer.units
        
        return total_params
    
    def calculate_fitness(self) -> float:
        """
        Calculate fitness score (0-1) based on performance and complexity
        
        Returns:
            Fitness score
        """
        # Get performance metrics
        val_loss = self.performance.get("val_loss", float('inf'))
        val_mae = self.performance.get("val_mae", float('inf'))
        
        if val_loss == float('inf') or val_mae == float('inf'):
            return 0.0  # No performance data
        
        # Normalize performance metrics
        # Lower is better, so convert to 0-1 range where higher is better
        norm_loss = max(0, 1.0 - (val_loss / 10.0))  # Assuming loss under 10 is good
        norm_mae = max(0, 1.0 - (val_mae / 5.0))    # Assuming MAE under 5 is good
        
        # Calculate complexity penalty
        complexity = self.estimate_complexity()
        # Penalize very complex models (over ~1M parameters)
        complexity_penalty = max(0, min(0.5, (complexity - 1_000_000) / 10_000_000))
        
        # Combine metrics with weights
        fitness = (
            norm_loss * 0.5 +
            norm_mae * 0.5 -
            complexity_penalty
        )
        
        return max(0, min(1, fitness))
    
    def get_summary(self) -> str:
        """
        Get a textual summary of the architecture
        
        Returns:
            Architecture summary string
        """
        lines = [
            f"Architecture ID: {self.architecture_id}",
            f"Created: {self.created_at.isoformat()}",
            f"Input shape: {self.input_shape}",
            f"Layers: {len(self.layers)}",
            "---"
        ]
        
        for i, layer in enumerate(self.layers):
            lines.append(f"Layer {i+1}: {layer.layer_type} - {layer.units} units - {layer.activation}")
            
            if layer.dropout_rate > 0:
                lines.append(f"  Dropout: {layer.dropout_rate}")
                
            if layer.batch_normalization:
                lines.append("  BatchNormalization: Yes")
            elif layer.layer_normalization:
                lines.append("  LayerNormalization: Yes")
                
            if layer.l2_regularization > 0:
                lines.append(f"  L2 Regularization: {layer.l2_regularization}")
        
        lines.append("---")
        lines.append(f"Learning rate: {self.learning_config.learning_rate}")
        lines.append(f"Batch size: {self.learning_config.batch_size}")
        lines.append(f"Optimizer: {self.learning_config.optimizer}")
        lines.append(f"Loss function: {self.learning_config.loss_function}")
        
        if self.performance:
            lines.append("---")
            lines.append("Performance:")
            for metric, value in self.performance.items():
                lines.append(f"  {metric}: {value}")
        
        lines.append(f"Estimated parameters: {self.estimate_complexity():,}")
        lines.append(f"Fitness score: {self.calculate_fitness():.4f}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"NeuralArchitecture(id={self.architecture_id}, layers={len(self.layers)}, fitness={self.calculate_fitness():.2f})"

class ArchitectureSearch:
    """
    Neural Architecture Search for finding optimal architectures
    """
    
    def __init__(self, input_shape: Tuple[int, ...], output_shape: int = 1):
        """
        Initialize the architecture search
        
        Args:
            input_shape: Input shape for the neural network
            output_shape: Output shape (number of units in final layer)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.population = []
        self.history = []
        self.generation = 0
        self.best_architecture = None
        self.best_fitness = 0.0
    
    def initialize_population(self, size: int = 10) -> None:
        """
        Initialize the population with random architectures
        
        Args:
            size: Population size
        """
        self.population = []
        
        for _ in range(size):
            arch = NeuralArchitecture.random(self.input_shape)
            self.population.append(arch)
            
        logger.info(f"Initialized population with {size} architectures")
    
    def evaluate_population(self, 
                          train_data: Tuple[np.ndarray, np.ndarray],
                          val_data: Tuple[np.ndarray, np.ndarray],
                          epochs: int = None) -> None:
        """
        Evaluate all architectures in the population
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            epochs: Number of epochs to train (overrides architecture config)
        """
        try:
            # Import TensorFlow locally
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            logger.error("TensorFlow not installed, cannot evaluate population")
            return
            
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        for i, arch in enumerate(self.population):
            logger.info(f"Evaluating architecture {i+1}/{len(self.population)}: {arch.architecture_id}")
            
            try:
                # Create model
                model = arch.to_keras_model(output_units=self.output_shape)
                
                if model is None:
                    logger.error(f"Failed to create model for architecture {arch.architecture_id}")
                    continue
                    
                # Configure callbacks
                callbacks = []
                
                if arch.learning_config.early_stopping:
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=arch.learning_config.patience,
                        restore_best_weights=True
                    )
                    callbacks.append(early_stopping)
                
                # Train the model
                actual_epochs = epochs or arch.learning_config.epochs
                
                history = model.fit(
                    X_train, y_train,
                    batch_size=arch.learning_config.batch_size,
                    epochs=actual_epochs,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Get final performance
                val_performance = model.evaluate(X_val, y_val, verbose=0)
                
                # Store performance metrics
                metrics = model.metrics_names
                arch.performance = {metrics[i]: float(val_performance[i]) for i in range(len(metrics))}
                
                # Add training history
                for key, values in history.history.items():
                    if key.startswith('val_'):
                        arch.performance[f"{key}_history"] = [float(v) for v in values]
                
                # Calculate fitness
                fitness = arch.calculate_fitness()
                
                # Update best architecture if needed
                if fitness > self.best_fitness:
                    self.best_architecture = arch
                    self.best_fitness = fitness
                    logger.info(f"New best architecture: {arch.architecture_id} with fitness {fitness:.4f}")
                
                logger.info(f"Evaluated {arch.architecture_id} - Loss: {arch.performance.get('loss', 'N/A')}, Fitness: {fitness:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating architecture {arch.architecture_id}: {e}")
                # Give it a very poor performance score
                arch.performance = {
                    "error": str(e),
                    "val_loss": float('inf'),
                    "val_mae": float('inf')
                }
    
    def evolve_population(self) -> None:
        """Evolve the population using genetic algorithm"""
        # Increment generation
        self.generation += 1
        
        # Calculate fitness for all architectures
        fitness_scores = [arch.calculate_fitness() for arch in self.population]
        
        # Add current best to history
        if self.best_architecture is not None:
            self.history.append(self.best_architecture)
            if len(self.history) > MAX_ARCHITECTURE_HISTORY:
                self.history = self.history[-MAX_ARCHITECTURE_HISTORY:]
        
        # Sort population by fitness
        sorted_pop = [a for _, a in sorted(
            zip(fitness_scores, self.population),
            key=lambda pair: pair[0],
            reverse=True
        )]
        
        # Keep track of best architectures
        best_arch = sorted_pop[0] if sorted_pop else None
        if best_arch and best_arch.calculate_fitness() > self.best_fitness:
            self.best_architecture = best_arch
            self.best_fitness = best_arch.calculate_fitness()
        
        # Elite selection - keep top performers
        elite_count = max(1, int(len(self.population) * 0.2))
        elite = sorted_pop[:elite_count]
        
        # Create new population
        new_population = list(elite)  # Start with elite
        
        # Fill rest of population with:
        # - Crossovers between good performers
        # - Mutations of good performers
        # - Some random architectures for diversity
        
        # Number of individuals to generate
        crossover_count = int(len(self.population) * 0.5)
        mutation_count = int(len(self.population) * 0.2)
        random_count = len(self.population) - elite_count - crossover_count - mutation_count
        
        # Create crossovers
        for _ in range(crossover_count):
            # Select parents using tournament selection
            parent1 = self._tournament_selection(sorted_pop, tournament_size=3)
            parent2 = self._tournament_selection(sorted_pop, tournament_size=3)
            
            # Avoid same parent
            attempts = 0
            while parent2 == parent1 and attempts < 5:
                parent2 = self._tournament_selection(sorted_pop, tournament_size=3)
                attempts += 1
                
            # Create child
            child = parent1.cross(parent2)
            new_population.append(child)
        
        # Create mutations
        for _ in range(mutation_count):
            # Select architecture to mutate
            parent = self._tournament_selection(sorted_pop, tournament_size=3)
            
            # Create mutated child
            child = parent.mutate()
            new_population.append(child)
        
        # Create random architectures
        for _ in range(random_count):
            arch = NeuralArchitecture.random(self.input_shape)
            new_population.append(arch)
        
        # Update population
        self.population = new_population
        
        logger.info(f"Evolved population to generation {self.generation} with {len(self.population)} architectures")
        if self.best_architecture:
            logger.info(f"Best architecture: {self.best_architecture.architecture_id} with fitness {self.best_fitness:.4f}")
    
    def _tournament_selection(self, population: List[NeuralArchitecture], tournament_size: int = 3) -> NeuralArchitecture:
        """
        Select an architecture using tournament selection
        
        Args:
            population: List of architectures
            tournament_size: Tournament size
            
        Returns:
            Selected architecture
        """
        # Randomly select tournament_size architectures
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Return the one with highest fitness
        return max(tournament, key=lambda arch: arch.calculate_fitness())
    
    def run_search(self, 
                  train_data: Tuple[np.ndarray, np.ndarray],
                  val_data: Tuple[np.ndarray, np.ndarray],
                  generations: int = 5,
                  population_size: int = 10,
                  epochs: int = None) -> NeuralArchitecture:
        """
        Run the architecture search
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            generations: Number of generations to evolve
            population_size: Population size
            epochs: Number of epochs to train each architecture
            
        Returns:
            Best architecture found
        """
        # Initialize population
        self.initialize_population(population_size)
        
        for gen in range(generations):
            logger.info(f"Starting generation {gen+1}/{generations}")
            
            # Evaluate population
            self.evaluate_population(train_data, val_data, epochs)
            
            # Evolve population
            self.evolve_population()
            
            # Log progress
            logger.info(f"Generation {gen+1} complete. Best fitness: {self.best_fitness:.4f}")
        
        logger.info(f"Architecture search complete. Best fitness: {self.best_fitness:.4f}")
        
        if self.best_architecture:
            logger.info(f"Best architecture summary:\n{self.best_architecture.get_summary()}")
            
        return self.best_architecture
    
    def save_state(self, filepath: str) -> None:
        """
        Save the search state to a file
        
        Args:
            filepath: Path to save to
        """
        state = {
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "population": [arch.to_dict() for arch in self.population],
            "history": [arch.to_dict() for arch in self.history],
            "generation": self.generation,
            "best_architecture": self.best_architecture.to_dict() if self.best_architecture else None,
            "best_fitness": self.best_fitness
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Saved architecture search state to {filepath}")
        except Exception as e:
            logger.error(f"Error saving search state: {e}")
    
    @classmethod
    def load_state(cls, filepath: str) -> 'ArchitectureSearch':
        """
        Load search state from a file
        
        Args:
            filepath: Path to load from
            
        Returns:
            ArchitectureSearch instance
        """
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            # Create search instance
            search = cls(state["input_shape"], state["output_shape"])
            
            # Load population
            search.population = [NeuralArchitecture.from_dict(arch_dict) for arch_dict in state["population"]]
            
            # Load history
            search.history = [NeuralArchitecture.from_dict(arch_dict) for arch_dict in state["history"]]
            
            # Load generation
            search.generation = state["generation"]
            
            # Load best architecture
            if state["best_architecture"]:
                search.best_architecture = NeuralArchitecture.from_dict(state["best_architecture"])
                
            # Load best fitness
            search.best_fitness = state["best_fitness"]
            
            logger.info(f"Loaded architecture search state from {filepath}")
            logger.info(f"Generation: {search.generation}, Population size: {len(search.population)}")
            if search.best_architecture:
                logger.info(f"Best fitness: {search.best_fitness:.4f}")
            
            return search
            
        except Exception as e:
            logger.error(f"Error loading search state: {e}")
            raise e
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get a summary of search progress
        
        Returns:
            Dictionary with progress summary
        """
        # Calculate stats about the population
        fitnesses = [arch.calculate_fitness() for arch in self.population]
        
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
        complexity_stats = {
            "min": min([arch.estimate_complexity() for arch in self.population], default=0),
            "max": max([arch.estimate_complexity() for arch in self.population], default=0),
            "avg": sum([arch.estimate_complexity() for arch in self.population]) / len(self.population) if self.population else 0
        }
        
        layer_counts = [len(arch.layers) for arch in self.population]
        avg_layers = sum(layer_counts) / len(layer_counts) if layer_counts else 0
        
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_fitness,
            "avg_fitness": avg_fitness,
            "best_architecture_id": self.best_architecture.architecture_id if self.best_architecture else None,
            "complexity_stats": complexity_stats,
            "avg_layers": avg_layers,
            "history_size": len(self.history)
        }

class TransferLearningArchitectureSearch(ArchitectureSearch):
    """
    Neural Architecture Search with transfer learning from previous searches
    """
    
    def __init__(self, input_shape: Tuple[int, ...], output_shape: int = 1, 
                source_architectures: List[NeuralArchitecture] = None):
        """
        Initialize the transfer learning architecture search
        
        Args:
            input_shape: Input shape for the neural network
            output_shape: Output shape (number of units in final layer)
            source_architectures: Previous good architectures to learn from
        """
        super().__init__(input_shape, output_shape)
        self.source_architectures = source_architectures or []
    
    def initialize_population(self, size: int = 10) -> None:
        """
        Initialize the population with a mix of random and transferred architectures
        
        Args:
            size: Population size
        """
        self.population = []
        
        # Determine how many architectures to transfer
        transfer_count = min(len(self.source_architectures), int(size * 0.3))
        random_count = size - transfer_count
        
        # Transfer source architectures
        if transfer_count > 0:
            # Sort source architectures by fitness
            sorted_sources = sorted(
                self.source_architectures,
                key=lambda arch: arch.calculate_fitness(),
                reverse=True
            )
            
            # Take top architectures and adapt them to new input shape if needed
            for i in range(transfer_count):
                source_arch = sorted_sources[i % len(sorted_sources)]
                
                # Create a copy with the right input shape
                arch = NeuralArchitecture(
                    input_shape=self.input_shape,
                    layers=source_arch.layers,
                    learning_config=source_arch.learning_config,
                    architecture_id=f"transfer_{int(time.time())}_{random.randint(1000, 9999)}"
                )
                
                # Add to population
                self.population.append(arch)
                
            logger.info(f"Transferred {transfer_count} architectures from source")
        
        # Add random architectures
        for _ in range(random_count):
            arch = NeuralArchitecture.random(self.input_shape)
            self.population.append(arch)
            
        logger.info(f"Initialized population with {size} architectures ({transfer_count} transferred, {random_count} random)")

class ProgressiveGrowthArchitectureSearch(ArchitectureSearch):
    """
    Neural Architecture Search with progressive growth of architectures
    """
    
    def evolve_population(self) -> None:
        """Evolve the population with progressive growth"""
        # Standard evolution
        super().evolve_population()
        
        # Progressive growth: periodically try growing the best architectures
        if self.generation % 2 == 0 and self.best_architecture is not None:
            # Take top architectures
            top_archs = sorted(
                self.population,
                key=lambda arch: arch.calculate_fitness(),
                reverse=True
            )[:3]
            
            # Create grown versions
            for parent in top_archs:
                # Create grown architecture
                grown = self._grow_architecture(parent)
                
                # Replace a random low-performing architecture
                if len(self.population) > 5:
                    # Sort by fitness (ascending)
                    sorted_pop = sorted(
                        self.population,
                        key=lambda arch: arch.calculate_fitness()
                    )
                    
                    # Replace one of the bottom 3
                    replace_idx = self.population.index(sorted_pop[random.randint(0, 2)])
                    self.population[replace_idx] = grown
    
    def _grow_architecture(self, arch: NeuralArchitecture) -> NeuralArchitecture:
        """
        Create a grown version of an architecture
        
        Args:
            arch: Source architecture
            
        Returns:
            Grown architecture
        """
        # Create a new instance
        grown = NeuralArchitecture(
            input_shape=arch.input_shape,
            architecture_id=f"grown_{int(time.time())}_{random.randint(1000, 9999)}"
        )
        
        # Copy learning config
        grown.learning_config = arch.learning_config
        
        # Copy existing layers
        grown.layers = arch.layers.copy()
        
        # Add a new layer at a random position (not first or last)
        if len(grown.layers) >= 3:
            insert_pos = random.randint(1, len(grown.layers) - 1)
            
            # Create a layer similar to either the layer before or after
            reference_layer = random.choice([grown.layers[insert_pos-1], grown.layers[insert_pos]])
            
            # Create new layer with same type but potentially different parameters
            new_layer = LayerSpec(
                layer_type=reference_layer.layer_type,
                units=reference_layer.units,
                activation=reference_layer.activation,
                dropout_rate=reference_layer.dropout_rate
            )
            
            # Mutate some parameters
            new_layer = new_layer.mutate(mutation_rate=0.5)
            
            # Insert the new layer
            grown.layers.insert(insert_pos, new_layer)
        
        return grown

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Example data for architecture search
    # In a real scenario, this would be your training data
    X_train = np.random.normal(0, 1, (1000, 10))
    y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.normal(0, 1, 1000)
    
    X_val = np.random.normal(0, 1, (200, 10))
    y_val = np.sin(X_val[:, 0]) + 0.1 * np.random.normal(0, 1, 200)
    
    # Initialize architecture search
    search = ArchitectureSearch(input_shape=(10,), output_shape=1)
    
    # Try to create and evaluate a sample architecture
    try:
        # Import TensorFlow
        import tensorflow as tf
        print("TensorFlow available, running neural architecture search...")
        
        # Run search for a small number of generations as a demo
        best_arch = search.run_search(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            generations=2,
            population_size=5,
            epochs=10
        )
        
        if best_arch:
            print(f"Best architecture fitness: {best_arch.calculate_fitness():.4f}")
            print(f"Estimated parameters: {best_arch.estimate_complexity():,}")
            
    except ImportError:
        print("TensorFlow not available, showing architecture search simulation...")
        
        # Just create and show architecture examples
        search.initialize_population(5)
        
        print("Sample Neural Architectures:")
        for i, arch in enumerate(search.population):
            print(f"\nArchitecture {i+1}:")
            print(f"Layers: {len(arch.layers)}")
            print(f"Learning rate: {arch.learning_config.learning_rate}")
            print(f"Batch size: {arch.learning_config.batch_size}")
            print(f"Estimated complexity: {arch.estimate_complexity():,} parameters")
            
        # Simulate evolution
        print("\nSimulating evolution...")
        
        # Assign random performance metrics
        for arch in search.population:
            arch.performance = {
                "val_loss": random.uniform(0.1, 1.0),
                "val_mae": random.uniform(0.1, 0.5)
            }
            
        # Evolve population
        search.evolve_population()
        
        print("\nAfter evolution, new generation:")
        for i, arch in enumerate(search.population[:3]):
            print(f"\nArchitecture {i+1}:")
            print(f"Layers: {len(arch.layers)}")
            print(f"Learning rate: {arch.learning_config.learning_rate}")
            print(f"Fitness: {arch.calculate_fitness():.4f}")
    
    print("\nNeural Architecture Search demonstrated successfully!")