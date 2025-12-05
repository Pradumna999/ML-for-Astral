"""
Neural network architecture for collision prediction.
Implements MLP with dropout and proper initialization.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollisionPredictionModel:
    """Multilayer Perceptron for satellite collision prediction."""
    
    def __init__(self, input_dim: int = 6):
        self.input_dim = input_dim
        self.model = None
        self.history = None
    
    def build_model(self, 
                   hidden_layers: list = [64, 32, 16],
                   dropout_rate: float = 0.2,
                   learning_rate: float = 0.001) -> Model:
        """
        Build the neural network architecture.
        
        Args:
            hidden_layers: List of neurons in each hidden layer
            dropout_rate: Dropout probability
            learning_rate: Adam optimizer learning rate
            
        Returns:
            Compiled Keras model
        """
        logger.info("Building model architecture...")
        
        # Input layer
        inputs = keras.Input(shape=(self.input_dim,), name='orbital_deltas')
        
        # First hidden layer with dropout
        x = layers.Dense(
            hidden_layers[0], 
            activation='relu',
            kernel_initializer='he_normal',
            name='hidden_1'
        )(inputs)
        x = layers.Dropout(dropout_rate, name='dropout_1')(x)
        
        # Additional hidden layers
        for i, units in enumerate(hidden_layers[1:], start=2):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_initializer='he_normal',
                name=f'hidden_{i}'
            )(x)
        
        # Output layer (sigmoid for binary classification)
        outputs = layers.Dense(
            1, 
            activation='sigmoid',
            name='collision_probability'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='collision_predictor')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='pr_auc', curve='PR')
            ]
        )
        
        self.model = model
        
        logger.info("Model built successfully")
        logger.info(f"Architecture: {' -> '.join([str(self.input_dim)] + [str(h) for h in hidden_layers] + ['1'])}")
        
        return model
    
    def get_callbacks(self, 
                     checkpoint_dir: str = 'models/checkpoints',
                     patience: int = 10) -> list:
        """
        Get training callbacks for monitoring and optimization.
        
        Args:
            checkpoint_dir: Directory to save model checkpoints
            patience: Early stopping patience
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            # Early stopping based on validation loss
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            
            # Save best model
            ModelCheckpoint(
                filepath='models/saved_models/collision_model_best.keras',
                monitor='val_recall',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        return callbacks
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            logger.error("Model not built yet. Call build_model() first.")
            return
        self.model.summary()
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            logger.error("No model to save.")
            return
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model
    
    def predict(self, X, threshold: float = 0.5):
        """
        Make predictions on input data.
        
        Args:
            X: Input features (N, 6)
            threshold: Classification threshold
            
        Returns:
            probabilities: Raw probabilities (N,)
            predictions: Binary predictions (N,)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        probabilities = self.model.predict(X, verbose=0).flatten()
        predictions = (probabilities >= threshold).astype(int)
        
        return probabilities, predictions


# Example usage
if __name__ == "__main__":
    # Create and build model
    collision_model = CollisionPredictionModel(input_dim=6)
    model = collision_model.build_model(
        hidden_layers=[64, 32, 16],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    # Display architecture
    collision_model.summary()
    
    # Test with random data
    import numpy as np
    X_test = np.random.randn(10, 6)
    probs, preds = collision_model.predict(X_test, threshold=0.5)
    
    print(f"\nTest predictions:")
    print(f"Probabilities: {probs}")
    print(f"Binary predictions: {preds}")