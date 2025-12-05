"""
Training pipeline for collision prediction model.
Handles data loading, splitting, training, and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import logging
from pathlib import Path
import json
from datetime import datetime

from src.data_preprocessing import SatelliteDataPreprocessor
from src.model_architecture import CollisionPredictionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline for collision prediction."""
    
    def __init__(self, config: dict):
        self.config = config
        self.preprocessor = SatelliteDataPreprocessor()
        self.model_wrapper = CollisionPredictionModel()
        self.history = None
        
        # Create directories
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        Path('models/saved_models').mkdir(parents=True, exist_ok=True)
        Path('outputs/visualizations').mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(parents=True, exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load data and prepare features."""
        logger.info("=" * 60)
        logger.info("STEP 1: Loading and preparing data")
        logger.info("=" * 60)
        
        # Load and extract features
        X, y = self.preprocessor.prepare_training_data(
            collision_file=self.config['collision_data_path'],
            safe_file=self.config['safe_data_path']
        )
        
        # Shuffle data
        X, y = shuffle(X, y, random_state=self.config['random_seed'])
        
        # Split into train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config['test_split'],
            random_state=self.config['random_seed'],
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['val_split'] / (1 - self.config['test_split']),
            random_state=self.config['random_seed'],
            stratify=y_temp
        )
        
        logger.info(f"Data splits:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Val: {len(X_val)} samples")
        logger.info(f"  Test: {len(X_test)} samples")
        
        # Fit scaler on training data only
        self.preprocessor.fit_scaler(X_train)
        
        # Transform all splits
        X_train_scaled = self.preprocessor.transform(X_train)
        X_val_scaled = self.preprocessor.transform(X_val)
        X_test_scaled = self.preprocessor.transform(X_test)
        
        # Save scaler
        self.preprocessor.save_scaler(self.config['scaler_path'])
        
        # Save processed data
        np.save('data/processed/X_train.npy', X_train_scaled)
        np.save('data/processed/X_val.npy', X_val_scaled)
        np.save('data/processed/X_test.npy', X_test_scaled)
        np.save('data/processed/y_train.npy', y_train)
        np.save('data/processed/y_val.npy', y_val)
        np.save('data/processed/y_test.npy', y_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """Train the neural network."""
        logger.info("=" * 60)
        logger.info("STEP 2: Training model")
        logger.info("=" * 60)
        
        # Build model
        self.model_wrapper.build_model(
            hidden_layers=self.config['hidden_layers'],
            dropout_rate=self.config['dropout_rate'],
            learning_rate=self.config['learning_rate']
        )
        
        # Display architecture
        self.model_wrapper.summary()
        
        # Get callbacks
        callbacks = self.model_wrapper.get_callbacks(
            patience=self.config['early_stopping_patience']
        )
        
        # Train model
        logger.info(f"Starting training for {self.config['epochs']} epochs...")
        self.history = self.model_wrapper.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model_wrapper.save_model(self.config['model_path'])
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            logger.warning("No training history to plot")
            return
        
        logger.info("Plotting training history...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training History', fontsize=16)
        
        metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'pr_auc']
        
        for idx, metric in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            if metric in self.history.history:
                ax.plot(self.history.history[metric], label=f'Train {metric}')
                ax.plot(self.history.history[f'val_{metric}'], label=f'Val {metric}')
                ax.set_title(metric.upper())
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/training_history.png', dpi=300)
        logger.info("Training history plot saved")
        plt.close()
    
    def save_training_log(self):
        """Save training configuration and results."""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'final_metrics': {
                metric: float(self.history.history[metric][-1])
                for metric in self.history.history.keys()
            },
            'best_epoch': int(np.argmin(self.history.history['val_loss'])) + 1,
            'total_epochs': len(self.history.history['loss'])
        }
        
        with open('logs/training_log.json', 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info("Training log saved")
    
    def run(self):
        """Execute complete training pipeline."""
        logger.info("\n" + "=" * 60)
        logger.info("SATELLITE COLLISION PREDICTION - TRAINING PIPELINE")
        logger.info("=" * 60 + "\n")
        
        # Step 1: Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data()
        
        # Step 2: Train model
        self.train_model(X_train, X_val, y_train, y_val)
        
        # Step 3: Visualize results
        self.plot_training_history()
        
        # Step 4: Save logs
        self.save_training_log()
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {self.config['model_path']}")
        logger.info(f"Scaler saved to: {self.config['scaler_path']}")
        logger.info(f"Visualizations saved to: outputs/visualizations/")
        
        return self.model_wrapper, self.preprocessor


# Main execution
if __name__ == "__main__":
    # Configuration
    config = {
        # Data paths
        'collision_data_path': 'data/raw/collision_data.json',
        'safe_data_path': 'data/raw/safe_data.json',
        'scaler_path': 'data/scalers/standard_scaler.pkl',
        'model_path': 'models/saved_models/collision_model.keras',
        
        # Model architecture
        'hidden_layers': [64, 32, 16],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        
        # Training parameters
        'epochs': 100,
        'batch_size': 64,
        'early_stopping_patience': 10,
        
        # Data splitting
        'test_split': 0.15,
        'val_split': 0.15,
        'random_seed': 42
    }
    
    # Run training pipeline
    pipeline = TrainingPipeline(config)
    model, preprocessor = pipeline.run()