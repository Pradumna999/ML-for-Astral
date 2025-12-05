"""
Main workflow orchestrator for satellite collision prediction system.
Provides unified interface for training, prediction, and countermeasures.
"""

import argparse
import sys
import logging
from pathlib import Path
import json
from typing import Dict

from src.train import TrainingPipeline
from src.predict import CollisionPredictor
from src.countermeasures import CollisionAvoidance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SatelliteCollisionSystem:
    """
    Complete satellite collision prediction and avoidance system.
    """
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file or use defaults."""
        config_path = 'config/config.yaml'
        
        # Default configuration
        default_config = {
            # Paths
            'collision_data_path': 'data/raw/collision_data.json',
            'safe_data_path': 'data/raw/safe_data.json',
            'model_path': 'models/saved_models/collision_model.keras',
            'best_model_path': 'models/saved_models/collision_model_best.keras',
            'scaler_path': 'data/scalers/standard_scaler.pkl',
            
            # Model architecture
            'hidden_layers': [64, 32, 16],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            
            # Training parameters
            'epochs': 100,
            'batch_size': 64,
            'early_stopping_patience': 10,
            
            # Data splits
            'test_split': 0.15,
            'val_split': 0.15,
            'random_seed': 42,
            
            # Prediction thresholds
            'collision_threshold': 0.1,
            'target_probability': 0.01
        }
        
        # Try to load from file if exists
        if Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config: {e}. Using defaults.")
        
        return default_config
    
    def train(self):
        """Train the collision prediction model."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING WORKFLOW")
        logger.info("=" * 80 + "\n")
        
        pipeline = TrainingPipeline(self.config)
        model, preprocessor = pipeline.run()
        
        logger.info("\nTraining completed successfully!")
        logger.info(f"Model saved to: {self.config['model_path']}")
        logger.info(f"Best model saved to: {self.config['best_model_path']}")
        
        return model, preprocessor
    
    def predict(self, input_file: str, threshold: float = None):
        """
        Run collision prediction on input satellite data.
        
        Args:
            input_file: Path to input JSON file
            threshold: Collision probability threshold
        """
        if threshold is None:
            threshold = self.config['collision_threshold']
        
        logger.info("\n" + "=" * 80)
        logger.info("STARTING PREDICTION WORKFLOW")
        logger.info("=" * 80 + "\n")
        
        # Check if model exists
        if not Path(self.config['best_model_path']).exists():
            logger.error("Trained model not found! Please run training first.")
            return None
        
        # Initialize predictor
        predictor = CollisionPredictor(
            model_path=self.config['best_model_path'],
            scaler_path=self.config['scaler_path'],
            threshold=threshold
        )
        
        # Run prediction
        results = predictor.predict_collisions(input_file)
        
        # Save results
        predictor.save_results(results)
        predictor.generate_text_report(results)
        
        logger.info("\nPrediction completed successfully!")
        return results
    
    def generate_countermeasures(self, 
                                collision_report_path: str = 'outputs/collision_report.json',
                                input_data_path: str = None):
        """
        Generate countermeasure suggestions for detected collisions.
        
        Args:
            collision_report_path: Path to collision report JSON
            input_data_path: Path to original satellite data
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COUNTERMEASURE GENERATION")
        logger.info("=" * 80 + "\n")
        
        # Load collision report
        if not Path(collision_report_path).exists():
            logger.error(f"Collision report not found: {collision_report_path}")
            logger.error("Please run prediction first!")
            return None
        
        with open(collision_report_path, 'r') as f:
            report = json.load(f)
        
        # Check if there are warnings
        if not report['collision_warnings']:
            logger.info("No collision warnings found. No countermeasures needed.")
            return None
        
        # Load satellite data
        if input_data_path is None:
            input_data_path = report['metadata']['input_file']
        
        with open(input_data_path, 'r') as f:
            sat_data = json.load(f)
        
        # Initialize avoidance system
        avoidance = CollisionAvoidance(
            model_path=self.config['best_model_path'],
            scaler_path=self.config['scaler_path'],
            target_probability=self.config['target_probability']
        )
        
        # Generate maneuvers
        maneuvers = avoidance.suggest_maneuvers(
            report['collision_warnings'],
            sat_data['dataset']
        )
        
        # Save results
        avoidance.save_maneuvers(maneuvers)
        
        logger.info("\nCountermeasure generation completed successfully!")
        return maneuvers
    
    def full_pipeline(self, input_file: str):
        """
        Run complete pipeline: prediction + countermeasures.
        
        Args:
            input_file: Path to input satellite data JSON
        """
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING COMPLETE COLLISION ANALYSIS PIPELINE")
        logger.info("=" * 80 + "\n")
        
        # Step 1: Prediction
        results = self.predict(input_file)
        
        if results is None:
            return
        
        # Step 2: Countermeasures (if needed)
        if results['summary']['total_warnings'] > 0:
            logger.info(f"\n{results['summary']['total_warnings']} collision warnings detected.")
            logger.info("Generating countermeasures...\n")
            
            self.generate_countermeasures(
                input_data_path=input_file
            )
        else:
            logger.info("\nNo collision warnings detected. System is safe!")
        
        # Print final summary
        self._print_final_summary(results)
    
    def _print_final_summary(self, results: Dict):
        """Print final summary of analysis."""
        print("\n" + "=" * 80)
        print("FINAL ANALYSIS SUMMARY")
        print("=" * 80)
        
        summary = results['summary']
        print(f"\nTotal pairs analyzed: {results['metadata']['total_pairs_analyzed']}")
        print(f"Collision warnings: {summary['total_warnings']}")
        
        if summary['total_warnings'] > 0:
            print(f"  - High risk: {summary['high_risk']}")
            print(f"  - Medium risk: {summary['medium_risk']}")
            print(f"  - Low risk: {summary['low_risk']}")
            print(f"\nMaximum collision probability: {summary['max_probability']:.6f}")
            
            print("\nGenerated outputs:")
            print("  1. outputs/collision_report.json (detailed results)")
            print("  2. outputs/collision_report.txt (human-readable)")
            print("  3. outputs/countermeasures.json (avoidance maneuvers)")
        else:
            print("\n✓ All satellite pairs are safe!")
        
        print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Satellite Collision Prediction and Avoidance System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py train
  
  # Run prediction on new data
  python main.py predict data/raw/test_input.json
  
  # Run full pipeline (prediction + countermeasures)
  python main.py full data/raw/test_input.json
  
  # Generate countermeasures from existing report
  python main.py countermeasures
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'predict', 'countermeasures', 'full'],
        help='Operation mode'
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input JSON file (required for predict/full modes)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Collision probability threshold (default: 0.1)'
    )
    
    parser.add_argument(
        '--target-prob',
        type=float,
        default=None,
        help='Target probability for countermeasures (default: 0.01)'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = SatelliteCollisionSystem()
    
    # Update config if args provided
    if args.threshold is not None:
        system.config['collision_threshold'] = args.threshold
    if args.target_prob is not None:
        system.config['target_probability'] = args.target_prob
    
    # Execute requested mode
    try:
        if args.mode == 'train':
            system.train()
        
        elif args.mode == 'predict':
            if not args.input_file:
                parser.error("predict mode requires input_file")
            system.predict(args.input_file)
        
        elif args.mode == 'countermeasures':
            system.generate_countermeasures()
        
        elif args.mode == 'full':
            if not args.input_file:
                parser.error("full mode requires input_file")
            system.full_pipeline(args.input_file)
        
        logger.info("\n✓ Operation completed successfully!")
        
    except Exception as e:
        logger.error(f"\n✗ Error occurred: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()