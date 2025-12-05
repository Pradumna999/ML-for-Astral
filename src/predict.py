"""
Prediction pipeline for collision detection.
Handles inference on new satellite TLE data.
"""

import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from src.data_preprocessing import SatelliteDataPreprocessor
from src.model_architecture import CollisionPredictionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollisionPredictor:
    """
    Handles collision prediction for satellite pairs.
    """
    
    def __init__(self, 
                 model_path: str = 'models/saved_models/collision_model_best.keras',
                 scaler_path: str = 'data/scalers/standard_scaler.pkl',
                 threshold: float = 0.1):
        """
        Initialize predictor with trained model and scaler.
        
        Args:
            model_path: Path to trained Keras model
            scaler_path: Path to fitted StandardScaler
            threshold: Probability threshold for collision warning (default 0.1)
        """
        self.threshold = threshold
        
        # Load preprocessor and scaler
        self.preprocessor = SatelliteDataPreprocessor()
        self.preprocessor.load_scaler(scaler_path)
        
        # Load model
        self.model = CollisionPredictionModel()
        self.model.load_model(model_path)
        
        logger.info(f"Predictor initialized with threshold: {self.threshold}")
    
    def predict_collisions(self, input_file: str) -> Dict:
        """
        Predict collisions for all satellite pairs in input file.
        
        Args:
            input_file: Path to JSON file with satellite TLEs
            
        Returns:
            Dictionary with prediction results
        """
        logger.info("=" * 60)
        logger.info("COLLISION PREDICTION ANALYSIS")
        logger.info("=" * 60)
        
        # Prepare data
        X_scaled, pair_info = self.preprocessor.prepare_prediction_data(input_file)
        
        # Make predictions
        logger.info(f"Analyzing {len(pair_info)} satellite pairs...")
        probabilities, predictions = self.model.predict(X_scaled, threshold=self.threshold)
        
        # Organize results
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_file': input_file,
                'total_pairs_analyzed': len(pair_info),
                'threshold': self.threshold,
                'model_used': 'collision_model_best.keras'
            },
            'collision_warnings': [],
            'safe_pairs': [],
            'summary': {}
        }
        
        # Categorize pairs
        high_risk = []
        medium_risk = []
        low_risk = []
        safe = []
        
        for i, (prob, pred, pair) in enumerate(zip(probabilities, predictions, pair_info)):
            pair_result = {
                **pair,
                'collision_probability': float(prob),
                'risk_level': self._categorize_risk(prob)
            }
            
            if prob >= self.threshold:
                results['collision_warnings'].append(pair_result)
                
                if prob >= 0.7:
                    high_risk.append(pair_result)
                elif prob >= 0.3:
                    medium_risk.append(pair_result)
                else:
                    low_risk.append(pair_result)
            else:
                results['safe_pairs'].append(pair_result)
                safe.append(pair_result)
        
        # Sort warnings by probability (descending)
        results['collision_warnings'].sort(
            key=lambda x: x['collision_probability'], 
            reverse=True
        )
        
        # Create summary
        results['summary'] = {
            'total_warnings': len(results['collision_warnings']),
            'high_risk': len(high_risk),
            'medium_risk': len(medium_risk),
            'low_risk': len(low_risk),
            'safe_pairs': len(safe),
            'max_probability': float(max(probabilities)) if len(probabilities) > 0 else 0,
            'critical_pairs': [
                {
                    'pair': f"{p['sat1_name']} - {p['sat2_name']}",
                    'probability': p['collision_probability']
                }
                for p in high_risk[:5]  # Top 5 critical
            ]
        }
        
        # Log summary
        self._log_summary(results['summary'])
        
        return results
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level based on probability."""
        if probability >= 0.7:
            return 'HIGH_RISK'
        elif probability >= 0.3:
            return 'MEDIUM_RISK'
        elif probability >= self.threshold:
            return 'LOW_RISK'
        else:
            return 'SAFE'
    
    def _log_summary(self, summary: Dict):
        """Log prediction summary."""
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total collision warnings: {summary['total_warnings']}")
        logger.info(f"  High risk: {summary['high_risk']}")
        logger.info(f"  Medium risk: {summary['medium_risk']}")
        logger.info(f"  Low risk: {summary['low_risk']}")
        logger.info(f"Safe pairs: {summary['safe_pairs']}")
        logger.info(f"Maximum probability: {summary['max_probability']:.4f}")
        
        if summary['critical_pairs']:
            logger.info("\nTop critical pairs:")
            for i, pair in enumerate(summary['critical_pairs'], 1):
                logger.info(f"  {i}. {pair['pair']}: {pair['probability']:.4f}")
    
    def save_results(self, results: Dict, output_file: str = 'outputs/collision_report.json'):
        """Save prediction results to JSON file."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_file}")
    
    def generate_text_report(self, results: Dict, output_file: str = 'outputs/collision_report.txt'):
        """Generate human-readable text report."""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SATELLITE COLLISION PREDICTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            f.write(f"Generated: {results['metadata']['timestamp']}\n")
            f.write(f"Input file: {results['metadata']['input_file']}\n")
            f.write(f"Threshold: {results['metadata']['threshold']}\n")
            f.write(f"Total pairs analyzed: {results['metadata']['total_pairs_analyzed']}\n\n")
            
            # Summary
            summary = results['summary']
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total collision warnings: {summary['total_warnings']}\n")
            f.write(f"  - High risk (p >= 0.7): {summary['high_risk']}\n")
            f.write(f"  - Medium risk (0.3 <= p < 0.7): {summary['medium_risk']}\n")
            f.write(f"  - Low risk (threshold <= p < 0.3): {summary['low_risk']}\n")
            f.write(f"Safe pairs: {summary['safe_pairs']}\n\n")
            
            # Collision warnings
            if results['collision_warnings']:
                f.write("\nCOLLISION WARNINGS (sorted by probability)\n")
                f.write("=" * 80 + "\n\n")
                
                for i, warning in enumerate(results['collision_warnings'], 1):
                    f.write(f"WARNING #{i} - {warning['risk_level']}\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Satellite 1: {warning['sat1_name']} (ID: {warning['sat1_id']}, NORAD: {warning['sat1_norad']})\n")
                    f.write(f"Satellite 2: {warning['sat2_name']} (ID: {warning['sat2_id']}, NORAD: {warning['sat2_norad']})\n")
                    f.write(f"Collision probability: {warning['collision_probability']:.6f}\n")
                    f.write(f"Risk level: {warning['risk_level']}\n\n")
            else:
                f.write("\nNo collision warnings detected.\n")
        
        logger.info(f"Text report saved to: {output_file}")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict satellite collisions')
    parser.add_argument('input_file', type=str, help='Path to input JSON file')
    parser.add_argument('--threshold', type=float, default=0.1, 
                       help='Collision probability threshold (default: 0.1)')
    parser.add_argument('--model', type=str, 
                       default='models/saved_models/collision_model_best.keras',
                       help='Path to trained model')
    parser.add_argument('--scaler', type=str,
                       default='data/scalers/standard_scaler.pkl',
                       help='Path to fitted scaler')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CollisionPredictor(
        model_path=args.model,
        scaler_path=args.scaler,
        threshold=args.threshold
    )
    
    # Run prediction
    results = predictor.predict_collisions(args.input_file)
    
    # Save results
    predictor.save_results(results)
    predictor.generate_text_report(results)
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE!")
    print("=" * 60)
    print(f"JSON report: outputs/collision_report.json")
    print(f"Text report: outputs/collision_report.txt")