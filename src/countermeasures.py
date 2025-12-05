"""
Countermeasure generation for collision avoidance.
Suggests orbital maneuvers to reduce collision probability.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple
from copy import deepcopy

from src.data_preprocessing import SatelliteDataPreprocessor
from src.model_architecture import CollisionPredictionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollisionAvoidance:
    """
    Generates countermeasure suggestions for collision avoidance.
    Uses iterative optimization to find minimal orbital adjustments.
    """
    
    def __init__(self,
                 model_path: str,
                 scaler_path: str,
                 target_probability: float = 0.01):
        """
        Initialize collision avoidance system.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            target_probability: Target collision probability after maneuver
        """
        self.target_probability = target_probability
        
        # Load model and preprocessor
        self.preprocessor = SatelliteDataPreprocessor()
        self.preprocessor.load_scaler(scaler_path)
        
        self.model = CollisionPredictionModel()
        self.model.load_model(model_path)
        
        logger.info(f"Collision avoidance initialized (target p={target_probability})")
    
    def suggest_maneuvers(self, 
                         collision_warnings: List[Dict],
                         satellites_data: List[Dict]) -> Dict:
        """
        Generate countermeasure suggestions for all collision warnings.
        
        Args:
            collision_warnings: List of collision warning dictionaries
            satellites_data: Original satellite data from input JSON
            
        Returns:
            Dictionary with maneuver suggestions
        """
        logger.info("=" * 60)
        logger.info("GENERATING COUNTERMEASURE SUGGESTIONS")
        logger.info("=" * 60)
        
        maneuvers = {
            'metadata': {
                'target_probability': self.target_probability,
                'total_warnings': len(collision_warnings)
            },
            'maneuver_plans': []
        }
        
        # Create satellite lookup
        sat_lookup = {sat['satellite_id']: sat for sat in satellites_data}
        
        # Process each collision warning
        for i, warning in enumerate(collision_warnings, 1):
            logger.info(f"\nProcessing warning {i}/{len(collision_warnings)}: "
                       f"{warning['sat1_name']} - {warning['sat2_name']}")
            
            sat1_id = warning['sat1_id']
            sat2_id = warning['sat2_id']
            
            if sat1_id not in sat_lookup or sat2_id not in sat_lookup:
                logger.warning(f"Satellite data not found, skipping...")
                continue
            
            sat1 = sat_lookup[sat1_id]
            sat2 = sat_lookup[sat2_id]
            
            # Generate maneuvers for both satellites
            maneuver_options = []
            
            # Option 1: Adjust satellite 1
            option1 = self._find_minimal_maneuver(
                sat1, sat2, sat1,
                warning['collision_probability']
            )
            if option1:
                option1['active_satellite'] = sat1['name']
                option1['maneuver_type'] = 'PRIMARY'
                maneuver_options.append(option1)
            
            # Option 2: Adjust satellite 2
            option2 = self._find_minimal_maneuver(
                sat1, sat2, sat2,
                warning['collision_probability']
            )
            if option2:
                option2['active_satellite'] = sat2['name']
                option2['maneuver_type'] = 'SECONDARY'
                maneuver_options.append(option2)
            
            # Add to plans
            plan = {
                'warning': {
                    'sat1_name': warning['sat1_name'],
                    'sat2_name': warning['sat2_name'],
                    'initial_probability': warning['collision_probability'],
                    'risk_level': warning['risk_level']
                },
                'maneuver_options': maneuver_options,
                'recommendation': self._select_best_option(maneuver_options)
            }
            
            maneuvers['maneuver_plans'].append(plan)
        
        return maneuvers
    
    def _find_minimal_maneuver(self,
                               sat1: Dict,
                               sat2: Dict,
                               active_sat: Dict,
                               initial_prob: float) -> Dict:
        """
        Find minimal orbital adjustment to reduce collision probability.
        
        Args:
            sat1: First satellite data
            sat2: Second satellite data
            active_sat: Satellite to maneuver
            initial_prob: Initial collision probability
            
        Returns:
            Dictionary with maneuver parameters
        """
        # Parameters to adjust
        adjustable_params = [
            ('mean_motion', 0.001, 'rev/day'),      # Small altitude change
            ('mean_anomaly', 1.0, 'degrees'),       # Phase adjustment
            ('argument_of_perigee', 1.0, 'degrees') # Orientation change
        ]
        
        best_maneuver = None
        best_delta = float('inf')
        
        # Try each parameter
        for param_name, step_size, unit in adjustable_params:
            maneuver = self._optimize_parameter(
                sat1, sat2, active_sat,
                param_name, step_size,
                initial_prob
            )
            
            if maneuver and abs(maneuver['delta']) < abs(best_delta):
                best_delta = maneuver['delta']
                best_maneuver = {
                    'parameter': param_name,
                    'adjustment': maneuver['delta'],
                    'unit': unit,
                    'final_probability': maneuver['final_prob'],
                    'iterations': maneuver['iterations'],
                    'success': maneuver['final_prob'] <= self.target_probability
                }
        
        return best_maneuver
    
    def _optimize_parameter(self,
                           sat1: Dict,
                           sat2: Dict,
                           active_sat: Dict,
                           param_name: str,
                           step_size: float,
                           initial_prob: float,
                           max_iterations: int = 100) -> Dict:
        """
        Optimize a single orbital parameter to reduce collision probability.
        
        Args:
            sat1: First satellite
            sat2: Second satellite
            active_sat: Satellite to adjust
            param_name: Parameter to optimize
            step_size: Adjustment step size
            initial_prob: Initial collision probability
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        sat1_copy = deepcopy(sat1)
        sat2_copy = deepcopy(sat2)
        
        # Determine which satellite to modify
        if active_sat['satellite_id'] == sat1['satellite_id']:
            active = sat1_copy
        else:
            active = sat2_copy
        
        cumulative_delta = 0.0
        current_prob = initial_prob
        
        # Try positive direction
        for i in range(max_iterations):
            # Apply adjustment
            active['orbital_parameters'][param_name] += step_size
            cumulative_delta += step_size
            
            # Compute new probability
            features = self.preprocessor.extract_features_from_pair(sat1_copy, sat2_copy)
            features_scaled = self.preprocessor.transform(features.reshape(1, -1))
            new_prob = self.model.model.predict(features_scaled, verbose=0)[0][0]
            
            if new_prob <= self.target_probability:
                return {
                    'delta': cumulative_delta,
                    'final_prob': float(new_prob),
                    'iterations': i + 1
                }
            
            current_prob = new_prob
        
        # Try negative direction if positive didn't work
        active = sat1_copy if active_sat['satellite_id'] == sat1['satellite_id'] else sat2_copy
        active['orbital_parameters'][param_name] = sat1['orbital_parameters'][param_name]
        cumulative_delta = 0.0
        
        for i in range(max_iterations):
            active['orbital_parameters'][param_name] -= step_size
            cumulative_delta -= step_size
            
            features = self.preprocessor.extract_features_from_pair(sat1_copy, sat2_copy)
            features_scaled = self.preprocessor.transform(features.reshape(1, -1))
            new_prob = self.model.model.predict(features_scaled, verbose=0)[0][0]
            
            if new_prob <= self.target_probability:
                return {
                    'delta': cumulative_delta,
                    'final_prob': float(new_prob),
                    'iterations': i + 1
                }
        
        return None
    
    def _select_best_option(self, options: List[Dict]) -> Dict:
        """Select the best maneuver option based on minimal adjustment."""
        if not options:
            return {'recommendation': 'NO_VIABLE_MANEUVER_FOUND'}
        
        # Select option with smallest absolute adjustment
        best = min(options, 
                  key=lambda x: abs(x['adjustment']) if x['success'] else float('inf'))
        
        if best['success']:
            return {
                'selected': best['active_satellite'],
                'parameter': best['parameter'],
                'adjustment': best['adjustment'],
                'unit': best['unit'],
                'expected_probability': best['final_probability'],
                'status': 'VIABLE'
            }
        else:
            return {'recommendation': 'NO_VIABLE_MANEUVER_FOUND'}
    
    def save_maneuvers(self, maneuvers: Dict, output_file: str = 'outputs/countermeasures.json'):
        """Save countermeasure suggestions to file."""
        with open(output_file, 'w') as f:
            json.dump(maneuvers, f, indent=2)
        logger.info(f"\nCountermeasures saved to: {output_file}")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate collision countermeasures')
    parser.add_argument('collision_report', type=str, 
                       help='Path to collision report JSON')
    parser.add_argument('input_data', type=str,
                       help='Path to original satellite data JSON')
    parser.add_argument('--target-prob', type=float, default=0.01,
                       help='Target collision probability (default: 0.01)')
    
    args = parser.parse_args()
    
    # Load collision report
    with open(args.collision_report, 'r') as f:
        report = json.load(f)
    
    # Load satellite data
    with open(args.input_data, 'r') as f:
        sat_data = json.load(f)
    
    # Initialize avoidance system
    avoidance = CollisionAvoidance(
        model_path='models/saved_models/collision_model_best.keras',
        scaler_path='data/scalers/standard_scaler.pkl',
        target_probability=args.target_prob
    )
    
    # Generate maneuvers
    maneuvers = avoidance.suggest_maneuvers(
        report['collision_warnings'],
        sat_data['dataset']
    )
    
    # Save results
    avoidance.save_maneuvers(maneuvers)
    
    print("\n" + "=" * 60)
    print("COUNTERMEASURE GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Results saved to: outputs/countermeasures.json")