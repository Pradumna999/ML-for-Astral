"""
Data preprocessing module for satellite collision prediction.
Handles JSON loading, feature extraction, and normalization.
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteDataPreprocessor:
    """Handles all data preprocessing for collision prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            'delta_inclination',
            'delta_raan',
            'delta_eccentricity',
            'delta_arg_perigee',
            'delta_mean_anomaly',
            'delta_mean_motion'
        ]
    
    def load_json_data(self, filepath: str) -> List[Dict]:
        """Load satellite data from JSON file."""
        logger.info(f"Loading data from {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Handle both 'dataset' and 'satellites' keys
        if 'dataset' in data:
            return data['dataset']
        elif 'satellites' in data:
            return data['satellites']
        else:
            raise KeyError(f"Expected 'dataset' or 'satellites' key in {filepath}")
    
    def circular_difference(self, angle1: float, angle2: float) -> float:
        """
        Compute circular difference between two angles in degrees.
        Returns value in range [-180, 180].
        """
        diff = angle1 - angle2
        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff
    
    def extract_features_from_pair(self, sat1: Dict, sat2: Dict) -> np.ndarray:
        """
        Extract delta features from two satellite records.
        
        Args:
            sat1: First satellite dictionary
            sat2: Second satellite dictionary
            
        Returns:
            np.ndarray: 6 delta features
        """
        params1 = sat1['orbital_parameters']
        params2 = sat2['orbital_parameters']
        
        features = np.array([
            # Angular differences (handle wrap-around)
            self.circular_difference(params1['inclination'], params2['inclination']),
            self.circular_difference(params1['raan'], params2['raan']),
            
            # Eccentricity (direct difference)
            params1['eccentricity'] - params2['eccentricity'],
            
            # Angular differences
            self.circular_difference(params1['argument_of_perigee'], 
                                    params2['argument_of_perigee']),
            self.circular_difference(params1['mean_anomaly'], 
                                    params2['mean_anomaly']),
            
            # Mean motion (revolutions per day)
            params1['mean_motion'] - params2['mean_motion']
        ])
        
        return features
    
    def prepare_training_data(self, 
                             collision_file: str, 
                             safe_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare complete training dataset from collision and safe files.
        
        Args:
            collision_file: Path to collision_data.json
            safe_file: Path to safe_data.json
            
        Returns:
            X: Feature matrix (N, 6)
            y: Labels (N,) - 1 for collision, 0 for safe
        """
        logger.info("Preparing training data...")
        
        # Load datasets
        collision_data = self.load_json_data(collision_file)
        safe_data = self.load_json_data(safe_file)
        
        # Group by pair_id to reconstruct pairs (collision data has pair_id)
        collision_pairs = self._group_by_pairs(collision_data)
        
        # For safe data: create random pairs if they don't have pair_id
        if safe_data and 'pair_id' not in safe_data[0]:
            # Create pairs from individual satellites randomly
            safe_pairs = self._create_random_pairs(safe_data)
        else:
            safe_pairs = self._group_by_pairs(safe_data)
        
        # Extract features
        X_collision = []
        for pair in collision_pairs:
            if len(pair) == 2:
                features = self.extract_features_from_pair(pair[0], pair[1])
                X_collision.append(features)
        
        X_safe = []
        for pair in safe_pairs:
            if len(pair) == 2:
                features = self.extract_features_from_pair(pair[0], pair[1])
                X_safe.append(features)
        
        # Combine features and labels
        X_collision = np.array(X_collision)
        X_safe = np.array(X_safe)
        
        X = np.vstack([X_collision, X_safe])
        y = np.hstack([np.ones(len(X_collision)), np.zeros(len(X_safe))])
        
        logger.info(f"Dataset prepared: {len(X)} samples")
        logger.info(f"  Collision samples: {len(X_collision)}")
        logger.info(f"  Safe samples: {len(X_safe)}")
        
        return X, y
    
    def _group_by_pairs(self, data: List[Dict]) -> List[List[Dict]]:
        """Group satellite records by pair_id."""
        pairs_dict = {}
        for record in data:
            pair_id = record.get('pair_id')
            if pair_id is None:
                continue
            if pair_id not in pairs_dict:
                pairs_dict[pair_id] = []
            pairs_dict[pair_id].append(record)
        return list(pairs_dict.values())
    
    def _create_random_pairs(self, data: List[Dict]) -> List[List[Dict]]:
        """Create random satellite pairs from individual records."""
        import random
        pairs = []
        n = len(data)
        
        # Create n//2 random pairs
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in range(0, n-1, 2):
            pairs.append([data[indices[i]], data[indices[i+1]]])
        
        logger.info(f"Created {len(pairs)} random satellite pairs from safe data")
        return pairs
    
    def fit_scaler(self, X: np.ndarray):
        """Fit the standard scaler on training data."""
        logger.info("Fitting scaler...")
        self.scaler.fit(X)
        
        # Log statistics
        logger.info("Feature statistics (mean, std):")
        for i, name in enumerate(self.feature_names):
            logger.info(f"  {name}: μ={self.scaler.mean_[i]:.4f}, "
                       f"σ={self.scaler.scale_[i]:.4f}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply scaling transformation."""
        return self.scaler.transform(X)
    
    def save_scaler(self, filepath: str):
        """Save fitted scaler to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load fitted scaler from disk."""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {filepath}")
    
    def prepare_prediction_data(self, input_file: str) -> Tuple[np.ndarray, List]:
        """
        Prepare data for prediction from user input JSON.
        
        Args:
            input_file: Path to input JSON with satellite TLEs
            
        Returns:
            X_scaled: Scaled feature matrix for all pairs
            pair_info: List of (sat1_info, sat2_info) tuples
        """
        logger.info(f"Preparing prediction data from {input_file}")
        
        satellites = self.load_json_data(input_file)
        
        # Generate all possible pairs
        features = []
        pair_info = []
        
        n = len(satellites)
        for i in range(n):
            for j in range(i + 1, n):
                sat1 = satellites[i]
                sat2 = satellites[j]
                
                # Extract features
                feat = self.extract_features_from_pair(sat1, sat2)
                features.append(feat)
                
                # Store pair information
                pair_info.append({
                    'sat1_id': sat1['satellite_id'],
                    'sat1_name': sat1['name'],
                    'sat2_id': sat2['satellite_id'],
                    'sat2_name': sat2['name'],
                    'sat1_norad': sat1['norad_id'],
                    'sat2_norad': sat2['norad_id']
                })
        
        X = np.array(features)
        X_scaled = self.transform(X)
        
        logger.info(f"Generated {len(features)} pairs for prediction")
        
        return X_scaled, pair_info


# Example usage
if __name__ == "__main__":
    preprocessor = SatelliteDataPreprocessor()
    
    # Prepare training data
    X, y = preprocessor.prepare_training_data(
        'data/raw/collision_data.json',
        'data/raw/safe_data.json'
    )
    
    # Fit and save scaler
    preprocessor.fit_scaler(X)
    X_scaled = preprocessor.transform(X)
    preprocessor.save_scaler('data/scalers/standard_scaler.pkl')
    
    print(f"X shape: {X_scaled.shape}")
    print(f"y shape: {y.shape}")