"""
Synthetic data generation for ML pipeline simulation.
Creates realistic datasets with controllable drift patterns.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SyntheticDataGenerator:
    """Generates synthetic datasets with controllable drift patterns."""
    
    def __init__(self, n_samples=1000, n_features=10, n_classes=3, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def generate_base_data(self):
        """Generate base dataset with classification problem."""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=6,
            n_redundant=2,
            n_repeated=2,
            n_classes=self.n_classes,
            n_clusters_per_class=1,
            random_state=self.random_state,
            class_sep=1.0
        )
        
        # Add some noise features
        noise_features = np.random.normal(0, 1, (self.n_samples, 3))
        X = np.hstack([X, noise_features])
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Create target names
        target_names = [f'class_{i}' for i in range(self.n_classes)]
        
        return pd.DataFrame(X, columns=feature_names), pd.Series(y, name='target')
    
    def generate_temporal_data(self, n_time_steps=100, drift_strength=0.1):
        """Generate temporal data with gradual drift."""
        all_data = []
        all_targets = []
        
        for t in range(n_time_steps):
            # Base drift: gradual change in feature distributions
            drift_factor = t * drift_strength / n_time_steps
            
            # Generate data with drift
            X, y = make_classification(
                n_samples=self.n_samples // n_time_steps,
                n_features=self.n_features,
                n_informative=6,
                n_redundant=2,
                n_repeated=2,
                n_classes=self.n_classes,
                n_clusters_per_class=1,
                random_state=self.random_state + t,
                class_sep=1.0
            )
            
            # Apply drift to features
            X = X + np.random.normal(drift_factor, 0.1, X.shape)
            
            # Add noise features with increasing variance
            noise_features = np.random.normal(0, 1 + drift_factor, (X.shape[0], 3))
            X = np.hstack([X, noise_features])
            
            # Add temporal features
            temporal_features = np.column_stack([
                np.sin(2 * np.pi * t / n_time_steps),
                np.cos(2 * np.pi * t / n_time_steps),
                t / n_time_steps
            ])
            temporal_features = np.repeat(temporal_features, X.shape[0], axis=0)
            X = np.hstack([X, temporal_features])
            
            all_data.append(X)
            all_targets.append(y)
        
        # Combine all time steps
        X_combined = np.vstack(all_data)
        y_combined = np.concatenate(all_targets)
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(X_combined.shape[1])]
        
        return pd.DataFrame(X_combined, columns=feature_names), pd.Series(y_combined, name='target')
    
    def generate_concept_drift_data(self, n_time_steps=100, concept_change_points=[30, 70]):
        """Generate data with concept drift at specific time points."""
        all_data = []
        all_targets = []
        
        for t in range(n_time_steps):
            # Determine if we're in a concept drift period
            in_drift = any(t >= change_point for change_point in concept_change_points)
            
            if in_drift:
                # Change the relationship between features and target
                X, y = make_classification(
                    n_samples=self.n_samples // n_time_steps,
                    n_features=self.n_features,
                    n_informative=4,  # Fewer informative features
                    n_redundant=4,    # More redundant features
                    n_repeated=2,
                    n_classes=self.n_classes,
                    n_clusters_per_class=1,
                    random_state=self.random_state + t + 1000,  # Different random state
                    class_sep=0.5  # Lower separation
                )
            else:
                # Normal data generation
                X, y = make_classification(
                    n_samples=self.n_samples // n_time_steps,
                    n_features=self.n_features,
                    n_informative=6,
                    n_redundant=2,
                    n_repeated=2,
                    n_classes=self.n_classes,
                    n_clusters_per_class=1,
                    random_state=self.random_state + t,
                    class_sep=1.0
                )
            
            # Add noise features
            noise_features = np.random.normal(0, 1, (X.shape[0], 3))
            X = np.hstack([X, noise_features])
            
            all_data.append(X)
            all_targets.append(y)
        
        # Combine all time steps
        X_combined = np.vstack(all_data)
        y_combined = np.concatenate(all_targets)
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(X_combined.shape[1])]
        
        return pd.DataFrame(X_combined, columns=feature_names), pd.Series(y_combined, name='target')
    
    def generate_covariate_shift_data(self, n_time_steps=100, shift_strength=0.5):
        """Generate data with covariate shift (changes in feature distributions)."""
        all_data = []
        all_targets = []
        
        for t in range(n_time_steps):
            # Generate base data
            X, y = make_classification(
                n_samples=self.n_samples // n_time_steps,
                n_features=self.n_features,
                n_informative=6,
                n_redundant=2,
                n_repeated=2,
                n_classes=self.n_classes,
                n_clusters_per_class=1,
                random_state=self.random_state + t,
                class_sep=1.0
            )
            
            # Apply covariate shift: change feature distributions over time
            shift_factor = np.sin(2 * np.pi * t / n_time_steps) * shift_strength
            X = X + np.random.normal(shift_factor, 0.2, X.shape)
            
            # Add noise features with varying distributions
            noise_std = 1 + 0.5 * np.sin(2 * np.pi * t / n_time_steps)
            noise_features = np.random.normal(0, noise_std, (X.shape[0], 3))
            X = np.hstack([X, noise_features])
            
            all_data.append(X)
            all_targets.append(y)
        
        # Combine all time steps
        X_combined = np.vstack(all_data)
        y_combined = np.concatenate(all_targets)
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(X_combined.shape[1])]
        
        return pd.DataFrame(X_combined, columns=feature_names), pd.Series(y_combined, name='target')


def create_mnist_simulation_data(n_samples=1000, n_time_steps=50):
    """Create MNIST-like simulation data with drift."""
    generator = SyntheticDataGenerator(n_samples=n_samples, n_features=64, n_classes=10)
    
    # Generate base data
    X, y = generator.generate_base_data()
    
    # Reshape to simulate image-like data (8x8)
    X_reshaped = X.iloc[:, :64].values.reshape(-1, 8, 8)
    
    # Add temporal drift
    all_data = []
    all_targets = []
    
    for t in range(n_time_steps):
        # Add drift to the "image" data
        drift_factor = t / n_time_steps
        X_drifted = X_reshaped + np.random.normal(drift_factor * 0.5, 0.1, X_reshaped.shape)
        
        # Flatten back to features
        X_flat = X_drifted.reshape(-1, 64)
        
        # Add noise features
        noise_features = np.random.normal(0, 1, (X_flat.shape[0], 10))
        X_final = np.hstack([X_flat, noise_features])
        
        all_data.append(X_final)
        all_targets.append(y.values)
    
    # Combine all time steps
    X_combined = np.vstack(all_data)
    y_combined = np.concatenate(all_targets)
    
    # Create feature names
    feature_names = [f'pixel_{i}' for i in range(64)] + [f'noise_{i}' for i in range(10)]
    
    return pd.DataFrame(X_combined, columns=feature_names), pd.Series(y_combined, name='target')


if __name__ == "__main__":
    # Test the data generator
    generator = SyntheticDataGenerator()
    
    print("Generating base data...")
    X_base, y_base = generator.generate_base_data()
    print(f"Base data shape: {X_base.shape}")
    
    print("\nGenerating temporal data with drift...")
    X_temp, y_temp = generator.generate_temporal_data(n_time_steps=50)
    print(f"Temporal data shape: {X_temp.shape}")
    
    print("\nGenerating concept drift data...")
    X_concept, y_concept = generator.generate_concept_drift_data()
    print(f"Concept drift data shape: {X_concept.shape}")
    
    print("\nGenerating covariate shift data...")
    X_covariate, y_covariate = generator.generate_covariate_shift_data()
    print(f"Covariate shift data shape: {X_covariate.shape}")
    
    print("\nAll data generators working correctly!") 