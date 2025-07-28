"""
Data drift simulation for ML pipeline testing.
Injects various types of drift patterns into synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class DriftSimulator:
    """Simulates various types of data drift in ML pipelines."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.drift_history = []
        
    def inject_gradual_drift(self, data, time_steps, drift_strength=0.1, feature_subset=None):
        """
        Inject gradual drift into the data over time.
        
        Args:
            data: Input DataFrame
            time_steps: Number of time steps to simulate
            drift_strength: Strength of the drift (0-1)
            feature_subset: Specific features to apply drift to
        """
        drifted_data = []
        
        for t in range(time_steps):
            # Calculate drift factor
            drift_factor = t * drift_strength / time_steps
            
            # Apply drift to features
            if feature_subset is None:
                features_to_drift = data.columns
            else:
                features_to_drift = feature_subset
            
            data_drifted = data.copy()
            
            for feature in features_to_drift:
                if feature in data.columns:
                    # Add gradual shift and increasing noise
                    shift = np.random.normal(drift_factor, 0.1)
                    noise = np.random.normal(0, drift_factor * 0.5, len(data))
                    data_drifted[feature] = data_drifted[feature] + shift + noise
            
            drifted_data.append(data_drifted)
            
            # Record drift info
            self.drift_history.append({
                'time_step': t,
                'drift_type': 'gradual',
                'drift_factor': drift_factor,
                'affected_features': list(features_to_drift)
            })
        
        return drifted_data
    
    def inject_sudden_drift(self, data, time_steps, change_points=[30, 70], drift_magnitude=1.0):
        """
        Inject sudden drift at specific time points.
        
        Args:
            data: Input DataFrame
            time_steps: Number of time steps to simulate
            change_points: List of time steps where drift occurs
            drift_magnitude: Magnitude of the sudden change
        """
        drifted_data = []
        
        for t in range(time_steps):
            data_drifted = data.copy()
            
            # Check if this is a change point
            if t in change_points:
                # Apply sudden shift to all features
                for col in data.columns:
                    shift = np.random.normal(0, drift_magnitude)
                    data_drifted[col] = data_drifted[col] + shift
                
                self.drift_history.append({
                    'time_step': t,
                    'drift_type': 'sudden',
                    'drift_magnitude': drift_magnitude,
                    'affected_features': list(data.columns)
                })
            else:
                self.drift_history.append({
                    'time_step': t,
                    'drift_type': 'none',
                    'drift_magnitude': 0,
                    'affected_features': []
                })
            
            drifted_data.append(data_drifted)
        
        return drifted_data
    
    def inject_concept_drift(self, data, targets, time_steps, change_points=[30, 70]):
        """
        Inject concept drift by changing the relationship between features and targets.
        
        Args:
            data: Input DataFrame
            targets: Target variable
            time_steps: Number of time steps to simulate
            change_points: List of time steps where concept drift occurs
        """
        drifted_data = []
        drifted_targets = []
        
        for t in range(time_steps):
            data_drifted = data.copy()
            targets_drifted = targets.copy()
            
            # Check if this is a concept drift point
            if t in change_points:
                # Simulate concept drift by adding noise to the relationship
                # This makes the features less predictive of the target
                noise_factor = 0.5
                
                # Add noise to features that are most correlated with target
                for col in data.columns:
                    if abs(data[col].corr(targets)) > 0.1:  # If feature is somewhat correlated
                        noise = np.random.normal(0, noise_factor, len(data))
                        data_drifted[col] = data_drifted[col] + noise
                
                # Also add some noise to targets to simulate changing relationships
                target_noise = np.random.normal(0, 0.3, len(targets))
                targets_drifted = targets_drifted + target_noise
                targets_drifted = np.round(np.clip(targets_drifted, 0, targets.max())).astype(int)
                
                self.drift_history.append({
                    'time_step': t,
                    'drift_type': 'concept',
                    'noise_factor': noise_factor,
                    'affected_features': list(data.columns)
                })
            else:
                self.drift_history.append({
                    'time_step': t,
                    'drift_type': 'none',
                    'noise_factor': 0,
                    'affected_features': []
                })
            
            drifted_data.append(data_drifted)
            drifted_targets.append(targets_drifted)
        
        return drifted_data, drifted_targets
    
    def inject_covariate_shift(self, data, time_steps, shift_pattern='cyclic'):
        """
        Inject covariate shift by changing feature distributions.
        
        Args:
            data: Input DataFrame
            time_steps: Number of time steps to simulate
            shift_pattern: 'cyclic', 'linear', or 'random'
        """
        drifted_data = []
        
        for t in range(time_steps):
            data_drifted = data.copy()
            
            if shift_pattern == 'cyclic':
                # Cyclic shift using sine wave
                shift_factor = np.sin(2 * np.pi * t / time_steps)
            elif shift_pattern == 'linear':
                # Linear shift
                shift_factor = t / time_steps
            elif shift_pattern == 'random':
                # Random shift
                shift_factor = np.random.normal(0, 1)
            else:
                shift_factor = 0
            
            # Apply shift to all features
            for col in data.columns:
                shift = np.random.normal(shift_factor, 0.2)
                data_drifted[col] = data_drifted[col] + shift
            
            self.drift_history.append({
                'time_step': t,
                'drift_type': 'covariate_shift',
                'shift_pattern': shift_pattern,
                'shift_factor': shift_factor,
                'affected_features': list(data.columns)
            })
            
            drifted_data.append(data_drifted)
        
        return drifted_data
    
    def inject_noise(self, data, time_steps, noise_level=0.1, increasing_noise=True):
        """
        Inject noise into the data over time.
        
        Args:
            data: Input DataFrame
            time_steps: Number of time steps to simulate
            noise_level: Base noise level
            increasing_noise: Whether noise increases over time
        """
        drifted_data = []
        
        for t in range(time_steps):
            data_drifted = data.copy()
            
            # Calculate noise level for this time step
            if increasing_noise:
                current_noise = noise_level * (1 + t / time_steps)
            else:
                current_noise = noise_level
            
            # Add noise to all features
            for col in data.columns:
                noise = np.random.normal(0, current_noise, len(data))
                data_drifted[col] = data_drifted[col] + noise
            
            self.drift_history.append({
                'time_step': t,
                'drift_type': 'noise',
                'noise_level': current_noise,
                'affected_features': list(data.columns)
            })
            
            drifted_data.append(data_drifted)
        
        return drifted_data
    
    def inject_missing_data_drift(self, data, time_steps, missing_rate=0.1, increasing_missing=True):
        """
        Inject drift by gradually increasing missing data.
        
        Args:
            data: Input DataFrame
            time_steps: Number of time steps to simulate
            missing_rate: Base missing data rate
            increasing_missing: Whether missing data increases over time
        """
        drifted_data = []
        
        for t in range(time_steps):
            data_drifted = data.copy()
            
            # Calculate missing rate for this time step
            if increasing_missing:
                current_missing_rate = missing_rate * (1 + t / time_steps)
            else:
                current_missing_rate = missing_rate
            
            # Randomly set values to NaN
            for col in data.columns:
                mask = np.random.random(len(data)) < current_missing_rate
                data_drifted.loc[mask, col] = np.nan
            
            self.drift_history.append({
                'time_step': t,
                'drift_type': 'missing_data',
                'missing_rate': current_missing_rate,
                'affected_features': list(data.columns)
            })
            
            drifted_data.append(data_drifted)
        
        return drifted_data
    
    def get_drift_summary(self):
        """Get a summary of all drift events."""
        if not self.drift_history:
            return "No drift events recorded."
        
        summary = {
            'total_time_steps': len(self.drift_history),
            'drift_types': {},
            'change_points': []
        }
        
        for event in self.drift_history:
            drift_type = event['drift_type']
            if drift_type not in summary['drift_types']:
                summary['drift_types'][drift_type] = 0
            summary['drift_types'][drift_type] += 1
            
            if drift_type != 'none':
                summary['change_points'].append(event['time_step'])
        
        return summary
    
    def plot_drift_timeline(self):
        """Create a simple timeline plot of drift events."""
        import matplotlib.pyplot as plt
        
        if not self.drift_history:
            print("No drift history to plot.")
            return
        
        time_steps = [event['time_step'] for event in self.drift_history]
        drift_types = [event['drift_type'] for event in self.drift_history]
        
        # Create color mapping for drift types
        unique_types = list(set(drift_types))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        color_map = dict(zip(unique_types, colors))
        
        plt.figure(figsize=(12, 6))
        for i, (t, drift_type) in enumerate(zip(time_steps, drift_types)):
            plt.scatter(t, 0, c=[color_map[drift_type]], s=100, alpha=0.7)
        
        plt.xlabel('Time Step')
        plt.ylabel('Drift Events')
        plt.title('Data Drift Timeline')
        plt.grid(True, alpha=0.3)
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map[t], markersize=10, label=t)
                          for t in unique_types]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()


def simulate_complex_drift_scenario(data, targets, time_steps=100):
    """
    Simulate a complex drift scenario combining multiple types of drift.
    
    Args:
        data: Input DataFrame
        targets: Target variable
        time_steps: Number of time steps to simulate
    """
    simulator = DriftSimulator()
    
    # Combine multiple drift types
    drifted_data = []
    drifted_targets = []
    
    for t in range(time_steps):
        data_t = data.copy()
        targets_t = targets.copy()
        
        # Gradual drift (always present)
        drift_factor = t * 0.05 / time_steps
        for col in data.columns:
            shift = np.random.normal(drift_factor, 0.1)
            data_t[col] = data_t[col] + shift
        
        # Sudden changes at specific points
        if t in [25, 75]:
            for col in data.columns:
                sudden_shift = np.random.normal(0, 0.5)
                data_t[col] = data_t[col] + sudden_shift
        
        # Concept drift at middle point
        if t == 50:
            # Add noise to predictive features
            for col in data.columns:
                if abs(data_t[col].corr(targets_t)) > 0.1:
                    noise = np.random.normal(0, 0.3, len(data_t))
                    data_t[col] = data_t[col] + noise
        
        # Cyclic covariate shift
        shift_factor = np.sin(2 * np.pi * t / time_steps) * 0.3
        for col in data.columns:
            data_t[col] = data_t[col] + shift_factor
        
        # Increasing noise
        noise_level = 0.1 * (1 + t / time_steps)
        for col in data.columns:
            noise = np.random.normal(0, noise_level, len(data_t))
            data_t[col] = data_t[col] + noise
        
        drifted_data.append(data_t)
        drifted_targets.append(targets_t)
        
        # Record drift info
        simulator.drift_history.append({
            'time_step': t,
            'drift_type': 'complex',
            'drift_factor': drift_factor,
            'noise_level': noise_level,
            'affected_features': list(data.columns)
        })
    
    return drifted_data, drifted_targets, simulator


if __name__ == "__main__":
    # Test the drift simulator
    from synthetic_data import SyntheticDataGenerator
    
    # Generate test data
    generator = SyntheticDataGenerator(n_samples=1000, n_features=10)
    X, y = generator.generate_base_data()
    
    # Test different drift types
    simulator = DriftSimulator()
    
    print("Testing gradual drift...")
    gradual_data = simulator.inject_gradual_drift(X, time_steps=50)
    print(f"Generated {len(gradual_data)} time steps of gradual drift")
    
    print("\nTesting sudden drift...")
    sudden_data = simulator.inject_sudden_drift(X, time_steps=50)
    print(f"Generated {len(sudden_data)} time steps of sudden drift")
    
    print("\nTesting concept drift...")
    concept_data, concept_targets = simulator.inject_concept_drift(X, y, time_steps=50)
    print(f"Generated {len(concept_data)} time steps of concept drift")
    
    print("\nTesting complex drift scenario...")
    complex_data, complex_targets, complex_simulator = simulate_complex_drift_scenario(X, y, time_steps=50)
    print(f"Generated {len(complex_data)} time steps of complex drift")
    
    # Print drift summary
    print("\nDrift Summary:")
    summary = complex_simulator.get_drift_summary()
    print(summary)
    
    print("\nAll drift simulation tests passed!") 