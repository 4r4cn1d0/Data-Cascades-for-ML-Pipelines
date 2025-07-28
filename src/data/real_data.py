"""
Real MNIST data handling with drift simulation.
Downloads and processes real MNIST data from Kaggle.
"""

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MNISTDataHandler:
    """Handles real MNIST data with drift simulation."""
    
    def __init__(self, data_dir='data', download_if_missing=True):
        self.data_dir = data_dir
        self.download_if_missing = download_if_missing
        self.scaler = StandardScaler()
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
    def download_mnist_data(self):
        """Download MNIST data from Kaggle."""
        try:
            import kaggle
            print("Downloading MNIST data from Kaggle...")
            
            # Download MNIST dataset
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'cdeotte/mnist-in-csv',
                path=self.data_dir,
                unzip=True
            )
            
            print("MNIST data downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Please ensure you have Kaggle API credentials set up.")
            return False
    
    def load_mnist_data(self):
        """Load MNIST data from CSV files."""
        train_path = os.path.join(self.data_dir, 'mnist_train.csv')
        test_path = os.path.join(self.data_dir, 'mnist_test.csv')
        
        # Check if files exist, download if missing
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            if self.download_if_missing:
                if not self.download_mnist_data():
                    return None, None, None, None
            else:
                print("MNIST data files not found. Please download manually.")
                return None, None, None, None
        
        print("Loading MNIST data...")
        
        # Load training data
        train_data = pd.read_csv(train_path)
        X_train = train_data.iloc[:, 1:].values  # Skip label column
        y_train = train_data.iloc[:, 0].values
        
        # Load test data
        test_data = pd.read_csv(test_path)
        X_test = test_data.iloc[:, 1:].values  # Skip label column
        y_test = test_data.iloc[:, 0].values
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        return X_train, y_train, X_test, y_test
    
    def preprocess_mnist_data(self, X_train, y_train, X_test, y_test, 
                             resize_to=(16, 16), normalize=True):
        """Preprocess MNIST data for drift simulation."""
        print("Preprocessing MNIST data...")
        
        # Resize images to smaller size for faster processing
        X_train_resized = self._resize_images(X_train, resize_to)
        X_test_resized = self._resize_images(X_test, resize_to)
        
        # Normalize pixel values
        if normalize:
            X_train_resized = X_train_resized / 255.0
            X_test_resized = X_test_resized / 255.0
        
        # Flatten images
        X_train_flat = X_train_resized.reshape(X_train_resized.shape[0], -1)
        X_test_flat = X_test_resized.reshape(X_test_resized.shape[0], -1)
        
        print(f"Preprocessed training data shape: {X_train_flat.shape}")
        print(f"Preprocessed test data shape: {X_test_flat.shape}")
        
        return X_train_flat, y_train, X_test_flat, y_test
    
    def _resize_images(self, X, target_size):
        """Resize images to target size."""
        n_samples = X.shape[0]
        img_size = int(np.sqrt(X.shape[1]))
        
        # Reshape to square images
        X_img = X.reshape(n_samples, img_size, img_size)
        
        # Resize each image
        X_resized = np.zeros((n_samples, target_size[0], target_size[1]))
        for i in range(n_samples):
            img = X_img[i].astype(np.uint8)
            resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            X_resized[i] = resized
        
        return X_resized
    
    def simulate_mnist_drift(self, X, y, n_time_steps=50, drift_types=['gradual', 'noise', 'blur']):
        """Simulate realistic drift in MNIST data."""
        print(f"Simulating MNIST drift over {n_time_steps} time steps...")
        
        all_data = []
        all_targets = []
        
        for t in range(n_time_steps):
            X_drifted = X.copy()
            
            # Apply different types of drift
            if 'gradual' in drift_types:
                X_drifted = self._apply_gradual_drift(X_drifted, t, n_time_steps)
            
            if 'noise' in drift_types:
                X_drifted = self._apply_noise_drift(X_drifted, t, n_time_steps)
            
            if 'blur' in drift_types:
                X_drifted = self._apply_blur_drift(X_drifted, t, n_time_steps)
            
            all_data.append(X_drifted)
            all_targets.append(y)
        
        return all_data, all_targets
    
    def _apply_gradual_drift(self, X, time_step, total_steps):
        """Apply gradual drift to MNIST data."""
        drift_factor = time_step / total_steps
        
        # Gradually shift pixel values
        shift = np.random.normal(drift_factor * 0.1, 0.05, X.shape)
        X_drifted = X + shift
        
        # Clip to valid range [0, 1]
        X_drifted = np.clip(X_drifted, 0, 1)
        
        return X_drifted
    
    def _apply_noise_drift(self, X, time_step, total_steps):
        """Apply increasing noise to MNIST data."""
        noise_level = (time_step / total_steps) * 0.3
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        
        # Clip to valid range [0, 1]
        X_noisy = np.clip(X_noisy, 0, 1)
        
        return X_noisy
    
    def _apply_blur_drift(self, X, time_step, total_steps):
        """Apply blur drift to MNIST data."""
        blur_factor = int(1 + (time_step / total_steps) * 3)
        
        # Reshape to images
        img_size = int(np.sqrt(X.shape[1]))
        X_img = X.reshape(-1, img_size, img_size)
        
        X_blurred = np.zeros_like(X_img)
        
        for i in range(X_img.shape[0]):
            img = (X_img[i] * 255).astype(np.uint8)
            
            # Apply Gaussian blur
            if blur_factor > 1:
                blurred = cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)
            else:
                blurred = img
            
            # Normalize back to [0, 1]
            X_blurred[i] = blurred.astype(np.float32) / 255.0
        
        # Flatten back
        X_blurred_flat = X_blurred.reshape(X.shape[0], -1)
        
        return X_blurred_flat
    
    def create_mnist_pipeline_data(self, n_samples_per_step=100, n_time_steps=50):
        """Create MNIST data for pipeline simulation."""
        # Load and preprocess data
        X_train, y_train, X_test, y_test = self.load_mnist_data()
        
        if X_train is None:
            print("Could not load MNIST data. Using synthetic data instead.")
            return self._create_fallback_data()
        
        # Use a subset for faster processing
        if n_samples_per_step * n_time_steps > len(X_train):
            n_samples_per_step = len(X_train) // n_time_steps
        
        # Preprocess data
        X_train_flat, y_train, X_test_flat, y_test = self.preprocess_mnist_data(
            X_train, y_train, X_test, y_test, resize_to=(16, 16)
        )
        
        # Take subset for simulation
        subset_size = n_samples_per_step * n_time_steps
        indices = np.random.choice(len(X_train_flat), subset_size, replace=False)
        X_subset = X_train_flat[indices]
        y_subset = y_train[indices]
        
        # Simulate drift
        drifted_data, drifted_targets = self.simulate_mnist_drift(
            X_subset, y_subset, n_time_steps=n_time_steps
        )
        
        # Create feature names
        feature_names = [f'pixel_{i}' for i in range(X_subset.shape[1])]
        
        return drifted_data, drifted_targets, feature_names
    
    def _create_fallback_data(self):
        """Create fallback synthetic data if MNIST is not available."""
        print("Creating fallback synthetic data...")
        
        # Generate synthetic data similar to MNIST
        n_samples = 1000
        n_features = 256  # 16x16 images
        n_classes = 10
        
        X = np.random.random((n_samples, n_features))
        y = np.random.randint(0, n_classes, n_samples)
        
        # Simulate drift
        drifted_data = []
        drifted_targets = []
        
        for t in range(50):
            drift_factor = t / 50
            X_drifted = X + np.random.normal(drift_factor * 0.1, 0.05, X.shape)
            X_drifted = np.clip(X_drifted, 0, 1)
            
            drifted_data.append(X_drifted)
            drifted_targets.append(y)
        
        feature_names = [f'pixel_{i}' for i in range(n_features)]
        
        return drifted_data, drifted_targets, feature_names
    
    def visualize_mnist_drift(self, drifted_data, sample_indices=[0, 1, 2], time_steps=[0, 25, 49]):
        """Visualize MNIST drift over time."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(sample_indices), len(time_steps), figsize=(12, 8))
        
        for i, sample_idx in enumerate(sample_indices):
            for j, time_step in enumerate(time_steps):
                if len(sample_indices) == 1:
                    ax = axes[j]
                elif len(time_steps) == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
                
                # Get image data
                img_data = drifted_data[time_step][sample_idx]
                img_size = int(np.sqrt(len(img_data)))
                img = img_data.reshape(img_size, img_size)
                
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Sample {sample_idx}, Time {time_step}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()


def get_mnist_data_for_pipeline(n_samples_per_step=100, n_time_steps=50):
    """Convenience function to get MNIST data for pipeline simulation."""
    handler = MNISTDataHandler()
    return handler.create_mnist_pipeline_data(n_samples_per_step, n_time_steps)


if __name__ == "__main__":
    # Test MNIST data handling
    handler = MNISTDataHandler()
    
    print("Testing MNIST data handling...")
    
    # Try to load MNIST data
    drifted_data, drifted_targets, feature_names = handler.create_mnist_pipeline_data(
        n_samples_per_step=50, n_time_steps=20
    )
    
    print(f"Generated {len(drifted_data)} time steps of drifted MNIST data")
    print(f"Data shape: {drifted_data[0].shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of classes: {len(np.unique(drifted_targets[0]))}")
    
    print("MNIST data handling working correctly!") 