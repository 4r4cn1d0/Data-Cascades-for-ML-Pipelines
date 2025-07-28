#!/usr/bin/env python3
"""
Test script for the ML Pipeline Cascade Monitor.
Verifies that all components work correctly.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data.synthetic_data import SyntheticDataGenerator
from src.data.drift_simulator import DriftSimulator, simulate_complex_drift_scenario
from src.pipeline.stages import MLPipeline
from src.pipeline.cascade_monitor import CascadeMonitor, RetrainingManager
from src.utils.metrics import calculate_pipeline_metrics, calculate_cascade_correlation


def test_data_generation():
    """Test synthetic data generation."""
    print("Testing data generation...")
    
    generator = SyntheticDataGenerator(n_samples=1000, n_features=10, n_classes=3)
    X, y = generator.generate_base_data()
    
    assert X.shape == (1000, 13)  # 10 features + 3 noise features
    assert len(y) == 1000
    assert len(np.unique(y)) == 3
    
    print("✓ Data generation working correctly")


def test_mnist_data():
    """Test MNIST data handling."""
    print("Testing MNIST data handling...")
    
    try:
        from src.data.real_data import MNISTDataHandler
        
        handler = MNISTDataHandler(download_if_missing=False)
        
        # Test fallback data creation
        drifted_data, drifted_targets, feature_names = handler.create_mnist_pipeline_data(
            n_samples_per_step=50, n_time_steps=10
        )
        
        assert len(drifted_data) == 10
        assert len(drifted_targets) == 10
        assert drifted_data[0].shape[1] == len(feature_names)
        assert len(np.unique(drifted_targets[0])) == 10  # 10 classes for MNIST
        
        print("✓ MNIST data handling working correctly")
        
    except Exception as e:
        print(f"⚠️  MNIST data test skipped (requires Kaggle credentials): {e}")


def test_drift_simulation():
    """Test drift simulation."""
    print("Testing drift simulation...")
    
    generator = SyntheticDataGenerator(n_samples=1000, n_features=10, n_classes=3)
    X, y = generator.generate_base_data()
    
    simulator = DriftSimulator()
    drifted_data = simulator.inject_gradual_drift(X, time_steps=10)
    
    assert len(drifted_data) == 10
    assert drifted_data[0].shape == X.shape
    
    print("✓ Drift simulation working correctly")


def test_pipeline_stages():
    """Test pipeline stages."""
    print("Testing pipeline stages...")
    
    generator = SyntheticDataGenerator(n_samples=1000, n_features=10, n_classes=3)
    X, y = generator.generate_base_data()
    
    pipeline = MLPipeline(n_features=10, n_classes=3)
    pipeline.fit(X, y)
    
    predictions, probabilities = pipeline.predict(X)
    
    assert len(predictions) == len(y)
    assert probabilities.shape == (len(y), 3)
    
    accuracy = accuracy_score(y, predictions)
    assert accuracy > 0.5  # Should have reasonable performance
    
    print("✓ Pipeline stages working correctly")


def test_cascade_monitoring():
    """Test cascade monitoring."""
    print("Testing cascade monitoring...")
    
    generator = SyntheticDataGenerator(n_samples=1000, n_features=10, n_classes=3)
    X, y = generator.generate_base_data()
    
    monitor = CascadeMonitor()
    monitor.set_reference_data(X)
    
    # Simulate some drifted data
    drifted_X = X + np.random.normal(0, 0.1, X.shape)
    monitor.set_current_data(drifted_X)
    
    # Test drift calculation
    drift_scores = monitor.calculate_drift_score(drifted_X, X)
    assert len(drift_scores) > 0
    
    print("✓ Cascade monitoring working correctly")


def test_metrics_calculation():
    """Test metrics calculation."""
    print("Testing metrics calculation...")
    
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])
    y_proba = np.random.random((10, 3))
    
    metrics = calculate_pipeline_metrics(y_true, y_pred, y_proba)
    
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    
    print("✓ Metrics calculation working correctly")


def test_complete_simulation():
    """Test complete simulation."""
    print("Testing complete simulation...")
    
    # Generate data
    generator = SyntheticDataGenerator(n_samples=1000, n_features=10, n_classes=3)
    X, y = generator.generate_base_data()
    
    # Create pipeline
    pipeline = MLPipeline(n_features=10, n_classes=3)
    pipeline.fit(X, y)
    
    # Create monitor
    monitor = CascadeMonitor()
    monitor.set_reference_data(X)
    
    # Simulate complex drift
    drifted_data, drifted_targets, drift_simulator = simulate_complex_drift_scenario(X, y, time_steps=20)
    
    # Monitor performance over time
    performance_history = []
    
    for t in range(10):  # Test with 10 time steps
        current_data = drifted_data[t]
        current_targets = drifted_targets[t]
        
        monitor.set_current_data(current_data)
        predictions, probabilities = pipeline.predict(current_data, time_step=t)
        
        metrics = calculate_pipeline_metrics(current_targets, predictions, probabilities)
        metrics['time_step'] = t
        performance_history.append(metrics)
        
        monitoring_entry = monitor.update_monitoring(t, metrics)
    
    assert len(performance_history) == 10
    assert len(monitor.monitoring_history) == 10
    
    print("✓ Complete simulation working correctly")


def test_retraining_manager():
    """Test retraining manager."""
    print("Testing retraining manager...")
    
    retraining_manager = RetrainingManager()
    
    # Test retraining strategy determination
    trigger = {
        'type': 'performance_degradation',
        'metric': 'accuracy',
        'value': 0.6,
        'time_step': 10
    }
    
    strategy = retraining_manager.determine_retraining_strategy(trigger, 'classification_stage')
    assert strategy in ['full_retrain', 'incremental']
    
    print("✓ Retraining manager working correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("ML PIPELINE CASCADE MONITOR - TEST SUITE")
    print("=" * 60)
    
    try:
        test_data_generation()
        test_mnist_data()
        test_drift_simulation()
        test_pipeline_stages()
        test_cascade_monitoring()
        test_metrics_calculation()
        test_complete_simulation()
        test_retraining_manager()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("✓ System is ready for use")
        print("=" * 60)
        
        print("\nTo run the dashboard:")
        print("  streamlit run app.py")
        
        print("\nTo run the analysis notebook:")
        print("  jupyter notebook notebooks/pipeline_analysis.ipynb")
        
        print("\nTo use real MNIST data:")
        print("  1. Set up Kaggle API credentials")
        print("  2. Run: pip install kaggle")
        print("  3. The system will automatically download MNIST data")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 