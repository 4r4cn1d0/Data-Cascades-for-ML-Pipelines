#!/usr/bin/env python3
"""
Test script to verify the actual metrics implementation
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('.')

def test_degradation_metrics():
    """Test degradation metrics with realistic data"""
    print("=== Testing Degradation Metrics ===")
    
    from src.utils.metrics import DegradationMetrics
    
    # Create realistic accuracy degradation
    np.random.seed(42)
    base_accuracy = 0.95
    degradation_rate = 0.002  # 0.2% per step
    steps = 50
    
    # Generate realistic accuracy with some noise
    accuracies = []
    current_acc = base_accuracy
    for i in range(steps):
        # Add some random variation
        noise = np.random.normal(0, 0.01)
        current_acc = max(0.5, current_acc - degradation_rate + noise)
        accuracies.append(current_acc)
    
    accuracies = np.array(accuracies)
    print(f"Accuracy range: {accuracies.min():.4f} to {accuracies.max():.4f}")
    print(f"Final accuracy: {accuracies[-1]:.4f}")
    
    # Test degradation calculation
    dm = DegradationMetrics()
    result = dm.calculate_degradation_slope(accuracies)
    
    print(f"Degradation slope: {result['slope']:.6f}")
    print(f"R-squared: {result['r_squared']:.4f}")
    print(f"P-value: {result['p_value']:.6f}")
    print(f"Significant: {result['significance']}")
    
    return result

def test_drift_detection():
    """Test drift detection with realistic data"""
    print("\n=== Testing Drift Detection ===")
    
    from src.utils.metrics import DriftDetectionMetrics
    from src.data.synthetic_data import SyntheticDataGenerator
    
    # Generate base and drifted data
    data_gen = SyntheticDataGenerator(n_samples=1000)
    X_base, y_base = data_gen.generate_base_data()
    
    # Create drifted data by modifying the generator
    drift_gen = SyntheticDataGenerator(n_samples=1000, random_state=123)
    X_drift, y_drift = drift_gen.generate_base_data()
    
    # Apply artificial drift
    X_drift = X_drift + np.random.normal(0.5, 0.2, X_drift.shape)
    
    print(f"Base data shape: {X_base.shape}")
    print(f"Drift data shape: {X_drift.shape}")
    
    # Test drift detection
    dd = DriftDetectionMetrics()
    drift_result = dd.calculate_distribution_drift(X_base, X_drift)
    
    # Count significant features
    significant_features = [f for f, data in drift_result.items() if data['significant']]
    print(f"Features with significant drift: {len(significant_features)}/{len(drift_result)}")
    
    # Calculate average drift score
    drift_scores = [data['ks_statistic'] for data in drift_result.values()]
    avg_drift = np.mean(drift_scores)
    max_drift = np.max(drift_scores)
    
    print(f"Average drift score: {avg_drift:.4f}")
    print(f"Max drift score: {max_drift:.4f}")
    
    return {
        'significant_features': len(significant_features),
        'total_features': len(drift_result),
        'avg_drift_score': avg_drift,
        'max_drift_score': max_drift
    }

def test_cascade_effects():
    """Test cascade effect metrics"""
    print("\n=== Testing Cascade Effects ===")
    
    from src.utils.metrics import CascadeEffectMetrics
    
    # Simulate realistic cascade effects
    np.random.seed(42)
    steps = 50
    
    # Generate realistic error propagation
    upstream_errors = np.random.beta(2, 5, steps)  # Low initial errors
    downstream_errors = upstream_errors * 1.5 + np.random.normal(0, 0.05, steps)  # Amplified errors
    
    print(f"Upstream error range: {upstream_errors.min():.4f} to {upstream_errors.max():.4f}")
    print(f"Downstream error range: {downstream_errors.min():.4f} to {downstream_errors.max():.4f}")
    
    # Test cascade correlation using the correct method
    ce = CascadeEffectMetrics()
    
    # Create stage errors dictionary
    stage_errors = {
        'upstream': upstream_errors,
        'downstream': downstream_errors
    }
    pipeline_stages = ['upstream', 'downstream']
    
    result = ce.calculate_error_propagation(pipeline_stages, stage_errors)
    
    print(f"Cascade strength: {result['cascade_strength']:.4f}")
    print(f"Error amplification: {result['error_amplification']:.4f}")
    print(f"Strongest cascade: {result['strongest_cascade']}")
    
    return result

def test_pipeline_performance():
    """Test the actual pipeline performance"""
    print("\n=== Testing Pipeline Performance ===")
    
    from src.pipeline.stages import ProductionMLPipeline
    from src.data.synthetic_data import SyntheticDataGenerator
    
    # Generate data
    data_gen = SyntheticDataGenerator(n_samples=1000)
    X_train, y_train = data_gen.generate_base_data()
    
    # Generate test data
    test_gen = SyntheticDataGenerator(n_samples=200, random_state=456)
    X_test, y_test = test_gen.generate_base_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training labels range: {y_train.min()} to {y_train.max()}")
    print(f"Test labels range: {y_test.min()} to {y_test.max()}")
    
    # Create and train pipeline
    pipeline = ProductionMLPipeline()
    
    try:
        # Train pipeline
        pipeline.train_pipeline(X_train, y_train)
        print("Pipeline training completed successfully")
        
        # Test predictions
        predictions_output = pipeline.predict(X_test)
        print(f"Predictions output type: {type(predictions_output)}")
        
        # Extract predictions from the output dictionary
        if isinstance(predictions_output, dict):
            predictions = predictions_output['predictions']
            print(f"Extracted predictions shape: {predictions.shape}")
            print(f"Predictions range: {predictions.min()} to {predictions.max()}")
        else:
            predictions = predictions_output
            print(f"Predictions shape: {predictions.shape}")
            print(f"Predictions range: {predictions.min()} to {predictions.max()}")
        
        print(f"Test labels shape: {y_test.shape}")
        print(f"Test labels range: {y_test.min()} to {y_test.max()}")
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y_test)
        print(f"Pipeline accuracy: {accuracy:.4f}")
        
        # Show some sample predictions vs actual
        print("Sample predictions vs actual:")
        for i in range(min(5, len(predictions))):
            actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            print(f"  Predicted: {predictions[i]}, Actual: {actual}")
        
        return accuracy
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all tests"""
    print("🔍 Testing Real Metrics Implementation")
    print("=" * 50)
    
    results = {}
    
    # Test degradation metrics
    results['degradation'] = test_degradation_metrics()
    
    # Test drift detection
    results['drift'] = test_drift_detection()
    
    # Test cascade effects
    results['cascade'] = test_cascade_effects()
    
    # Test pipeline performance
    results['pipeline'] = test_pipeline_performance()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY OF REAL METRICS")
    print("=" * 50)
    
    print(f"Degradation slope: {results['degradation']['slope']:.6f}")
    print(f"Drift detection: {results['drift']['significant_features']}/{results['drift']['total_features']} features")
    print(f"Average drift score: {results['drift']['avg_drift_score']:.4f}")
    print(f"Cascade strength: {results['cascade']['cascade_strength']:.4f}")
    print(f"Error amplification: {results['cascade']['error_amplification']:.4f}")
    print(f"Pipeline accuracy: {results['pipeline']:.4f}" if results['pipeline'] else "Pipeline: ERROR")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main() 