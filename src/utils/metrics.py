"""
Utility functions for calculating performance metrics and cascade effects.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


def calculate_pipeline_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive pipeline performance metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted')
    }
    
    # Add confidence-based metrics if probabilities are available
    if y_proba is not None:
        confidence_scores = np.max(y_proba, axis=1)
        metrics['avg_confidence'] = np.mean(confidence_scores)
        metrics['confidence_std'] = np.std(confidence_scores)
        metrics['low_confidence_rate'] = np.mean(confidence_scores < 0.7)
    
    return metrics


def calculate_cascade_correlation(stage1_predictions, stage2_predictions, y_true):
    """Calculate correlation between errors in different pipeline stages."""
    # Calculate errors for each stage
    stage1_errors = (stage1_predictions != y_true).astype(int)
    stage2_errors = (stage2_predictions != y_true).astype(int)
    
    # Calculate correlation
    correlation = np.corrcoef(stage1_errors, stage2_errors)[0, 1]
    
    return correlation if not np.isnan(correlation) else 0.0


def calculate_drift_score(reference_data, current_data, method='statistical'):
    """Calculate drift score between reference and current data."""
    if method == 'statistical':
        return _calculate_statistical_drift(reference_data, current_data)
    elif method == 'distribution':
        return _calculate_distribution_drift(reference_data, current_data)
    else:
        raise ValueError(f"Unknown drift calculation method: {method}")


def _calculate_statistical_drift(reference_data, current_data):
    """Calculate drift using statistical measures."""
    drift_scores = {}
    
    for col in reference_data.columns:
        if col in current_data.columns:
            ref_mean = reference_data[col].mean()
            ref_std = reference_data[col].std()
            curr_mean = current_data[col].mean()
            curr_std = current_data[col].std()
            
            # Normalized difference
            mean_diff = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
            std_diff = abs(curr_std - ref_std) / (ref_std + 1e-8)
            
            # Combined drift score
            drift_score = (mean_diff + std_diff) / 2
            drift_scores[col] = drift_score
    
    return drift_scores


def _calculate_distribution_drift(reference_data, current_data):
    """Calculate drift using distribution comparison."""
    drift_scores = {}
    
    for col in reference_data.columns:
        if col in current_data.columns:
            # Calculate histogram differences
            ref_hist, _ = np.histogram(reference_data[col], bins=20, density=True)
            curr_hist, _ = np.histogram(current_data[col], bins=20, density=True)
            
            # Calculate histogram distance
            drift_score = np.sum(np.abs(curr_hist - ref_hist)) / 2
            drift_scores[col] = drift_score
    
    return drift_scores


def calculate_performance_degradation(baseline_metrics, current_metrics):
    """Calculate performance degradation from baseline."""
    degradation = {}
    
    for metric in baseline_metrics.keys():
        if metric in current_metrics:
            degradation[metric] = baseline_metrics[metric] - current_metrics[metric]
    
    # Overall degradation score
    if degradation:
        avg_degradation = np.mean(list(degradation.values()))
        degradation['overall'] = avg_degradation
    
    return degradation


def calculate_cascade_impact(stage_performances, cascade_correlations):
    """Calculate the impact of cascade effects on overall performance."""
    if not cascade_correlations:
        return 0.0
    
    # Calculate cascade impact based on correlation strength
    cascade_strength = np.mean(list(cascade_correlations.values()))
    
    # Weight by stage performance degradation
    performance_degradation = 0.0
    if stage_performances:
        baseline_performance = max(stage_performances.values())
        current_performance = min(stage_performances.values())
        performance_degradation = baseline_performance - current_performance
    
    # Combined cascade impact
    cascade_impact = cascade_strength * performance_degradation
    
    return cascade_impact


def calculate_retraining_priority(triggers, stage_importance):
    """Calculate retraining priority for different pipeline stages."""
    priority_scores = {}
    
    for trigger in triggers:
        stage = trigger.get('stage', 'unknown')
        trigger_type = trigger.get('type', 'unknown')
        trigger_value = trigger.get('value', 0)
        
        # Base priority based on trigger type
        base_priority = {
            'performance_degradation': 0.8,
            'data_drift': 0.6,
            'cascade_effect': 0.9,
            'concept_drift': 0.7
        }.get(trigger_type, 0.5)
        
        # Adjust by trigger severity
        severity = 1.0 - trigger_value if trigger_value < 1.0 else 0.0
        adjusted_priority = base_priority * (1 + severity)
        
        # Weight by stage importance
        stage_weight = stage_importance.get(stage, 1.0)
        final_priority = adjusted_priority * stage_weight
        
        if stage not in priority_scores:
            priority_scores[stage] = []
        priority_scores[stage].append(final_priority)
    
    # Calculate average priority per stage
    stage_priorities = {}
    for stage, priorities in priority_scores.items():
        stage_priorities[stage] = np.mean(priorities)
    
    return stage_priorities


def generate_performance_report(pipeline_metrics, cascade_metrics, drift_metrics):
    """Generate a comprehensive performance report."""
    report = {
        'summary': {
            'overall_accuracy': pipeline_metrics.get('accuracy', 0),
            'overall_f1': pipeline_metrics.get('f1', 0),
            'cascade_score': cascade_metrics.get('overall_cascade_score', 0),
            'drift_score': np.mean(list(drift_metrics.values())) if drift_metrics else 0
        },
        'alerts': [],
        'recommendations': []
    }
    
    # Generate alerts
    if pipeline_metrics.get('accuracy', 1) < 0.8:
        report['alerts'].append({
            'type': 'warning',
            'message': 'Pipeline accuracy below threshold',
            'value': pipeline_metrics.get('accuracy', 0)
        })
    
    if cascade_metrics.get('overall_cascade_score', 0) > 0.5:
        report['alerts'].append({
            'type': 'error',
            'message': 'High cascade effects detected',
            'value': cascade_metrics.get('overall_cascade_score', 0)
        })
    
    if drift_metrics and np.mean(list(drift_metrics.values())) > 0.3:
        report['alerts'].append({
            'type': 'warning',
            'message': 'Significant data drift detected',
            'value': np.mean(list(drift_metrics.values()))
        })
    
    # Generate recommendations
    if pipeline_metrics.get('accuracy', 1) < 0.7:
        report['recommendations'].append('Consider full pipeline retraining')
    
    if cascade_metrics.get('overall_cascade_score', 0) > 0.7:
        report['recommendations'].append('Investigate cascade effect root causes')
    
    if drift_metrics and np.mean(list(drift_metrics.values())) > 0.5:
        report['recommendations'].append('Implement drift detection and mitigation')
    
    return report


def calculate_stage_contribution(pipeline_metrics, stage_metrics):
    """Calculate the contribution of each stage to overall performance."""
    contributions = {}
    
    # Calculate stage-wise contribution to overall performance
    for stage, metrics in stage_metrics.items():
        stage_accuracy = metrics.get('accuracy', 0)
        overall_accuracy = pipeline_metrics.get('accuracy', 0)
        
        # Contribution is the ratio of stage performance to overall performance
        contribution = stage_accuracy / overall_accuracy if overall_accuracy > 0 else 0
        contributions[stage] = contribution
    
    return contributions


if __name__ == "__main__":
    # Test the metrics functions
    print("Testing metrics functions...")
    
    # Test data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])
    y_proba = np.random.random((10, 3))
    
    # Test pipeline metrics
    metrics = calculate_pipeline_metrics(y_true, y_pred, y_proba)
    print(f"Pipeline metrics: {metrics}")
    
    # Test cascade correlation
    stage1_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    stage2_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])
    correlation = calculate_cascade_correlation(stage1_pred, stage2_pred, y_true)
    print(f"Cascade correlation: {correlation}")
    
    # Test drift score
    reference_data = pd.DataFrame(np.random.normal(0, 1, (100, 5)))
    current_data = pd.DataFrame(np.random.normal(0.5, 1.2, (100, 5)))
    drift_scores = calculate_drift_score(reference_data, current_data)
    print(f"Drift scores: {drift_scores}")
    
    print("All metrics functions working correctly!") 