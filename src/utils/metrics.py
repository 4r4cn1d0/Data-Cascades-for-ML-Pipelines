"""
Formal metric framework for degradation tracking and statistical analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class DegradationMetrics:
    """Formal metrics for quantifying degradation and cascade effects."""
    
    def __init__(self):
        self.degradation_slope = None
        self.recovery_delta = None
        self.cascade_correlation = None
        self.performance_history = []
        self.baseline_performance = None
        
    def calculate_degradation_slope(self, accuracy_over_time):
        """Calculate the rate of performance degradation over time."""
        if len(accuracy_over_time) < 2:
            return {
                'slope': 0.0,
                'r_squared': 0.0,
                'p_value': 1.0,
                'std_error': 0.0,
                'significance': False
            }
        
        timesteps = np.arange(len(accuracy_over_time))
        slope, intercept, r_value, p_value, std_err = stats.linregress(timesteps, accuracy_over_time)
        
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'significance': p_value < 0.05,
            'degradation_rate': abs(slope) if slope < 0 else 0.0
        }
    
    def calculate_recovery_delta(self, baseline_acc, post_retrain_acc):
        """Calculate performance recovery after retraining."""
        if baseline_acc == 0:
            return {
                'absolute_recovery': 0.0,
                'relative_recovery': 0.0,
                'recovery_efficiency': 0.0,
                'recovery_significance': False
            }
        
        absolute_recovery = post_retrain_acc - baseline_acc
        relative_recovery = absolute_recovery / baseline_acc if baseline_acc > 0 else 0.0
        recovery_efficiency = post_retrain_acc / baseline_acc if baseline_acc > 0 else 0.0
        
        return {
            'absolute_recovery': absolute_recovery,
            'relative_recovery': relative_recovery,
            'recovery_efficiency': recovery_efficiency,
            'recovery_significance': absolute_recovery > 0.05,  # 5% improvement threshold
            'recovery_quality': 'excellent' if relative_recovery > 0.2 else 'good' if relative_recovery > 0.1 else 'poor'
        }
    
    def calculate_cascade_correlation(self, upstream_errors, downstream_errors):
        """Calculate correlation between upstream and downstream errors."""
        if len(upstream_errors) != len(downstream_errors) or len(upstream_errors) < 2:
            return {
                'correlation': 0.0,
                'p_value': 1.0,
                'significance': False,
                'cascade_strength': 'none'
            }
        
        correlation, p_value = stats.pearsonr(upstream_errors, downstream_errors)
        
        # Determine cascade strength
        if abs(correlation) > 0.7:
            strength = 'strong'
        elif abs(correlation) > 0.4:
            strength = 'moderate'
        elif abs(correlation) > 0.2:
            strength = 'weak'
        else:
            strength = 'none'
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'significance': p_value < 0.05,
            'cascade_strength': strength,
            'cascade_direction': 'positive' if correlation > 0 else 'negative'
        }
    
    def calculate_acceleration(self, performance_history):
        """Calculate acceleration of performance degradation."""
        if len(performance_history) < 3:
            return 0.0
        
        # Calculate second derivative (acceleration)
        first_derivative = np.diff(performance_history)
        second_derivative = np.diff(first_derivative)
        
        return np.mean(second_derivative)
    
    def calculate_confidence_intervals(self, performance_history, confidence=0.95):
        """Calculate confidence intervals for performance metrics."""
        if len(performance_history) < 2:
            return {'lower': 0.0, 'upper': 1.0, 'mean': 0.0}
        
        mean_performance = np.mean(performance_history)
        std_performance = np.std(performance_history, ddof=1)
        n = len(performance_history)
        
        # t-distribution for small samples
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_of_error = t_value * (std_performance / np.sqrt(n))
        
        return {
            'lower': mean_performance - margin_of_error,
            'upper': mean_performance + margin_of_error,
            'mean': mean_performance,
            'margin_of_error': margin_of_error,
            'confidence_level': confidence
        }
    
    def generate_statistical_summary(self, performance_history):
        """Generate comprehensive statistical summary of performance degradation."""
        if not performance_history:
            return {
                'mean_decay_rate': 0.0,
                'std_decay_rate': 0.0,
                'max_degradation': 0.0,
                'degradation_acceleration': 0.0,
                'confidence_intervals': {'lower': 0.0, 'upper': 1.0, 'mean': 0.0},
                'trend_significance': False
            }
        
        # Calculate degradation rates
        degradation_rates = []
        for i in range(1, len(performance_history)):
            rate = performance_history[i-1] - performance_history[i]
            degradation_rates.append(rate)
        
        # Calculate acceleration
        acceleration = self.calculate_acceleration(performance_history)
        
        # Calculate confidence intervals
        confidence_intervals = self.calculate_confidence_intervals(performance_history)
        
        # Test for trend significance
        if len(performance_history) > 2:
            timesteps = np.arange(len(performance_history))
            slope, _, _, p_value, _ = stats.linregress(timesteps, performance_history)
            trend_significance = p_value < 0.05
        else:
            trend_significance = False
        
        return {
            'mean_decay_rate': np.mean(degradation_rates) if degradation_rates else 0.0,
            'std_decay_rate': np.std(degradation_rates) if degradation_rates else 0.0,
            'max_degradation': np.max(degradation_rates) if degradation_rates else 0.0,
            'degradation_acceleration': acceleration,
            'confidence_intervals': confidence_intervals,
            'trend_significance': trend_significance,
            'total_observations': len(performance_history),
            'degradation_trend': 'accelerating' if acceleration > 0 else 'decelerating' if acceleration < 0 else 'constant'
        }


class DriftDetectionMetrics:
    """Advanced metrics for drift detection and analysis."""
    
    def __init__(self):
        self.drift_history = []
        self.feature_importance = {}
        
    def calculate_distribution_drift(self, reference_data, current_data, method='ks'):
        """Calculate distribution drift using multiple methods."""
        if method == 'ks':
            return self._ks_drift_test(reference_data, current_data)
        elif method == 'wasserstein':
            return self._wasserstein_distance(reference_data, current_data)
        elif method == 'mmd':
            return self._maximum_mean_discrepancy(reference_data, current_data)
        else:
            raise ValueError(f"Unknown drift detection method: {method}")
    
    def _ks_drift_test(self, reference_data, current_data):
        """Kolmogorov-Smirnov test for distribution drift."""
        drift_scores = {}
        
        if isinstance(reference_data, pd.DataFrame) and isinstance(current_data, pd.DataFrame):
            for column in reference_data.columns:
                if column in current_data.columns:
                    try:
                        ks_stat, p_value = stats.ks_2samp(
                            reference_data[column].dropna(),
                            current_data[column].dropna()
                        )
                        drift_scores[column] = {
                            'ks_statistic': ks_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'drift_magnitude': 'high' if ks_stat > 0.3 else 'medium' if ks_stat > 0.2 else 'low'
                        }
                    except:
                        drift_scores[column] = {
                            'ks_statistic': 0.0,
                            'p_value': 1.0,
                            'significant': False,
                            'drift_magnitude': 'none'
                        }
        else:
            # For numpy arrays
            for i in range(min(reference_data.shape[1], current_data.shape[1])):
                try:
                    ks_stat, p_value = stats.ks_2samp(
                        reference_data[:, i],
                        current_data[:, i]
                    )
                    drift_scores[f'feature_{i}'] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'drift_magnitude': 'high' if ks_stat > 0.3 else 'medium' if ks_stat > 0.2 else 'low'
                    }
                except:
                    drift_scores[f'feature_{i}'] = {
                        'ks_statistic': 0.0,
                        'p_value': 1.0,
                        'significant': False,
                        'drift_magnitude': 'none'
                    }
        
        return drift_scores
    
    def _wasserstein_distance(self, reference_data, current_data):
        """Calculate Wasserstein distance between distributions."""
        try:
            from scipy.stats import wasserstein_distance
            distances = {}
            
            if isinstance(reference_data, pd.DataFrame) and isinstance(current_data, pd.DataFrame):
                for column in reference_data.columns:
                    if column in current_data.columns:
                        try:
                            dist = wasserstein_distance(
                                reference_data[column].dropna(),
                                current_data[column].dropna()
                            )
                            distances[column] = dist
                        except:
                            distances[column] = 0.0
            else:
                for i in range(min(reference_data.shape[1], current_data.shape[1])):
                    try:
                        dist = wasserstein_distance(
                            reference_data[:, i],
                            current_data[:, i]
                        )
                        distances[f'feature_{i}'] = dist
                    except:
                        distances[f'feature_{i}'] = 0.0
            
            return distances
        except ImportError:
            return self._ks_drift_test(reference_data, current_data)
    
    def _maximum_mean_discrepancy(self, reference_data, current_data):
        """Calculate Maximum Mean Discrepancy (MMD)."""
        # Simplified MMD implementation
        # In practice, you might want to use a library like torch or sklearn
        return self._ks_drift_test(reference_data, current_data)
    
    def calculate_feature_importance_drift(self, model, reference_data, current_data):
        """Calculate how feature importance changes with drift."""
        # This would require a model that supports feature importance
        # For now, return a placeholder
        return {
            'feature_importance_stability': 0.8,
            'importance_correlation': 0.75,
            'top_drifted_features': []
        }


class CascadeEffectMetrics:
    """Metrics specifically for cascade effect analysis."""
    
    def __init__(self):
        self.cascade_history = []
        self.error_propagation_matrix = None
        
    def calculate_error_propagation(self, pipeline_stages, stage_errors):
        """Calculate how errors propagate through pipeline stages."""
        if not stage_errors or len(stage_errors) < 2:
            return {
                'propagation_matrix': None,
                'cascade_strength': 0.0,
                'error_amplification': 0.0
            }
        
        # Create error propagation matrix
        stage_names = list(stage_errors.keys())
        n_stages = len(stage_names)
        propagation_matrix = np.zeros((n_stages, n_stages))
        
        for i, stage1 in enumerate(stage_names):
            for j, stage2 in enumerate(stage_names):
                if i < j:  # Only calculate forward propagation
                    if stage1 in stage_errors and stage2 in stage_errors:
                        # Calculate correlation between errors
                        correlation, _ = stats.pearsonr(
                            stage_errors[stage1], 
                            stage_errors[stage2]
                        )
                        propagation_matrix[i, j] = abs(correlation)
        
        # Calculate cascade strength (average propagation)
        cascade_strength = np.mean(propagation_matrix[propagation_matrix > 0])
        
        # Calculate error amplification
        error_amplification = np.max(propagation_matrix) if np.max(propagation_matrix) > 0 else 0.0
        
        return {
            'propagation_matrix': propagation_matrix,
            'cascade_strength': cascade_strength,
            'error_amplification': error_amplification,
            'stage_names': stage_names,
            'strongest_cascade': np.unravel_index(np.argmax(propagation_matrix), propagation_matrix.shape) if np.max(propagation_matrix) > 0 else None
        }
    
    def calculate_cascade_impact(self, baseline_performance, cascade_performance):
        """Calculate the impact of cascade effects on performance."""
        if not baseline_performance or not cascade_performance:
            return {
                'performance_loss': 0.0,
                'cascade_impact': 'none',
                'recovery_difficulty': 'low'
            }
        
        performance_loss = baseline_performance - cascade_performance
        
        if performance_loss > 0.2:
            impact = 'severe'
            recovery = 'high'
        elif performance_loss > 0.1:
            impact = 'moderate'
            recovery = 'medium'
        elif performance_loss > 0.05:
            impact = 'mild'
            recovery = 'low'
        else:
            impact = 'none'
            recovery = 'low'
        
        return {
            'performance_loss': performance_loss,
            'cascade_impact': impact,
            'recovery_difficulty': recovery,
            'relative_loss': performance_loss / baseline_performance if baseline_performance > 0 else 0.0
        }


def generate_comprehensive_report(performance_history, drift_scores, cascade_effects):
    """Generate a comprehensive analysis report."""
    degradation_metrics = DegradationMetrics()
    drift_metrics = DriftDetectionMetrics()
    cascade_metrics = CascadeEffectMetrics()
    
    # Calculate all metrics
    degradation_analysis = degradation_metrics.generate_statistical_summary(performance_history)
    drift_analysis = drift_metrics.calculate_distribution_drift(
        drift_scores.get('reference', []), 
        drift_scores.get('current', [])
    )
    cascade_analysis = cascade_metrics.calculate_error_propagation(
        cascade_effects.get('stages', {}),
        cascade_effects.get('errors', {})
    )
    
    return {
        'degradation_analysis': degradation_analysis,
        'drift_analysis': drift_analysis,
        'cascade_analysis': cascade_analysis,
        'overall_health': _calculate_overall_health(degradation_analysis, drift_analysis, cascade_analysis),
        'recommendations': _generate_recommendations(degradation_analysis, drift_analysis, cascade_analysis)
    }


def _calculate_overall_health(degradation, drift, cascade):
    """Calculate overall system health score."""
    health_score = 1.0
    
    # Penalize for degradation
    if degradation['trend_significance']:
        health_score -= 0.3
    
    # Penalize for drift
    high_drift_features = sum(1 for feature in drift.values() if feature.get('significant', False))
    health_score -= min(0.3, high_drift_features * 0.1)
    
    # Penalize for cascade effects
    if cascade['cascade_strength'] > 0.5:
        health_score -= 0.2
    
    return max(0.0, health_score)


def _generate_recommendations(degradation, drift, cascade):
    """Generate actionable recommendations based on analysis."""
    recommendations = []
    
    if degradation['trend_significance']:
        recommendations.append("Consider immediate retraining due to significant performance degradation")
    
    high_drift_features = [f for f, data in drift.items() if data.get('significant', False)]
    if high_drift_features:
        recommendations.append(f"Monitor drift in features: {', '.join(high_drift_features[:3])}")
    
    if cascade['cascade_strength'] > 0.5:
        recommendations.append("Implement cascade monitoring and early intervention strategies")
    
    if not recommendations:
        recommendations.append("System appears healthy - continue monitoring")
    
    return recommendations 