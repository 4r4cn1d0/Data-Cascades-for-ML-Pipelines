"""
Cascade monitoring and retraining management for ML pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')


class CascadeMonitor:
    """Monitors cascade effects in ML pipelines."""
    
    def __init__(self):
        self.reference_data = None
        self.current_data = None
        self.monitoring_history = []
        self.drift_scores_history = []
        self.cascade_history = []
        self.retraining_triggers = []
        self.performance_history = []
        
    def set_reference_data(self, data):
        """Set reference data for drift detection."""
        self.reference_data = data
        if isinstance(data, pd.DataFrame):
            self.reference_stats = {
                'mean': data.mean(),
                'std': data.std(),
                'distribution': data.describe()
            }
        else:
            self.reference_stats = {
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0)
            }
    
    def set_current_data(self, data):
        """Set current data for comparison."""
        self.current_data = data
    
    def calculate_drift_score(self, current_data, reference_data=None):
        """Calculate drift score between current and reference data."""
        if reference_data is None:
            reference_data = self.reference_data
        
        if reference_data is None:
            return {}
        
        drift_scores = {}
        
        if isinstance(current_data, pd.DataFrame) and isinstance(reference_data, pd.DataFrame):
            # For DataFrame data
            for column in current_data.columns:
                if column in reference_data.columns:
                    # KS test for distribution drift
                    try:
                        ks_stat, p_value = ks_2samp(
                            reference_data[column].dropna(),
                            current_data[column].dropna()
                        )
                        drift_scores[column] = ks_stat
                    except:
                        drift_scores[column] = 0.0
        else:
            # For numpy array data
            try:
                for i in range(min(current_data.shape[1], reference_data.shape[1])):
                    ks_stat, p_value = ks_2samp(
                        reference_data[:, i],
                        current_data[:, i]
                    )
                    drift_scores[f'feature_{i}'] = ks_stat
            except:
                drift_scores['overall'] = 0.0
        
        return drift_scores
    
    def detect_cascade_effects(self, predictions, probabilities, time_step):
        """Detect cascade effects in pipeline predictions."""
        if len(self.performance_history) < 2:
            return {
                'avg_cascade_score': 0.0,
                'max_cascade_score': 0.0,
                'cascade_correlations': {}
            }
        
        # Calculate cascade correlations
        cascade_correlations = {}
        
        # Performance degradation correlation
        if len(self.performance_history) > 1:
            prev_metrics = self.performance_history[-2]
            curr_metrics = self.performance_history[-1]
            
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                if metric in prev_metrics and metric in curr_metrics:
                    degradation = prev_metrics[metric] - curr_metrics[metric]
                    cascade_correlations[f'{metric}_degradation'] = degradation
        
        # Drift correlation
        if self.drift_scores_history:
            avg_drift = np.mean(list(self.drift_scores_history[-1].values()))
            cascade_correlations['drift_correlation'] = avg_drift
        
        # Calculate overall cascade score
        if cascade_correlations:
            avg_cascade_score = np.mean(list(cascade_correlations.values()))
            max_cascade_score = np.max(list(cascade_correlations.values()))
        else:
            avg_cascade_score = 0.0
            max_cascade_score = 0.0
        
        return {
            'avg_cascade_score': avg_cascade_score,
            'max_cascade_score': max_cascade_score,
            'cascade_correlations': cascade_correlations,
            'time_step': time_step
        }
    
    def check_retraining_triggers(self, metrics, time_step):
        """Check if retraining is needed."""
        triggers = []
        
        # Performance degradation trigger
        if len(self.performance_history) > 1:
            prev_accuracy = self.performance_history[-2].get('accuracy', 1.0)
            curr_accuracy = metrics.get('accuracy', 0.0)
            
            if prev_accuracy - curr_accuracy > 0.1:  # 10% degradation
                triggers.append({
                    'type': 'performance_degradation',
                    'metric': 'accuracy',
                    'value': curr_accuracy,
                    'threshold': 0.1,
                    'time_step': time_step
                })
        
        # Drift trigger
        if self.drift_scores_history:
            avg_drift = np.mean(list(self.drift_scores_history[-1].values()))
            if avg_drift > 0.3:  # High drift threshold
                triggers.append({
                    'type': 'high_drift',
                    'metric': 'drift_score',
                    'value': avg_drift,
                    'threshold': 0.3,
                    'time_step': time_step
                })
        
        # Cascade effect trigger
        if len(self.cascade_history) > 0:
            cascade_score = self.cascade_history[-1].get('avg_cascade_score', 0.0)
            if cascade_score > 0.5:  # High cascade effect
                triggers.append({
                    'type': 'cascade_effect',
                    'metric': 'cascade_score',
                    'value': cascade_score,
                    'threshold': 0.5,
                    'time_step': time_step
                })
        
        return triggers
    
    def update_monitoring(self, time_step, metrics):
        """Update monitoring with new metrics."""
        # Store performance metrics
        metrics['time_step'] = time_step
        self.performance_history.append(metrics)
        
        # Calculate drift scores
        if self.current_data is not None and self.reference_data is not None:
            drift_scores = self.calculate_drift_score(self.current_data, self.reference_data)
            self.drift_scores_history.append(drift_scores)
        else:
            drift_scores = {}
        
        # Detect cascade effects
        cascade_analysis = self.detect_cascade_effects(None, None, time_step)
        self.cascade_history.append(cascade_analysis)
        
        # Check retraining triggers
        triggers = self.check_retraining_triggers(metrics, time_step)
        self.retraining_triggers.extend(triggers)
        
        # Create monitoring entry
        monitoring_entry = {
            'time_step': time_step,
            'metrics': metrics,
            'drift_scores': drift_scores,
            'cascade_analysis': cascade_analysis,
            'retraining_triggers': triggers
        }
        
        self.monitoring_history.append(monitoring_entry)
        
        return monitoring_entry
    
    def get_monitoring_summary(self):
        """Get summary of monitoring results."""
        if not self.monitoring_history:
            return "No monitoring data available."
        
        # Performance trends
        accuracies = [entry['metrics'].get('accuracy', 0) for entry in self.monitoring_history]
        f1_scores = [entry['metrics'].get('f1', 0) for entry in self.monitoring_history]
        
        performance_trends = {
            'current_accuracy': accuracies[-1] if accuracies else 0,
            'current_f1': f1_scores[-1] if f1_scores else 0,
            'accuracy_trend': accuracies[0] - accuracies[-1] if len(accuracies) > 1 else 0,
            'f1_trend': f1_scores[0] - f1_scores[-1] if len(f1_scores) > 1 else 0
        }
        
        # Drift analysis
        drift_analysis = {
            'total_drift_checks': len(self.drift_scores_history),
            'avg_drift_score': np.mean([np.mean(list(scores.values())) for scores in self.drift_scores_history]) if self.drift_scores_history else 0,
            'max_drift_score': np.max([np.max(list(scores.values())) for scores in self.drift_scores_history]) if self.drift_scores_history else 0
        }
        
        # Cascade analysis
        cascade_analysis = {
            'total_cascade_checks': len(self.cascade_history),
            'avg_cascade_score': np.mean([c.get('avg_cascade_score', 0) for c in self.cascade_history]) if self.cascade_history else 0,
            'max_cascade_score': np.max([c.get('max_cascade_score', 0) for c in self.cascade_history]) if self.cascade_history else 0
        }
        
        # Retraining analysis
        retraining_analysis = {
            'total_triggers': len(self.retraining_triggers),
            'trigger_types': list(set([t['type'] for t in self.retraining_triggers])),
            'recent_triggers': self.retraining_triggers[-5:] if self.retraining_triggers else []
        }
        
        return {
            'performance_trends': performance_trends,
            'drift_analysis': drift_analysis,
            'cascade_analysis': cascade_analysis,
            'retraining_analysis': retraining_analysis,
            'total_time_steps': len(self.monitoring_history)
        }
    
    def get_drift_summary(self):
        """Get summary of drift detection results."""
        if not self.drift_scores_history:
            return "No drift data available."
        
        # Calculate drift statistics
        all_drift_scores = []
        for scores in self.drift_scores_history:
            if scores:
                all_drift_scores.extend(list(scores.values()))
        
        if not all_drift_scores:
            return "No drift scores available."
        
        drift_summary = {
            'total_drift_checks': len(self.drift_scores_history),
            'avg_drift_score': np.mean(all_drift_scores),
            'max_drift_score': np.max(all_drift_scores),
            'min_drift_score': np.min(all_drift_scores),
            'drift_trend': 'increasing' if len(all_drift_scores) > 1 and all_drift_scores[-1] > all_drift_scores[0] else 'stable',
            'high_drift_features': []
        }
        
        # Find features with high drift
        if self.drift_scores_history:
            latest_scores = self.drift_scores_history[-1]
            for feature, score in latest_scores.items():
                if score > 0.3:  # High drift threshold
                    drift_summary['high_drift_features'].append({
                        'feature': feature,
                        'drift_score': score
                    })
        
        return drift_summary
    
    def get_cascade_summary(self):
        """Get summary of cascade effects."""
        if not self.cascade_history:
            return "No cascade effects recorded."
        
        cascade_scores = [c.get('avg_cascade_score', 0) for c in self.cascade_history]
        
        return {
            'total_cascade_checks': len(self.cascade_history),
            'avg_cascade_score': np.mean(cascade_scores),
            'max_cascade_score': np.max(cascade_scores),
            'cascade_trend': 'increasing' if len(cascade_scores) > 1 and cascade_scores[-1] > cascade_scores[0] else 'stable',
            'high_cascade_periods': len([s for s in cascade_scores if s > 0.5])
        }
    
    def run_simulation(self, n_steps=50):
        """Run a complete simulation."""
        print(f"Running cascade monitoring simulation for {n_steps} steps...")
        
        for step in range(n_steps):
            # Simulate some metrics
            metrics = {
                'accuracy': max(0.1, 0.9 - step * 0.02),
                'f1': max(0.1, 0.85 - step * 0.02),
                'precision': max(0.1, 0.88 - step * 0.02),
                'recall': max(0.1, 0.82 - step * 0.02)
            }
            
            self.update_monitoring(step, metrics)
        
        print("Simulation completed!")
        return self.get_monitoring_summary()


class RetrainingManager:
    """Manages retraining strategies for pipeline stages."""
    
    def __init__(self):
        self.retraining_history = []
        self.strategies = {
            'full_retrain': 'Complete retraining of the model',
            'incremental': 'Incremental learning with new data',
            'feature_adaptation': 'Adapt features to new data distribution',
            'ensemble_update': 'Update ensemble weights'
        }
    
    def determine_retraining_strategy(self, trigger, stage_name):
        """Determine the best retraining strategy based on trigger."""
        if trigger['type'] == 'performance_degradation':
            if trigger['value'] < 0.5:
                return 'full_retrain'
            else:
                return 'incremental'
        elif trigger['type'] == 'high_drift':
            return 'feature_adaptation'
        elif trigger['type'] == 'cascade_effect':
            return 'ensemble_update'
        else:
            return 'incremental'
    
    def execute_retraining(self, strategy, stage, new_data):
        """Execute retraining with the specified strategy."""
        retraining_record = {
            'strategy': strategy,
            'stage': stage,
            'timestamp': pd.Timestamp.now(),
            'data_size': len(new_data) if hasattr(new_data, '__len__') else 0
        }
        
        # Simulate retraining process
        if strategy == 'full_retrain':
            retraining_record['status'] = 'completed'
            retraining_record['message'] = 'Full retraining completed'
        elif strategy == 'incremental':
            retraining_record['status'] = 'completed'
            retraining_record['message'] = 'Incremental learning completed'
        elif strategy == 'feature_adaptation':
            retraining_record['status'] = 'completed'
            retraining_record['message'] = 'Feature adaptation completed'
        elif strategy == 'ensemble_update':
            retraining_record['status'] = 'completed'
            retraining_record['message'] = 'Ensemble weights updated'
        
        self.retraining_history.append(retraining_record)
        return retraining_record
    
    def get_retraining_summary(self):
        """Get summary of retraining activities."""
        if not self.retraining_history:
            return "No retraining activities recorded."
        
        strategy_counts = {}
        for record in self.retraining_history:
            strategy = record['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_retraining_events': len(self.retraining_history),
            'strategy_distribution': strategy_counts,
            'recent_events': self.retraining_history[-5:] if self.retraining_history else []
        } 