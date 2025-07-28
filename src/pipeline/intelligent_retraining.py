"""
Intelligent retraining framework with multiple strategies and cost-benefit analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')


class IntelligentRetrainingManager:
    """Advanced retraining manager with multiple strategies and cost analysis."""
    
    def __init__(self):
        self.retraining_strategies = {
            'threshold_based': self.threshold_based_retraining,
            'scheduled': self.scheduled_retraining,
            'confidence_based': self.confidence_based_retraining,
            'cost_aware': self.cost_aware_retraining,
            'adaptive': self.adaptive_retraining,
            'ensemble': self.ensemble_retraining
        }
        self.retraining_history = []
        self.cost_benefit_analysis = {}
        self.performance_tracking = {}
        
    def threshold_based_retraining(self, current_acc, threshold=0.05, baseline_acc=None):
        """Retrain only if accuracy drops below threshold."""
        if baseline_acc is None:
            baseline_acc = 0.9  # Default baseline
        
        performance_drop = baseline_acc - current_acc
        should_retrain = performance_drop > threshold
        
        return {
            'should_retrain': should_retrain,
            'reason': f"Performance drop: {performance_drop:.3f} (threshold: {threshold})",
            'strategy': 'threshold_based',
            'performance_drop': performance_drop,
            'threshold': threshold
        }
    
    def scheduled_retraining(self, timestep, frequency=100, last_retrain_step=0):
        """Retrain every N timesteps."""
        should_retrain = (timestep - last_retrain_step) >= frequency
        
        return {
            'should_retrain': should_retrain,
            'reason': f"Schedule: {timestep} - {last_retrain_step} >= {frequency}",
            'strategy': 'scheduled',
            'time_since_last': timestep - last_retrain_step,
            'frequency': frequency
        }
    
    def confidence_based_retraining(self, model_confidence, threshold=0.8, confidence_history=None):
        """Retrain based on model confidence levels."""
        if confidence_history is None:
            confidence_history = []
        
        avg_confidence = np.mean(confidence_history) if confidence_history else model_confidence
        confidence_drop = avg_confidence - model_confidence
        
        should_retrain = model_confidence < threshold or confidence_drop > 0.1
        
        return {
            'should_retrain': should_retrain,
            'reason': f"Confidence: {model_confidence:.3f} (threshold: {threshold})",
            'strategy': 'confidence_based',
            'confidence_drop': confidence_drop,
            'avg_confidence': avg_confidence
        }
    
    def cost_aware_retraining(self, performance_gain, retraining_cost, 
                             data_size=1000, model_complexity='medium'):
        """Retrain only if benefit exceeds cost."""
        # Calculate retraining cost based on model complexity and data size
        complexity_costs = {
            'simple': 1.0,
            'medium': 2.0,
            'complex': 5.0
        }
        
        base_cost = complexity_costs.get(model_complexity, 2.0)
        scaled_cost = base_cost * (data_size / 1000)  # Scale with data size
        
        # Calculate expected benefit
        expected_benefit = performance_gain * 100  # Convert to percentage points
        
        # Cost-benefit ratio
        benefit_cost_ratio = expected_benefit / scaled_cost
        
        should_retrain = benefit_cost_ratio > 1.0
        
        return {
            'should_retrain': should_retrain,
            'reason': f"Benefit/Cost ratio: {benefit_cost_ratio:.2f}",
            'strategy': 'cost_aware',
            'expected_benefit': expected_benefit,
            'retraining_cost': scaled_cost,
            'benefit_cost_ratio': benefit_cost_ratio
        }
    
    def adaptive_retraining(self, performance_history, drift_scores, 
                           retraining_frequency=10):
        """Adaptive retraining based on performance and drift patterns."""
        if len(performance_history) < 5:
            return {
                'should_retrain': False,
                'reason': "Insufficient history for adaptive retraining",
                'strategy': 'adaptive'
            }
        
        # Calculate performance trend
        recent_performance = performance_history[-5:]
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # Calculate drift trend
        if drift_scores and len(drift_scores) >= 3:
            recent_drift = list(drift_scores.values())[-3:]
            drift_trend = np.polyfit(range(len(recent_drift)), recent_drift, 1)[0]
        else:
            drift_trend = 0.0
        
        # Adaptive decision logic
        should_retrain = False
        reason = "No retraining needed"
        
        if performance_trend < -0.01:  # Declining performance
            should_retrain = True
            reason = f"Declining performance trend: {performance_trend:.3f}"
        elif drift_trend > 0.05:  # Increasing drift
            should_retrain = True
            reason = f"Increasing drift trend: {drift_trend:.3f}"
        elif len(performance_history) % retraining_frequency == 0:  # Periodic
            should_retrain = True
            reason = f"Periodic retraining (every {retraining_frequency} steps)"
        
        return {
            'should_retrain': should_retrain,
            'reason': reason,
            'strategy': 'adaptive',
            'performance_trend': performance_trend,
            'drift_trend': drift_trend
        }
    
    def ensemble_retraining(self, model_performances, ensemble_weights=None):
        """Retrain based on ensemble model performance."""
        if not model_performances:
            return {
                'should_retrain': False,
                'reason': "No model performances provided",
                'strategy': 'ensemble',
                'ensemble_performance': 0.0,
                'models_to_retrain': []
            }
        
        if ensemble_weights is None:
            ensemble_weights = [1.0] * len(model_performances)
        
        # Extract performance values
        performances = list(model_performances.values())
        model_names = list(model_performances.keys())
        
        # Calculate weighted ensemble performance
        weighted_performance = np.average(performances, weights=ensemble_weights[:len(performances)])
        
        # Determine which models need retraining
        models_to_retrain = []
        for model_name, performance in model_performances.items():
            if performance < weighted_performance * 0.9:  # 10% below ensemble average
                models_to_retrain.append(model_name)
        
        should_retrain = len(models_to_retrain) > 0
        
        return {
            'should_retrain': should_retrain,
            'reason': f"Models to retrain: {models_to_retrain}" if models_to_retrain else "All models performing well",
            'strategy': 'ensemble',
            'ensemble_performance': weighted_performance,
            'models_to_retrain': models_to_retrain
        }
    
    def determine_optimal_strategy(self, current_metrics, historical_data, 
                                 available_strategies=None):
        """Determine the optimal retraining strategy based on current conditions."""
        if available_strategies is None:
            available_strategies = list(self.retraining_strategies.keys())
        
        strategy_results = {}
        
        for strategy_name in available_strategies:
            if strategy_name == 'threshold_based':
                result = self.threshold_based_retraining(
                    current_metrics.get('accuracy', 0.8),
                    threshold=0.05
                )
            elif strategy_name == 'scheduled':
                result = self.scheduled_retraining(
                    current_metrics.get('timestep', 0),
                    frequency=50
                )
            elif strategy_name == 'confidence_based':
                result = self.confidence_based_retraining(
                    current_metrics.get('confidence', 0.8)
                )
            elif strategy_name == 'cost_aware':
                result = self.cost_aware_retraining(
                    performance_gain=0.1,
                    retraining_cost=2.0
                )
            elif strategy_name == 'adaptive':
                result = self.adaptive_retraining(
                    historical_data.get('performance_history', []),
                    historical_data.get('drift_scores', {})
                )
            elif strategy_name == 'ensemble':
                result = self.ensemble_retraining(
                    current_metrics.get('model_performances', {})
                )
            else:
                continue
            
            strategy_results[strategy_name] = result
        
        # Select optimal strategy based on multiple criteria
        optimal_strategy = self._select_optimal_strategy(strategy_results)
        
        return optimal_strategy, strategy_results
    
    def _select_optimal_strategy(self, strategy_results):
        """Select the optimal strategy based on multiple criteria."""
        # Scoring criteria
        scores = {}
        
        for strategy_name, result in strategy_results.items():
            score = 0
            
            # Prefer strategies that recommend retraining when needed
            if result['should_retrain']:
                score += 2
            
            # Prefer cost-aware strategies
            if strategy_name == 'cost_aware':
                score += 1
            
            # Prefer adaptive strategies
            if strategy_name == 'adaptive':
                score += 1
            
            # Penalize overly aggressive strategies
            if strategy_name == 'scheduled':
                score -= 0.5
            
            scores[strategy_name] = score
        
        # Return strategy with highest score
        optimal_strategy = max(scores, key=scores.get)
        
        return optimal_strategy
    
    def execute_retraining(self, strategy_result, model, new_data, new_labels=None):
        """Execute retraining with the specified strategy."""
        start_time = time.time()
        
        # Record retraining start
        retraining_record = {
            'strategy': strategy_result['strategy'],
            'reason': strategy_result['reason'],
            'start_time': start_time,
            'data_size': len(new_data) if hasattr(new_data, '__len__') else 0
        }
        
        try:
            # Execute retraining
            if hasattr(model, 'fit'):
                if new_labels is not None:
                    model.fit(new_data, new_labels)
                else:
                    # For unsupervised models or feature engineering
                    model.fit(new_data)
                
                # Calculate retraining metrics
                if hasattr(model, 'score') and new_labels is not None:
                    new_score = model.score(new_data, new_labels)
                    retraining_record['new_score'] = new_score
                
                retraining_record['success'] = True
                retraining_record['duration'] = time.time() - start_time
                
            else:
                retraining_record['success'] = False
                retraining_record['error'] = "Model does not support retraining"
        
        except Exception as e:
            retraining_record['success'] = False
            retraining_record['error'] = str(e)
            retraining_record['duration'] = time.time() - start_time
        
        # Store retraining history
        self.retraining_history.append(retraining_record)
        
        return retraining_record
    
    def get_retraining_summary(self):
        """Get comprehensive retraining summary."""
        if not self.retraining_history:
            return {
                'total_retrainings': 0,
                'success_rate': 0.0,
                'avg_duration': 0.0,
                'strategy_distribution': {},
                'recent_retrainings': []
            }
        
        successful_retrainings = [r for r in self.retraining_history if r.get('success', False)]
        
        # Strategy distribution
        strategy_counts = {}
        for record in self.retraining_history:
            strategy = record.get('strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_retrainings': len(self.retraining_history),
            'successful_retrainings': len(successful_retrainings),
            'success_rate': len(successful_retrainings) / len(self.retraining_history),
            'avg_duration': np.mean([r.get('duration', 0) for r in self.retraining_history]),
            'strategy_distribution': strategy_counts,
            'recent_retrainings': self.retraining_history[-5:],
            'total_data_processed': sum(r.get('data_size', 0) for r in self.retraining_history)
        }
    
    def analyze_cost_benefit(self, performance_before, performance_after, 
                           retraining_cost, time_period=30):
        """Analyze cost-benefit of retraining decisions."""
        performance_improvement = performance_after - performance_before
        
        # Calculate ROI
        roi = (performance_improvement * 100) / retraining_cost  # Percentage improvement per cost unit
        
        # Calculate break-even point
        break_even_time = retraining_cost / (performance_improvement * 100) if performance_improvement > 0 else float('inf')
        
        # Determine if retraining was worthwhile
        worthwhile = roi > 1.0 and break_even_time < time_period
        
        return {
            'performance_improvement': performance_improvement,
            'roi': roi,
            'break_even_time': break_even_time,
            'worthwhile': worthwhile,
            'recommendation': 'Continue retraining' if worthwhile else 'Review retraining strategy'
        }


class RetrainingScheduler:
    """Advanced scheduler for coordinating retraining across multiple pipeline stages."""
    
    def __init__(self):
        self.stage_priorities = {
            'data_ingestion': 1,
            'feature_engineering': 2,
            'embedding_generation': 3,
            'primary_classifier': 4,
            'secondary_classifier': 5,
            'post_processing': 6
        }
        self.retraining_queue = []
        self.scheduled_retrainings = {}
        
    def schedule_retraining(self, stage_name, strategy, priority_override=None):
        """Schedule retraining for a specific stage."""
        priority = priority_override or self.stage_priorities.get(stage_name, 10)
        
        retraining_task = {
            'stage': stage_name,
            'strategy': strategy,
            'priority': priority,
            'scheduled_time': time.time(),
            'status': 'scheduled'
        }
        
        self.retraining_queue.append(retraining_task)
        
        # Sort by priority
        self.retraining_queue.sort(key=lambda x: x['priority'])
        
        return retraining_task
    
    def get_next_retraining_task(self):
        """Get the next retraining task from the queue."""
        if not self.retraining_queue:
            return None
        
        return self.retraining_queue.pop(0)
    
    def cancel_retraining(self, stage_name):
        """Cancel scheduled retraining for a stage."""
        self.retraining_queue = [task for task in self.retraining_queue if task['stage'] != stage_name]
    
    def get_queue_status(self):
        """Get current queue status."""
        return {
            'queue_length': len(self.retraining_queue),
            'next_task': self.retraining_queue[0] if self.retraining_queue else None,
            'stage_distribution': self._get_stage_distribution()
        }
    
    def _get_stage_distribution(self):
        """Get distribution of stages in the queue."""
        stage_counts = {}
        for task in self.retraining_queue:
            stage = task['stage']
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        return stage_counts 