"""
ML Pipeline stages for cascade effect simulation.
Implements feature extraction, classification, and post-processing stages.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractionStage:
    """Stage 1: Feature extraction with drift simulation."""
    
    def __init__(self, n_features=10, feature_drift_rate=0.01):
        self.n_features = n_features
        self.feature_drift_rate = feature_drift_rate
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.is_fitted = False
        self.feature_importance = None
        self.drift_history = []
        
    def fit(self, X, y=None):
        """Fit the feature extraction stage."""
        # Fit scaler
        self.scaler.fit(X)
        
        # Create polynomial features
        X_scaled = self.scaler.transform(X)
        X_poly = self.poly_features.fit_transform(X_scaled)
        
        # Calculate feature importance (correlation with target if available)
        if y is not None:
            correlations = []
            for i in range(X_poly.shape[1]):
                if len(np.unique(y)) > 1:
                    corr = np.corrcoef(X_poly[:, i], y)[0, 1]
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)
                else:
                    correlations.append(0)
            self.feature_importance = np.array(correlations)
        else:
            self.feature_importance = np.ones(X_poly.shape[1])
        
        self.is_fitted = True
        return self
    
    def transform(self, X, time_step=0):
        """Transform input data with potential drift."""
        if not self.is_fitted:
            raise ValueError("Feature extraction stage must be fitted first.")
        
        # Apply scaling
        X_scaled = self.scaler.transform(X)
        
        # Apply polynomial features
        X_poly = self.poly_features.transform(X_scaled)
        
        # Simulate feature drift over time
        if time_step > 0:
            drift_factor = time_step * self.feature_drift_rate
            
            # Add noise to features based on their importance
            for i in range(X_poly.shape[1]):
                importance = self.feature_importance[i] if i < len(self.feature_importance) else 1.0
                noise = np.random.normal(0, drift_factor * importance, X_poly.shape[0])
                X_poly[:, i] = X_poly[:, i] + noise
            
            # Record drift
            self.drift_history.append({
                'time_step': time_step,
                'drift_factor': drift_factor,
                'affected_features': list(range(X_poly.shape[1]))
            })
        
        return X_poly
    
    def get_feature_names(self):
        """Get feature names after transformation."""
        if not self.is_fitted:
            return []
        
        base_features = [f'feature_{i}' for i in range(self.n_features)]
        poly_names = self.poly_features.get_feature_names_out(base_features)
        return poly_names.tolist()


class ClassificationStage:
    """Stage 2: Classification with performance degradation simulation."""
    
    def __init__(self, n_classes=3, degradation_rate=0.005):
        self.n_classes = n_classes
        self.degradation_rate = degradation_rate
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_fitted = False
        self.base_performance = None
        self.performance_history = []
        
    def fit(self, X, y):
        """Fit the classification stage."""
        self.classifier.fit(X, y)
        self.is_fitted = True
        
        # Record base performance
        y_pred = self.classifier.predict(X)
        self.base_performance = {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred, average='weighted'),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted')
        }
        
        return self
    
    def predict(self, X, time_step=0):
        """Make predictions with simulated performance degradation."""
        if not self.is_fitted:
            raise ValueError("Classification stage must be fitted first.")
        
        # Get base predictions
        y_pred = self.classifier.predict(X)
        y_proba = self.classifier.predict_proba(X)
        
        # Simulate performance degradation over time
        if time_step > 0:
            degradation_factor = time_step * self.degradation_rate
            
            # Add noise to predictions based on confidence
            for i in range(len(y_pred)):
                confidence = np.max(y_proba[i])
                
                # Higher confidence predictions are less likely to be affected
                if confidence < 0.8:
                    # Add noise to low confidence predictions
                    noise = np.random.normal(0, degradation_factor)
                    if noise > 0.3:  # 30% chance of wrong prediction
                        # Flip to a different class
                        wrong_class = np.random.choice(self.n_classes)
                        while wrong_class == y_pred[i]:
                            wrong_class = np.random.choice(self.n_classes)
                        y_pred[i] = wrong_class
            
            # Record performance
            if hasattr(self, 'true_labels') and self.true_labels is not None:
                current_performance = {
                    'time_step': time_step,
                    'accuracy': accuracy_score(self.true_labels, y_pred),
                    'f1': f1_score(self.true_labels, y_pred, average='weighted'),
                    'precision': precision_score(self.true_labels, y_pred, average='weighted'),
                    'recall': recall_score(self.true_labels, y_pred, average='weighted'),
                    'degradation_factor': degradation_factor
                }
                self.performance_history.append(current_performance)
        
        return y_pred, y_proba
    
    def set_true_labels(self, y_true):
        """Set true labels for performance tracking."""
        self.true_labels = y_true
    
    def get_performance_summary(self):
        """Get performance summary over time."""
        if not self.performance_history:
            return self.base_performance
        
        return {
            'base_performance': self.base_performance,
            'current_performance': self.performance_history[-1] if self.performance_history else None,
            'performance_history': self.performance_history
        }


class PostProcessingStage:
    """Stage 3: Post-processing with business rules and cascade effects."""
    
    def __init__(self, confidence_threshold=0.7, cascade_sensitivity=0.1):
        self.confidence_threshold = confidence_threshold
        self.cascade_sensitivity = cascade_sensitivity
        self.business_rules = []
        self.cascade_history = []
        
    def add_business_rule(self, rule_func, rule_name):
        """Add a business rule function."""
        self.business_rules.append({
            'function': rule_func,
            'name': rule_name
        })
    
    def process(self, predictions, probabilities, time_step=0):
        """Apply post-processing with cascade effects."""
        final_decisions = []
        cascade_effects = []
        
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            confidence = np.max(proba)
            original_pred = pred
            
            # Apply business rules
            for rule in self.business_rules:
                pred = rule['function'](pred, confidence, proba)
            
            # Simulate cascade effects from upstream errors
            if time_step > 0:
                # Cascade effect: errors from previous stages affect this stage
                cascade_factor = time_step * self.cascade_sensitivity
                
                # If confidence is low, cascade effects are more likely
                if confidence < self.confidence_threshold:
                    cascade_noise = np.random.normal(0, cascade_factor)
                    if cascade_noise > 0.5:
                        # Cascade effect: change prediction
                        new_pred = np.random.choice(len(proba))
                        while new_pred == pred:
                            new_pred = np.random.choice(len(proba))
                        pred = new_pred
                
                cascade_effects.append({
                    'original_pred': original_pred,
                    'final_pred': pred,
                    'confidence': confidence,
                    'cascade_factor': cascade_factor,
                    'cascade_affected': pred != original_pred
                })
            
            final_decisions.append(pred)
        
        # Record cascade effects
        if cascade_effects:
            self.cascade_history.append({
                'time_step': time_step,
                'cascade_effects': cascade_effects,
                'cascade_rate': sum(1 for effect in cascade_effects if effect['cascade_affected']) / len(cascade_effects)
            })
        
        return final_decisions
    
    def get_cascade_summary(self):
        """Get cascade effect summary."""
        if not self.cascade_history:
            return "No cascade effects recorded."
        
        total_effects = sum(len(entry['cascade_effects']) for entry in self.cascade_history)
        affected_predictions = sum(
            sum(1 for effect in entry['cascade_effects'] if effect['cascade_affected'])
            for entry in self.cascade_history
        )
        
        return {
            'total_predictions': total_effects,
            'affected_predictions': affected_predictions,
            'cascade_rate': affected_predictions / total_effects if total_effects > 0 else 0,
            'cascade_history': self.cascade_history
        }


class MLPipeline:
    """Complete ML pipeline with all stages."""
    
    def __init__(self, n_features=10, n_classes=3):
        self.feature_stage = FeatureExtractionStage(n_features=n_features)
        self.classification_stage = ClassificationStage(n_classes=n_classes)
        self.post_processing_stage = PostProcessingStage()
        
        # Add some default business rules
        self._setup_default_business_rules()
        
        self.pipeline_history = []
        
    def _setup_default_business_rules(self):
        """Setup default business rules."""
        
        def high_confidence_rule(pred, confidence, proba):
            """High confidence predictions are more reliable."""
            if confidence > 0.9:
                return pred  # Keep high confidence predictions
            elif confidence < 0.5:
                # Low confidence: apply conservative approach
                return np.argmax(proba)  # Most likely class
            return pred
        
        def uncertainty_penalty(pred, confidence, proba):
            """Apply penalty for uncertain predictions."""
            if confidence < 0.6:
                # Apply penalty: prefer class 0 (assumed to be "safe" class)
                return 0
            return pred
        
        self.post_processing_stage.add_business_rule(high_confidence_rule, "high_confidence_rule")
        self.post_processing_stage.add_business_rule(uncertainty_penalty, "uncertainty_penalty")
    
    def fit(self, X, y):
        """Fit all pipeline stages."""
        # Fit feature extraction
        self.feature_stage.fit(X, y)
        
        # Transform data for classification
        X_transformed = self.feature_stage.transform(X)
        
        # Fit classification
        self.classification_stage.fit(X_transformed, y)
        
        return self
    
    def predict(self, X, time_step=0):
        """Run complete pipeline prediction."""
        # Stage 1: Feature extraction
        X_transformed = self.feature_stage.transform(X, time_step)
        
        # Stage 2: Classification
        predictions, probabilities = self.classification_stage.predict(X_transformed, time_step)
        
        # Stage 3: Post-processing
        final_predictions = self.post_processing_stage.process(predictions, probabilities, time_step)
        
        # Record pipeline execution
        self.pipeline_history.append({
            'time_step': time_step,
            'input_shape': X.shape,
            'transformed_shape': X_transformed.shape,
            'predictions': final_predictions,
            'probabilities': probabilities
        })
        
        return final_predictions, probabilities
    
    def get_pipeline_summary(self):
        """Get comprehensive pipeline summary."""
        return {
            'feature_stage': {
                'drift_history': self.feature_stage.drift_history,
                'feature_importance': self.feature_stage.feature_importance
            },
            'classification_stage': {
                'performance': self.classification_stage.get_performance_summary(),
                'degradation_rate': self.classification_stage.degradation_rate
            },
            'post_processing_stage': {
                'cascade_summary': self.post_processing_stage.get_cascade_summary(),
                'business_rules': [rule['name'] for rule in self.post_processing_stage.business_rules]
            },
            'pipeline_history': self.pipeline_history
        }


if __name__ == "__main__":
    # Test the pipeline stages
    from src.data.synthetic_data import SyntheticDataGenerator
    
    # Generate test data
    generator = SyntheticDataGenerator(n_samples=1000, n_features=10, n_classes=3)
    X, y = generator.generate_base_data()
    
    print("Testing pipeline stages...")
    
    # Test feature extraction
    feature_stage = FeatureExtractionStage()
    feature_stage.fit(X, y)
    X_transformed = feature_stage.transform(X)
    print(f"Feature extraction: {X.shape} -> {X_transformed.shape}")
    
    # Test classification
    classification_stage = ClassificationStage()
    classification_stage.fit(X_transformed, y)
    predictions, probabilities = classification_stage.predict(X_transformed)
    print(f"Classification accuracy: {accuracy_score(y, predictions):.3f}")
    
    # Test post-processing
    post_stage = PostProcessingStage()
    final_predictions = post_stage.process(predictions, probabilities)
    print(f"Post-processing completed: {len(final_predictions)} predictions")
    
    # Test complete pipeline
    pipeline = MLPipeline()
    pipeline.fit(X, y)
    final_preds, final_probs = pipeline.predict(X)
    print(f"Complete pipeline accuracy: {accuracy_score(y, final_preds):.3f}")
    
    print("\nAll pipeline stages working correctly!") 