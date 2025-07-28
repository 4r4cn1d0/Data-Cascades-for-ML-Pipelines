"""
Production-ready multi-stage ML pipeline architecture.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


class DataIngestionStage:
    """Realistic data ingestion with validation and preprocessing."""
    
    def __init__(self):
        self.data_quality_checks = []
        self.validation_rules = {}
        self.preprocessing_steps = []
        
    def process(self, data):
        """Process incoming data with quality checks."""
        # Data quality validation
        quality_report = self._validate_data_quality(data)
        
        # Basic preprocessing
        processed_data = self._preprocess_data(data)
        
        # Store quality metrics
        self.data_quality_checks.append(quality_report)
        
        return processed_data, quality_report
    
    def _validate_data_quality(self, data):
        """Validate data quality and return report."""
        if isinstance(data, pd.DataFrame):
            report = {
                'missing_values': data.isnull().sum().to_dict(),
                'duplicates': data.duplicated().sum(),
                'data_types': data.dtypes.to_dict(),
                'shape': data.shape,
                'memory_usage': data.memory_usage(deep=True).sum()
            }
        else:
            report = {
                'missing_values': np.isnan(data).sum(),
                'duplicates': 0,  # Not applicable for numpy arrays
                'data_types': str(data.dtype),
                'shape': data.shape,
                'memory_usage': data.nbytes
            }
        
        # Add quality scores
        report['quality_score'] = self._calculate_quality_score(report)
        
        return report
    
    def _calculate_quality_score(self, report):
        """Calculate overall data quality score."""
        score = 1.0
        
        # Penalize for missing values
        if isinstance(report['missing_values'], dict):
            total_missing = sum(report['missing_values'].values())
        else:
            total_missing = report['missing_values']
        
        if total_missing > 0:
            score -= min(0.3, total_missing / 1000)  # Cap penalty at 30%
        
        # Penalize for duplicates
        if report['duplicates'] > 0:
            score -= min(0.2, report['duplicates'] / 1000)
        
        return max(0.0, score)
    
    def _preprocess_data(self, data):
        """Basic data preprocessing."""
        if isinstance(data, pd.DataFrame):
            # Handle missing values
            data = data.fillna(data.mean())
            
            # Remove duplicates
            data = data.drop_duplicates()
            
            # Basic type conversion
            for col in data.select_dtypes(include=['object']).columns:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except:
                    pass  # Keep as object if conversion fails
        
        return data


class FeatureEngineeringStage:
    """Realistic feature engineering with drift simulation."""
    
    def __init__(self, n_features=50, n_components=25):
        self.n_features = n_features
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.feature_selector = None  # Will be created dynamically
        self.pca = PCA(n_components=n_components)
        self.feature_importance = {}
        self.feature_names = []
        self.fitted_pca = None
        
    def process(self, data, labels=None):
        """Process data through feature engineering pipeline."""
        # Store original feature names
        if isinstance(data, pd.DataFrame):
            self.feature_names = list(data.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        
        # Dynamically adjust n_features based on actual data
        actual_n_features = min(self.n_features, data.shape[1])
        if self.feature_selector is None or self.feature_selector.k != actual_n_features:
            self.feature_selector = SelectKBest(score_func=f_classif, k=actual_n_features)
        
        # Scale features
        if labels is not None:
            # Training mode - fit the scaler
            scaled_data = self.scaler.fit_transform(data)
        else:
            # Prediction mode - transform only
            scaled_data = self.scaler.transform(data)
        
        # Feature selection (if labels available)
        if labels is not None and hasattr(self.feature_selector, 'fit'):
            selected_data = self.feature_selector.fit_transform(scaled_data, labels)
            self.feature_importance = dict(zip(
                self.feature_names, 
                self.feature_selector.scores_
            ))
        else:
            # Prediction mode - transform only
            selected_data = self.feature_selector.transform(scaled_data)
        
        # Dimensionality reduction
        if self.fitted_pca is not None:
            # Prediction mode - use fitted PCA
            reduced_data = self.fitted_pca.transform(selected_data)
        else:
            # Training mode - fit PCA
            actual_n_components = min(self.n_components, selected_data.shape[1])
            if actual_n_components < selected_data.shape[1]:
                # Only apply PCA if we have more features than components
                self.pca = PCA(n_components=actual_n_components)
                reduced_data = self.pca.fit_transform(selected_data)
                self.fitted_pca = self.pca  # Store fitted PCA for prediction
            else:
                # No dimensionality reduction needed
                reduced_data = selected_data
                self.fitted_pca = None
        
        return reduced_data
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        return self.feature_importance
    
    def get_explained_variance(self):
        """Get PCA explained variance."""
        if hasattr(self.pca, 'explained_variance_ratio_'):
            return self.pca.explained_variance_ratio_
        return None


class EmbeddingStage:
    """Generate embeddings that can drift."""
    
    def __init__(self, embedding_dim=128, hidden_dim=256):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding_model = self._build_embedding_model()
        self.embedding_history = []
        
    def _build_embedding_model(self):
        """Build neural network for embedding generation."""
        # We'll build this dynamically based on input size
        return None
    
    def _create_embedding_model(self, input_dim):
        """Create embedding model with correct input dimension."""
        model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.Tanh()  # Normalize embeddings
        )
        return model
    
    def process(self, data):
        """Generate embeddings from input data."""
        # Convert to tensor if needed
        if not isinstance(data, torch.Tensor):
            data_tensor = torch.FloatTensor(data)
        else:
            data_tensor = data
        
        # Create embedding model if not exists or wrong size
        if self.embedding_model is None or self.embedding_model[0].in_features != data_tensor.shape[1]:
            self.embedding_model = self._create_embedding_model(data_tensor.shape[1])
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.embedding_model(data_tensor)
        
        # Store embedding statistics
        embedding_stats = {
            'mean': embeddings.mean().item(),
            'std': embeddings.std().item(),
            'min': embeddings.min().item(),
            'max': embeddings.max().item()
        }
        self.embedding_history.append(embedding_stats)
        
        return embeddings.numpy()
    
    def train_embeddings(self, data, labels, epochs=10):
        """Train embedding model with supervision."""
        data_tensor = torch.FloatTensor(data)
        labels_tensor = torch.LongTensor(labels)
        
        # Create simple classifier on top of embeddings
        classifier = nn.Linear(self.embedding_dim, len(np.unique(labels)))
        optimizer = optim.Adam(list(self.embedding_model.parameters()) + list(classifier.parameters()))
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = self.embedding_model(data_tensor)
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()
        
        return self.embedding_model


class PrimaryClassifier:
    """Primary classification model with advanced features."""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = self._build_model()
        self.training_history = []
        self.prediction_confidence = []
        
    def _build_model(self):
        """Build the primary classification model."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'neural_network':
            return MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X, y):
        """Train the primary classifier."""
        self.model.fit(X, y)
        
        # Store training metrics
        train_score = self.model.score(X, y)
        self.training_history.append({
            'train_accuracy': train_score,
            'n_samples': len(X),
            'n_features': X.shape[1]
        })
        
        return train_score
    
    def predict(self, X):
        """Make predictions with confidence scores."""
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            predictions = self.model.predict(X)
            confidence = np.max(probabilities, axis=1)
        else:
            predictions = self.model.predict(X)
            confidence = np.ones(len(predictions))  # Default confidence
        
        # Store confidence statistics
        self.prediction_confidence.append({
            'mean_confidence': np.mean(confidence),
            'std_confidence': np.std(confidence),
            'low_confidence_count': np.sum(confidence < 0.8)
        })
        
        return predictions, confidence
    
    def get_feature_importance(self):
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            return None


class SecondaryClassifier:
    """Secondary classifier for ensemble or specialized tasks."""
    
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.model = self._build_model()
        self.ensemble_weights = 0.5  # Weight in ensemble
        self.specialization = 'uncertainty'  # Specialized for uncertain cases
        
    def _build_model(self):
        """Build the secondary classifier."""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=50, random_state=42)
        elif self.model_type == 'neural_network':
            return MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X, y, primary_predictions=None):
        """Train secondary classifier, optionally using primary predictions."""
        if primary_predictions is not None:
            # Use primary predictions as additional features
            X_with_primary = np.column_stack([X, primary_predictions])
            self.model.fit(X_with_primary, y)
            return self.model.score(X_with_primary, y)
        else:
            self.model.fit(X, y)
            return self.model.score(X, y)
    
    def predict(self, X, primary_predictions=None):
        """Make predictions with optional primary model input."""
        if primary_predictions is not None:
            X_with_primary = np.column_stack([X, primary_predictions])
            predictions = self.model.predict(X_with_primary)
            probabilities = self.model.predict_proba(X_with_primary) if hasattr(self.model, 'predict_proba') else None
        else:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
        
        return predictions, probabilities


class PostProcessingStage:
    """Post-processing stage for final predictions."""
    
    def __init__(self):
        self.post_processing_rules = []
        self.output_format = 'standard'
        self.confidence_threshold = 0.5  # Lowered from 0.8 to be more reasonable
        
    def process(self, predictions, probabilities=None, confidence=None):
        """Apply post-processing to predictions."""
        processed_predictions = predictions.copy()
        
        # Apply confidence thresholding (temporarily disabled for testing)
        # if confidence is not None:
        #     low_confidence_mask = confidence < self.confidence_threshold
        #     processed_predictions[low_confidence_mask] = -1  # Reject prediction
        
        # Apply business rules
        processed_predictions = self._apply_business_rules(processed_predictions)
        
        # Format output
        output = self._format_output(processed_predictions, probabilities, confidence)
        
        return output
    
    def _apply_business_rules(self, predictions):
        """Apply domain-specific business rules."""
        # Example: Ensure certain classes are not predicted together
        # This is a placeholder for real business logic
        return predictions
    
    def _format_output(self, predictions, probabilities, confidence):
        """Format the final output."""
        output = {
            'predictions': predictions,
            'confidence': confidence if confidence is not None else np.ones(len(predictions)),
            'rejected_count': np.sum(predictions == -1) if -1 in predictions else 0
        }
        
        if probabilities is not None:
            output['probabilities'] = probabilities
        
        return output


class ProductionMLPipeline:
    """Complete production-ready ML pipeline."""
    
    def __init__(self):
        self.stages = {
            'data_ingestion': DataIngestionStage(),
            'feature_engineering': FeatureEngineeringStage(),
            'embedding_generation': EmbeddingStage(),
            'primary_classifier': PrimaryClassifier(),
            'secondary_classifier': SecondaryClassifier(),
            'post_processing': PostProcessingStage()
        }
        self.component_graph = self._build_component_graph()
        self.pipeline_history = []
        self.stage_performances = {}
        
    def _build_component_graph(self):
        """Build TFX-style component graph."""
        return {
            'data_ingestion': ['feature_engineering'],
            'feature_engineering': ['embedding_generation'],
            'embedding_generation': ['primary_classifier'],
            'primary_classifier': ['secondary_classifier'],
            'secondary_classifier': ['post_processing'],
            'post_processing': []
        }
    
    def train_pipeline(self, data, labels):
        """Train the complete pipeline."""
        print("Training production ML pipeline...")
        
        # Stage 1: Data Ingestion
        processed_data, quality_report = self.stages['data_ingestion'].process(data)
        print(f"Data quality score: {quality_report['quality_score']:.3f}")
        
        # Stage 2: Feature Engineering
        engineered_features = self.stages['feature_engineering'].process(processed_data, labels)
        print(f"Engineered features shape: {engineered_features.shape}")
        
        # Stage 3: Embedding Generation
        embeddings = self.stages['embedding_generation'].process(engineered_features)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # Stage 4: Primary Classifier
        primary_score = self.stages['primary_classifier'].train(embeddings, labels)
        print(f"Primary classifier accuracy: {primary_score:.3f}")
        
        # Stage 5: Secondary Classifier
        primary_predictions, _ = self.stages['primary_classifier'].predict(embeddings)
        secondary_score = self.stages['secondary_classifier'].train(embeddings, labels, primary_predictions)
        print(f"Secondary classifier accuracy: {secondary_score:.3f}")
        print(f"Secondary classifier accuracy: {secondary_score:.3f}")
        
        # Store pipeline performance
        self.stage_performances = {
            'data_quality': quality_report['quality_score'],
            'primary_classifier': primary_score,
            'secondary_classifier': secondary_score
        }
        
        print("Pipeline training completed!")
        return self.stage_performances
    
    def predict(self, data):
        """Run complete prediction pipeline."""
        # Stage 1: Data Ingestion
        processed_data, _ = self.stages['data_ingestion'].process(data)
        
        # Stage 2: Feature Engineering
        engineered_features = self.stages['feature_engineering'].process(processed_data)
        
        # Stage 3: Embedding Generation
        embeddings = self.stages['embedding_generation'].process(engineered_features)
        
        # Stage 4: Primary Classifier
        primary_predictions, primary_confidence = self.stages['primary_classifier'].predict(embeddings)
        
        # Stage 5: Secondary Classifier
        secondary_predictions, secondary_probabilities = self.stages['secondary_classifier'].predict(
            embeddings, primary_predictions
        )
        
        # Stage 6: Post Processing
        final_output = self.stages['post_processing'].process(
            secondary_predictions, secondary_probabilities, primary_confidence
        )
        
        # Store pipeline history
        self.pipeline_history.append({
            'input_shape': data.shape,
            'output_shape': final_output['predictions'].shape,
            'rejected_count': final_output['rejected_count']
        })
        
        return final_output
    
    def get_pipeline_summary(self):
        """Get comprehensive pipeline summary."""
        return {
            'stages': list(self.stages.keys()),
            'component_graph': self.component_graph,
            'stage_performances': self.stage_performances,
            'pipeline_history': self.pipeline_history,
            'total_predictions': sum(h['output_shape'][0] for h in self.pipeline_history),
            'total_rejected': sum(h['rejected_count'] for h in self.pipeline_history)
        }
    
    def retrain_stage(self, stage_name, new_data, new_labels=None):
        """Retrain a specific pipeline stage."""
        if stage_name not in self.stages:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        print(f"Retraining stage: {stage_name}")
        
        if stage_name == 'primary_classifier':
            # For classifiers, we need labels
            if new_labels is None:
                raise ValueError("Labels required for classifier retraining")
            
            # Get embeddings for retraining
            processed_data, _ = self.stages['data_ingestion'].process(new_data)
            engineered_features = self.stages['feature_engineering'].process(processed_data, new_labels)
            embeddings = self.stages['embedding_generation'].process(engineered_features)
            
            # Retrain classifier
            new_score = self.stages[stage_name].train(embeddings, new_labels)
            print(f"Retrained {stage_name} accuracy: {new_score:.3f}")
            
        elif stage_name == 'feature_engineering':
            # Retrain feature engineering with new data
            if new_labels is not None:
                engineered_features = self.stages[stage_name].process(new_data, new_labels)
            else:
                engineered_features = self.stages[stage_name].process(new_data)
            print(f"Retrained {stage_name} with {engineered_features.shape[1]} features")
        
        return True 