"""
Advanced visualization suite for drift analysis and error propagation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP for feature attribution
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# Try to import torch for saliency maps
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


class AdvancedVisualizations:
    """Advanced visualization techniques for drift analysis and error propagation."""
    
    def __init__(self):
        self.feature_attribution_data = {}
        self.error_propagation_data = {}
        self.saliency_maps = {}
        
    def create_feature_attribution_plot(self, model, X_sample, feature_names=None, 
                                       method='shap', max_features=20):
        """Create feature attribution plot using SHAP or other methods."""
        if method == 'shap' and SHAP_AVAILABLE:
            return self._create_shap_plot(model, X_sample, feature_names, max_features)
        elif method == 'permutation':
            return self._create_permutation_importance_plot(model, X_sample, feature_names, max_features)
        else:
            return self._create_basic_importance_plot(model, X_sample, feature_names, max_features)
    
    def _create_shap_plot(self, model, X_sample, feature_names, max_features):
        """Create SHAP-based feature attribution plot."""
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict_proba, X_sample[:100])
            else:
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict, X_sample[:100])
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(X_sample.shape[1])]
            
            # Calculate mean absolute SHAP values
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_shap_values)[-max_features:]
            top_features = [feature_names[i] for i in top_indices]
            top_values = mean_shap_values[top_indices]
            
            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_values,
                y=top_features,
                orientation='h',
                marker_color='lightcoral',
                name='SHAP Importance'
            ))
            
            fig.update_layout(
                title="Feature Attribution (SHAP)",
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Features",
                height=400 + len(top_features) * 20,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"SHAP analysis failed: {e}")
            return self._create_basic_importance_plot(model, X_sample, feature_names, max_features)
    
    def _create_permutation_importance_plot(self, model, X_sample, feature_names, max_features):
        """Create permutation importance plot."""
        from sklearn.inspection import permutation_importance
        
        try:
            # Calculate permutation importance
            result = permutation_importance(model, X_sample, np.zeros(len(X_sample)), 
                                         n_repeats=10, random_state=42)
            
            # Get top features
            top_indices = np.argsort(result.importances_mean)[-max_features:]
            top_features = [feature_names[i] if feature_names else f'Feature_{i}' for i in top_indices]
            top_values = result.importances_mean[top_indices]
            
            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_values,
                y=top_features,
                orientation='h',
                marker_color='lightblue',
                name='Permutation Importance'
            ))
            
            fig.update_layout(
                title="Feature Attribution (Permutation Importance)",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=400 + len(top_features) * 20,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Permutation importance failed: {e}")
            return self._create_basic_importance_plot(model, X_sample, feature_names, max_features)
    
    def _create_basic_importance_plot(self, model, X_sample, feature_names, max_features):
        """Create basic feature importance plot."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                # Fallback: use variance of features
                importance = np.var(X_sample, axis=0)
            
            # Get top features
            top_indices = np.argsort(importance)[-max_features:]
            top_features = [feature_names[i] if feature_names else f'Feature_{i}' for i in top_indices]
            top_values = importance[top_indices]
            
            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_values,
                y=top_features,
                orientation='h',
                marker_color='lightgreen',
                name='Feature Importance'
            ))
            
            fig.update_layout(
                title="Feature Attribution (Basic)",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=400 + len(top_features) * 20,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Feature importance analysis failed: {e}")
            return None
    
    def create_error_propagation_heatmap(self, pipeline_stages, error_data):
        """Create heatmap showing error propagation through pipeline."""
        stage_names = list(pipeline_stages.keys())
        
        # Create error matrix
        error_matrix = np.zeros((len(stage_names), len(stage_names)))
        
        for i, stage1 in enumerate(stage_names):
            for j, stage2 in enumerate(stage_names):
                if stage1 in error_data and stage2 in error_data:
                    # Calculate correlation between errors
                    errors1 = error_data[stage1]
                    errors2 = error_data[stage2]
                    
                    if len(errors1) == len(errors2) and len(errors1) > 1:
                        correlation = np.corrcoef(errors1, errors2)[0, 1]
                        if not np.isnan(correlation):
                            error_matrix[i, j] = abs(correlation)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=error_matrix,
            x=stage_names,
            y=stage_names,
            colorscale='Reds',
            zmin=0,
            zmax=1,
            text=np.round(error_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Error Propagation Heatmap",
            xaxis_title="Pipeline Stages",
            yaxis_title="Pipeline Stages",
            height=500,
            width=600
        )
        
        return fig
    
    def create_drift_contribution_analysis(self, feature_names, drift_scores, 
                                         performance_history, max_features=15):
        """Analyze which features contribute most to drift and performance degradation."""
        if not feature_names or not drift_scores:
            return None
        
        # Calculate drift contribution
        drift_contributions = {}
        for feature, score in drift_scores.items():
            if feature in feature_names:
                drift_contributions[feature] = score
        
        # Get top drifting features
        sorted_features = sorted(drift_contributions.items(), key=lambda x: x[1], reverse=True)
        top_drift_features = sorted_features[:max_features]
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Top Drift-Contributing Features", "Performance vs Drift"),
            vertical_spacing=0.1
        )
        
        # Feature drift scores
        feature_names_plot = [f[0] for f in top_drift_features]
        drift_values = [f[1] for f in top_drift_features]
        
        fig.add_trace(
            go.Bar(x=feature_names_plot, y=drift_values, name="Drift Score"),
            row=1, col=1
        )
        
        # Performance trend
        if performance_history:
            timesteps = list(range(len(performance_history)))
            fig.add_trace(
                go.Scatter(x=timesteps, y=performance_history, name="Performance", mode='lines+markers'),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Drift Contribution Analysis",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_saliency_maps(self, model, images, labels, num_samples=5):
        """Generate saliency maps for image data."""
        if not TORCH_AVAILABLE:
            st.warning("PyTorch not available for saliency maps")
            return None
        
        try:
            saliency_maps = []
            
            # Convert model to torch if needed
            if not isinstance(model, torch.nn.Module):
                st.warning("Model must be a PyTorch module for saliency maps")
                return None
            
            model.eval()
            
            for i in range(min(num_samples, len(images))):
                img = images[i]
                label = labels[i]
                
                # Convert to tensor
                if not isinstance(img, torch.Tensor):
                    img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad=True)
                else:
                    img_tensor = img.clone().detach().requires_grad_(True)
                
                # Forward pass
                output = model(img_tensor.unsqueeze(0))
                
                # Backward pass for target class
                if len(output.shape) > 1:
                    output[0, label].backward()
                else:
                    output.backward()
                
                # Get gradients
                saliency_map = img_tensor.grad.abs().squeeze().numpy()
                saliency_maps.append(saliency_map)
            
            # Create visualization
            fig = make_subplots(
                rows=len(saliency_maps), cols=2,
                subplot_titles=[f"Sample {i+1}" for i in range(len(saliency_maps))] * 2
            )
            
            for i, (img, saliency) in enumerate(zip(images[:len(saliency_maps)], saliency_maps)):
                # Original image
                fig.add_trace(
                    go.Heatmap(z=img, colorscale='gray', showscale=False),
                    row=i+1, col=1
                )
                
                # Saliency map
                fig.add_trace(
                    go.Heatmap(z=saliency, colorscale='Reds', showscale=False),
                    row=i+1, col=2
                )
            
            fig.update_layout(
                title="Saliency Maps for Drift Analysis",
                height=200 * len(saliency_maps),
                width=800
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Saliency map generation failed: {e}")
            return None
    
    def create_cascade_network_visualization(self, cascade_data, stage_names):
        """Create network visualization of cascade effects."""
        import networkx as nx
        
        try:
            # Create graph
            G = nx.DiGraph()
            
            # Add nodes
            for stage in stage_names:
                G.add_node(stage)
            
            # Add edges with weights based on cascade strength
            for i, stage1 in enumerate(stage_names):
                for j, stage2 in enumerate(stage_names):
                    if i < j and stage1 in cascade_data and stage2 in cascade_data:
                        cascade_strength = cascade_data.get(f"{stage1}_{stage2}", 0)
                        if cascade_strength > 0.1:  # Only show significant connections
                            G.add_edge(stage1, stage2, weight=cascade_strength)
            
            # Calculate layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Create plot
            fig = go.Figure()
            
            # Add edges
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(edge[2]['weight'])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='gray'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            node_x = []
            node_y = []
            node_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(size=20, color='lightblue'),
                textfont=dict(size=10)
            ))
            
            fig.update_layout(
                title="Cascade Effect Network",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500
            )
            
            return fig
            
        except ImportError:
            st.warning("NetworkX not available for cascade network visualization")
            return None
        except Exception as e:
            st.warning(f"Cascade network visualization failed: {e}")
            return None
    
    def create_performance_degradation_timeline(self, performance_history, drift_scores_history, 
                                              retraining_events=None):
        """Create comprehensive timeline showing performance degradation and drift."""
        timesteps = list(range(len(performance_history)))
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Performance Over Time", "Drift Scores", "Cumulative Drift"),
            vertical_spacing=0.1
        )
        
        # Performance timeline
        fig.add_trace(
            go.Scatter(x=timesteps, y=performance_history, name="Performance", mode='lines+markers'),
            row=1, col=1
        )
        
        # Add retraining events if available
        if retraining_events:
            for event in retraining_events:
                fig.add_vline(x=event['timestep'], line_dash="dash", line_color="red", 
                             annotation_text="Retraining", row=1, col=1)
        
        # Drift scores
        if drift_scores_history:
            avg_drift_scores = [np.mean(list(scores.values())) if scores else 0 
                              for scores in drift_scores_history]
            fig.add_trace(
                go.Scatter(x=timesteps, y=avg_drift_scores, name="Avg Drift Score", mode='lines+markers'),
                row=2, col=1
            )
        
        # Cumulative drift
        if drift_scores_history:
            cumulative_drift = np.cumsum(avg_drift_scores)
            fig.add_trace(
                go.Scatter(x=timesteps, y=cumulative_drift, name="Cumulative Drift", mode='lines'),
                row=3, col=1
            )
        
        fig.update_layout(
            title="Performance Degradation Timeline",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_alert_dashboard(self, monitoring_summary):
        """Create alert dashboard with actionable insights."""
        alerts = []
        
        # Performance alerts
        if 'performance_history' in monitoring_summary:
            recent_performance = monitoring_summary['performance_history'][-5:]
            if len(recent_performance) >= 2:
                performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
                if performance_trend < -0.01:
                    alerts.append({
                        'type': 'warning',
                        'message': f"Performance declining (trend: {performance_trend:.3f})",
                        'severity': 'medium'
                    })
        
        # Drift alerts
        if 'drift_analysis' in monitoring_summary:
            drift_scores = monitoring_summary['drift_analysis']
            high_drift_features = [f for f, data in drift_scores.items() 
                                 if data.get('significant', False)]
            if high_drift_features:
                alerts.append({
                    'type': 'error',
                    'message': f"High drift detected in {len(high_drift_features)} features",
                    'severity': 'high'
                })
        
        # Cascade alerts
        if 'cascade_analysis' in monitoring_summary:
            cascade_strength = monitoring_summary['cascade_analysis'].get('cascade_strength', 0)
            if cascade_strength > 0.5:
                alerts.append({
                    'type': 'warning',
                    'message': f"Strong cascade effects detected (strength: {cascade_strength:.3f})",
                    'severity': 'medium'
                })
        
        return alerts


def create_advanced_dashboard_section():
    """Create advanced dashboard section with all visualizations."""
    st.header("🔬 Advanced Analysis")
    
    # Create tabs for different advanced features
    tab1, tab2, tab3, tab4 = st.tabs([
        "Feature Attribution", "Error Propagation", "Drift Analysis", "Alerts"
    ])
    
    viz = AdvancedVisualizations()
    
    with tab1:
        st.subheader("Feature Attribution Analysis")
        
        # Feature attribution controls
        col1, col2 = st.columns(2)
        with col1:
            attribution_method = st.selectbox(
                "Attribution Method",
                ["shap", "permutation", "basic"]
            )
        
        with col2:
            max_features = st.slider("Max Features", 5, 50, 20)
        
        # Placeholder for feature attribution plot
        st.info("Feature attribution analysis will be available when model and data are loaded")
    
    with tab2:
        st.subheader("Error Propagation Analysis")
        
        # Error propagation controls
        show_heatmap = st.checkbox("Show Error Propagation Heatmap", value=True)
        
        if show_heatmap:
            st.info("Error propagation heatmap will be available when pipeline data is loaded")
    
    with tab3:
        st.subheader("Advanced Drift Analysis")
        
        # Drift analysis controls
        col1, col2 = st.columns(2)
        with col1:
            show_drift_contribution = st.checkbox("Show Drift Contribution", value=True)
        
        with col2:
            show_saliency = st.checkbox("Show Saliency Maps", value=False)
        
        if show_drift_contribution:
            st.info("Drift contribution analysis will be available when drift data is loaded")
        
        if show_saliency:
            st.info("Saliency maps will be available for image data with PyTorch models")
    
    with tab4:
        st.subheader("Alert Dashboard")
        
        # Alert controls
        alert_threshold = st.slider("Alert Threshold", 0.1, 1.0, 0.5)
        
        st.info("Alerts will be generated based on monitoring data") 