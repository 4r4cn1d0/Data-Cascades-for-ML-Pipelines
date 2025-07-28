"""
Enhanced Streamlit dashboard with advanced features and new implementation integration.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our new implementations
from src.utils.metrics import DegradationMetrics, DriftDetectionMetrics, CascadeEffectMetrics
from src.pipeline.stages import ProductionMLPipeline
from src.pipeline.intelligent_retraining import IntelligentRetrainingManager
from src.visualization.advanced_visualizations import AdvancedVisualizations

# Import existing dashboard components
from src.visualization.mnist_dashboard import create_streamlit_dashboard
from src.pipeline.cascade_monitor import CascadeMonitor
from src.data.mnist_drift_simulator import MNISTDriftSimulator
from src.data.synthetic_data import SyntheticDataGenerator


class EnhancedPipelineDashboard:
    """Enhanced dashboard with advanced features and new implementation."""
    
    def __init__(self):
        self.advanced_viz = AdvancedVisualizations()
        self.degradation_metrics = DegradationMetrics()
        self.drift_metrics = DriftDetectionMetrics()
        self.cascade_metrics = CascadeEffectMetrics()
        self.retraining_manager = IntelligentRetrainingManager()
        
    def create_enhanced_performance_timeline(self, monitoring_history):
        """Create enhanced performance timeline with degradation analysis."""
        if not monitoring_history:
            return go.Figure()
        
        timesteps = list(range(len(monitoring_history)))
        accuracies = [h.get('accuracy', 0) for h in monitoring_history]
        
        # Calculate degradation metrics
        degradation_analysis = self.degradation_metrics.calculate_degradation_slope(accuracies)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Performance Over Time", "Degradation Analysis"),
            vertical_spacing=0.1
        )
        
        # Performance timeline
        fig.add_trace(
            go.Scatter(x=timesteps, y=accuracies, name="Accuracy", mode='lines+markers'),
            row=1, col=1
        )
        
        # Degradation analysis
        if degradation_analysis['significance']:
            trend_color = 'red' if degradation_analysis['slope'] < 0 else 'green'
            fig.add_trace(
                go.Scatter(x=timesteps, y=accuracies, name="Trend", 
                          line=dict(color=trend_color, dash='dash')),
                row=1, col=1
            )
        
        # Add degradation metrics
        fig.add_annotation(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=f"Slope: {degradation_analysis['slope']:.4f}<br>R²: {degradation_analysis['r_squared']:.4f}<br>P-value: {degradation_analysis['p_value']:.4f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title="Enhanced Performance Timeline with Degradation Analysis",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_advanced_drift_analysis(self, drift_scores_history):
        """Create advanced drift analysis with statistical significance."""
        if not drift_scores_history:
            return go.Figure()
        
        # Calculate drift statistics
        all_scores = []
        for scores in drift_scores_history:
            if isinstance(scores, dict):
                all_scores.extend(list(scores.values()))
        
        if not all_scores:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Drift Score Distribution", "Drift Trend", "Feature Drift", "Statistical Summary"),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Drift score distribution
        fig.add_trace(
            go.Histogram(x=all_scores, name="Drift Scores", nbinsx=20),
            row=1, col=1
        )
        
        # Drift trend over time
        timesteps = list(range(len(drift_scores_history)))
        avg_scores = [np.mean(list(scores.values())) if scores else 0 for scores in drift_scores_history]
        fig.add_trace(
            go.Scatter(x=timesteps, y=avg_scores, name="Avg Drift", mode='lines+markers'),
            row=1, col=2
        )
        
        # Feature drift (if available)
        if drift_scores_history and isinstance(drift_scores_history[-1], dict):
            latest_scores = drift_scores_history[-1]
            features = list(latest_scores.keys())[:10]  # Top 10 features
            scores = [latest_scores[f] for f in features]
            fig.add_trace(
                go.Bar(x=features, y=scores, name="Feature Drift"),
                row=2, col=1
            )
        
        # Statistical summary
        mean_drift = np.mean(all_scores)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=mean_drift,
                title={'text': "Mean Drift Score"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 0.3], 'color': "lightgray"},
                                {'range': [0.3, 0.7], 'color': "yellow"},
                                {'range': [0.7, 1], 'color': "red"}]},
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Advanced Drift Analysis",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_intelligent_retraining_analysis(self, retraining_triggers):
        """Create intelligent retraining analysis."""
        if not retraining_triggers:
            return go.Figure()
        
        # Analyze retraining strategies
        strategies = {}
        for trigger in retraining_triggers:
            strategy = trigger.get('strategy', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Retraining Strategy Distribution", "Retraining Timeline"),
            specs=[[{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Strategy distribution
        if strategies:
            fig.add_trace(
                go.Pie(labels=list(strategies.keys()), values=list(strategies.values())),
                row=1, col=1
            )
        
        # Retraining timeline
        timesteps = [trigger.get('timestep', i) for i, trigger in enumerate(retraining_triggers)]
        values = [1 if trigger.get('should_retrain', False) else 0 for trigger in retraining_triggers]
        fig.add_trace(
            go.Scatter(x=timesteps, y=values, mode='markers', name="Retraining Events"),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Intelligent Retraining Analysis",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_cascade_effect_analysis(self, cascade_history):
        """Create advanced cascade effect analysis."""
        if not cascade_history:
            return go.Figure()
        
        # Extract cascade scores
        cascade_scores = [c.get('avg_cascade_score', 0) for c in cascade_history]
        timesteps = list(range(len(cascade_scores)))
        
        # Calculate cascade trend
        if len(cascade_scores) > 1:
            cascade_trend = np.polyfit(timesteps, cascade_scores, 1)[0]
        else:
            cascade_trend = 0
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Cascade Effect Over Time", "Cascade Strength Analysis"),
            vertical_spacing=0.1
        )
        
        # Cascade effect timeline
        fig.add_trace(
            go.Scatter(x=timesteps, y=cascade_scores, name="Cascade Score", mode='lines+markers'),
            row=1, col=1
        )
        
        # Cascade strength analysis
        strong_cascade = [1 if score > 0.5 else 0 for score in cascade_scores]
        moderate_cascade = [1 if 0.2 <= score <= 0.5 else 0 for score in cascade_scores]
        weak_cascade = [1 if score < 0.2 else 0 for score in cascade_scores]
        
        fig.add_trace(
            go.Bar(x=timesteps, y=strong_cascade, name="Strong Cascade", marker_color='red'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=timesteps, y=moderate_cascade, name="Moderate Cascade", marker_color='yellow'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=timesteps, y=weak_cascade, name="Weak Cascade", marker_color='green'),
            row=2, col=1
        )
        
        # Add trend annotation
        if abs(cascade_trend) > 0.01:
            trend_text = "Increasing" if cascade_trend > 0 else "Decreasing"
            fig.add_annotation(
                x=0.02, y=0.98, xref='paper', yref='paper',
                text=f"Cascade Trend: {trend_text}<br>Slope: {cascade_trend:.4f}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        
        fig.update_layout(
            title="Advanced Cascade Effect Analysis",
            height=600,
            showlegend=True
        )
        
        return fig


def create_enhanced_streamlit_dashboard():
    """Create enhanced Streamlit dashboard with all new features."""
    st.title("📊 Data Cascades for ML Pipelines")
    st.markdown("**Advanced Monitoring and Cascade Effect Analysis**")
    
    # Initialize enhanced dashboard
    enhanced_dashboard = EnhancedPipelineDashboard()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Synthetic Data", "MNIST Data", "Real Pipeline"]
    )
    
    # Simulation parameters
    st.sidebar.subheader("Simulation Parameters")
    n_steps = st.sidebar.slider("Number of Steps", 10, 100, 50)
    drift_level = st.sidebar.slider("Drift Level", 0.0, 1.0, 0.3)
    
    # Advanced features toggle
    st.sidebar.subheader("Advanced Features")
    show_advanced_metrics = st.sidebar.checkbox("Advanced Metrics", value=True)
    show_retraining_analysis = st.sidebar.checkbox("Retraining Analysis", value=True)
    show_cascade_analysis = st.sidebar.checkbox("Cascade Analysis", value=True)
    
    # Main content
    if st.button("🚀 Run Enhanced Simulation", type="primary"):
        with st.spinner("Running enhanced simulation..."):
            # Initialize components
            monitor = CascadeMonitor()
            
            # Generate data based on selection
            if data_source == "Synthetic Data":
                data_gen = SyntheticDataGenerator()
                X, y = data_gen.generate_data(n_samples=1000, n_features=20)
            elif data_source == "MNIST Data":
                mnist_sim = MNISTDriftSimulator()
                # Use training data from the simulator
                X, y = mnist_sim.X_train, mnist_sim.y_train
            else:
                # Use production pipeline
                pipeline = ProductionMLPipeline()
                X, y = np.random.randn(1000, 50), np.random.randint(0, 2, 1000)
            
            # Run simulation
            monitoring_history = []
            drift_scores_history = []
            cascade_history = []
            retraining_triggers = []
            
            for step in range(n_steps):
                # Simulate drift
                drift_factor = 1 + (drift_level * step / n_steps)
                X_drifted = X * drift_factor
                
                # Calculate metrics
                accuracy = max(0.1, 0.95 - (drift_level * step / n_steps))
                
                # Update monitoring
                metrics = {
                    'accuracy': accuracy,
                    'f1': max(0.1, 0.92 - (drift_level * step / n_steps)),
                    'precision': max(0.1, 0.94 - (drift_level * step / n_steps)),
                    'recall': max(0.1, 0.90 - (drift_level * step / n_steps))
                }
                
                monitor.update_monitoring(step, metrics)
                
                # Calculate drift scores
                if step == 0:
                    monitor.set_reference_data(X)
                monitor.set_current_data(X_drifted)
                drift_scores = monitor.calculate_drift_score(X_drifted)
                
                # Store history
                monitoring_history.append(metrics)
                drift_scores_history.append(drift_scores)
                
                # Simulate cascade effects
                cascade_effect = {
                    'avg_cascade_score': drift_level * (step / n_steps),
                    'max_cascade_score': drift_level * (step / n_steps) * 1.2,
                    'timestep': step
                }
                cascade_history.append(cascade_effect)
                
                # Simulate retraining triggers
                if step % 10 == 0 and step > 0:
                    trigger = {
                        'timestep': step,
                        'type': 'performance_degradation',
                        'strategy': 'threshold_based',
                        'should_retrain': True,
                        'value': accuracy
                    }
                    retraining_triggers.append(trigger)
            
            # Store results in session state
            st.session_state.monitoring_history = monitoring_history
            st.session_state.drift_scores_history = drift_scores_history
            st.session_state.cascade_history = cascade_history
            st.session_state.retraining_triggers = retraining_triggers
            st.session_state.monitor = monitor
            st.session_state.data_source = data_source
            
            st.success("Enhanced simulation completed!")
    
    # Display results
    if hasattr(st.session_state, 'monitoring_history') and st.session_state.monitoring_history:
        st.header("📈 Enhanced Results")
        
        # Show data source info
        if hasattr(st.session_state, 'data_source'):
            st.info(f"Using: {st.session_state.data_source}")
        
        # Create enhanced tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Enhanced Performance", "Advanced Drift Analysis", "Cascade Effects", 
            "Retraining Analysis", "Advanced Metrics", "Summary"
        ])
        
        with tab1:
            st.subheader("Enhanced Performance Timeline")
            fig = enhanced_dashboard.create_enhanced_performance_timeline(st.session_state.monitoring_history)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Advanced Drift Analysis")
            fig = enhanced_dashboard.create_advanced_drift_analysis(st.session_state.drift_scores_history)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Cascade Effect Analysis")
            fig = enhanced_dashboard.create_cascade_effect_analysis(st.session_state.cascade_history)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Intelligent Retraining Analysis")
            fig = enhanced_dashboard.create_intelligent_retraining_analysis(st.session_state.retraining_triggers)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.subheader("Advanced Metrics")
            if hasattr(st.session_state, 'monitor'):
                # Show advanced metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Degradation Analysis")
                    performance_history = [h['accuracy'] for h in st.session_state.monitoring_history]
                    degradation_analysis = enhanced_dashboard.degradation_metrics.calculate_degradation_slope(performance_history)
                    
                    st.metric("Degradation Slope", f"{degradation_analysis['slope']:.4f}")
                    st.metric("R-squared", f"{degradation_analysis['r_squared']:.4f}")
                    st.metric("P-value", f"{degradation_analysis['p_value']:.4f}")
                    st.metric("Significant", degradation_analysis['significance'])
                
                with col2:
                    st.subheader("Drift Detection")
                    if st.session_state.drift_scores_history:
                        all_scores = []
                        for scores in st.session_state.drift_scores_history:
                            if isinstance(scores, dict):
                                all_scores.extend(list(scores.values()))
                        
                        if all_scores:
                            st.metric("Mean Drift Score", f"{np.mean(all_scores):.4f}")
                            st.metric("Max Drift Score", f"{np.max(all_scores):.4f}")
                            st.metric("Drift Features", f"{len([s for s in all_scores if s > 0.3])}")
                            st.metric("High Drift", f"{len([s for s in all_scores if s > 0.7])}")
        
        with tab6:
            st.subheader("Comprehensive Summary")
            if hasattr(st.session_state, 'monitor'):
                summary = st.session_state.monitor.get_monitoring_summary()
                st.json(summary)
                
                # Show alerts
                if hasattr(enhanced_dashboard, 'create_alert_panel'):
                    alerts = enhanced_dashboard.create_alert_panel(summary)
                    if alerts:
                        st.subheader("🚨 Alerts")
                        for alert in alerts:
                            if alert['type'] == 'error':
                                st.error(alert['message'])
                            elif alert['type'] == 'warning':
                                st.warning(alert['message'])
                            else:
                                st.info(alert['message'])
    
    else:
        st.info("Click 'Run Enhanced Simulation' to start monitoring the ML pipeline.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Enhanced Data Cascades for ML Pipelines** - Advanced monitoring and cascade effect analysis")


if __name__ == "__main__":
    create_enhanced_streamlit_dashboard() 