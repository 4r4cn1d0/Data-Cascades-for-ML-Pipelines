"""
Real-time dashboard for ML pipeline monitoring and cascade effects visualization.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # Temporarily commented out due to compatibility issue
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class PipelineDashboard:
    """Real-time dashboard for ML pipeline monitoring."""
    
    def __init__(self):
        self.figures = {}
        self.data_cache = {}
        
    def create_performance_timeline(self, monitoring_history):
        """Create performance timeline chart."""
        if not monitoring_history:
            return go.Figure()
        
        df = pd.DataFrame(monitoring_history)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Over Time', 'F1 Score Over Time', 
                          'Precision Over Time', 'Recall Over Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, pos in zip(metrics, positions):
            if metric in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['time_step'],
                        y=df[metric],
                        mode='lines+markers',
                        name=metric.capitalize(),
                        line=dict(width=2),
                        marker=dict(size=6)
                    ),
                    row=pos[0], col=pos[1]
                )
        
        fig.update_layout(
            height=600,
            title_text="Pipeline Performance Timeline",
            showlegend=True
        )
        
        return fig
    
    def create_drift_heatmap(self, drift_scores_history):
        """Create drift detection heatmap."""
        if not drift_scores_history:
            return go.Figure()
        
        # Prepare data for heatmap
        time_steps = []
        features = []
        drift_values = []
        
        for t, drift_scores in enumerate(drift_scores_history):
            for feature, score in drift_scores.items():
                time_steps.append(t)
                features.append(feature)
                drift_values.append(score)
        
        if not drift_values:
            return go.Figure()
        
        # Create pivot table
        df = pd.DataFrame({
            'time_step': time_steps,
            'feature': features,
            'drift_score': drift_values
        })
        
        pivot_df = df.pivot(index='feature', columns='time_step', values='drift_score')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Reds',
            colorbar=dict(title="Drift Score")
        ))
        
        fig.update_layout(
            title_text="Data Drift Heatmap",
            xaxis_title="Time Step",
            yaxis_title="Features",
            height=500
        )
        
        return fig
    
    def create_cascade_network(self, cascade_history):
        """Create cascade effect network visualization."""
        if not cascade_history:
            return go.Figure()
        
        # Extract cascade correlations
        cascade_data = []
        for entry in cascade_history:
            if 'cascade_correlations' in entry:
                for correlation_name, correlation_value in entry['cascade_correlations'].items():
                    cascade_data.append({
                        'time_step': entry['time_step'],
                        'correlation': correlation_name,
                        'value': correlation_value
                    })
        
        if not cascade_data:
            return go.Figure()
        
        df = pd.DataFrame(cascade_data)
        
        fig = go.Figure()
        
        # Plot cascade correlations over time
        for correlation in df['correlation'].unique():
            correlation_data = df[df['correlation'] == correlation]
            fig.add_trace(go.Scatter(
                x=correlation_data['time_step'],
                y=correlation_data['value'],
                mode='lines+markers',
                name=correlation,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title_text="Cascade Effect Correlations",
            xaxis_title="Time Step",
            yaxis_title="Correlation Coefficient",
            height=400
        )
        
        return fig
    
    def create_retraining_triggers_chart(self, retraining_triggers):
        """Create retraining triggers visualization."""
        if not retraining_triggers:
            return go.Figure()
        
        df = pd.DataFrame(retraining_triggers)
        
        # Count triggers by type
        trigger_counts = df['type'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=trigger_counts.index,
                y=trigger_counts.values,
                marker_color=['red', 'orange', 'yellow', 'green']
            )
        ])
        
        fig.update_layout(
            title_text="Retraining Triggers by Type",
            xaxis_title="Trigger Type",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    def create_stage_performance_comparison(self, stage_performances):
        """Create stage-wise performance comparison."""
        if not stage_performances:
            return go.Figure()
        
        # Prepare data
        stages = list(stage_performances.keys())
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [stage_performances[stage].get(metric, 0) for stage in stages]
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=stages,
                y=values
            ))
        
        fig.update_layout(
            title_text="Stage-wise Performance Comparison",
            xaxis_title="Pipeline Stages",
            yaxis_title="Performance Score",
            barmode='group',
            height=500
        )
        
        return fig
    
    def create_drift_timeline(self, drift_history):
        """Create drift timeline visualization."""
        if not drift_history:
            return go.Figure()
        
        df = pd.DataFrame(drift_history)
        
        fig = go.Figure()
        
        # Plot drift factor over time
        if 'drift_factor' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['time_step'],
                y=df['drift_factor'],
                mode='lines+markers',
                name='Drift Factor',
                line=dict(color='red', width=2)
            ))
        
        # Add threshold line
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                     annotation_text="Drift Threshold")
        
        fig.update_layout(
            title_text="Data Drift Timeline",
            xaxis_title="Time Step",
            yaxis_title="Drift Factor",
            height=400
        )
        
        return fig
    
    def create_cascade_summary(self, cascade_summary):
        """Create cascade effect summary."""
        if not cascade_summary or cascade_summary == "No cascade effects recorded.":
            return go.Figure()
        
        # Create summary metrics
        metrics = ['total_predictions', 'affected_predictions']
        values = [cascade_summary.get(metric, 0) for metric in metrics]
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                marker_color=['blue', 'red']
            )
        ])
        
        fig.update_layout(
            title_text="Cascade Effect Summary",
            xaxis_title="Metric",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    def create_alert_panel(self, monitoring_summary):
        """Create alert panel for critical issues."""
        alerts = []
        
        if monitoring_summary and isinstance(monitoring_summary, dict):
            # Performance alerts
            if 'performance_trends' in monitoring_summary:
                trends = monitoring_summary['performance_trends']
                if trends.get('accuracy_trend', 0) < -0.1:
                    alerts.append({
                        'type': 'warning',
                        'message': f"Accuracy declining: {trends.get('current_accuracy', 0):.3f}"
                    })
                
                if trends.get('f1_trend', 0) < -0.1:
                    alerts.append({
                        'type': 'warning',
                        'message': f"F1 score declining: {trends.get('current_f1', 0):.3f}"
                    })
            
            # Cascade alerts
            if 'cascade_analysis' in monitoring_summary:
                cascade = monitoring_summary['cascade_analysis']
                if cascade.get('avg_cascade_score', 0) > 0.5:
                    alerts.append({
                        'type': 'error',
                        'message': f"High cascade effects: {cascade.get('avg_cascade_score', 0):.3f}"
                    })
            
            # Retraining alerts
            if monitoring_summary.get('retraining_triggers', 0) > 5:
                alerts.append({
                    'type': 'info',
                    'message': f"Multiple retraining triggers: {monitoring_summary['retraining_triggers']}"
                })
        
        return alerts
    
    def create_metrics_cards(self, current_metrics):
        """Create metric cards for current performance."""
        if not current_metrics:
            return []
        
        cards = []
        for metric, value in current_metrics.items():
            if isinstance(value, (int, float)):
                color = 'green' if value > 0.8 else 'orange' if value > 0.6 else 'red'
                cards.append({
                    'metric': metric.capitalize(),
                    'value': f"{value:.3f}",
                    'color': color
                })
        
        return cards


def create_streamlit_dashboard():
    """Create the main Streamlit dashboard."""
    st.set_page_config(
        page_title="ML Pipeline Cascade Monitor",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 ML Pipeline Cascade Monitor")
    st.markdown("Real-time monitoring of data cascades in long-running ML pipelines using **real MNIST data**")
    
    # Initialize dashboard
    dashboard = PipelineDashboard()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Real MNIST Data", "Synthetic Data"],
        help="Choose between real MNIST data from Kaggle or synthetic data"
    )
    
    # Simulation parameters
    st.sidebar.subheader("Simulation Parameters")
    n_time_steps = st.sidebar.slider("Time Steps", 10, 100, 50)
    drift_strength = st.sidebar.slider("Drift Strength", 0.0, 1.0, 0.1)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
    
    # MNIST-specific parameters
    if data_source == "Real MNIST Data":
        st.sidebar.subheader("MNIST Parameters")
        drift_types = st.sidebar.multiselect(
            "Drift Types",
            ["gradual", "noise", "blur"],
            default=["gradual", "noise"],
            help="Types of drift to apply to MNIST data"
        )
        n_samples_per_step = st.sidebar.slider("Samples per Time Step", 50, 200, 100)
    
    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation"):
        with st.spinner("Running pipeline simulation..."):
            # Import and run simulation
            from src.pipeline.stages import MLPipeline
            from src.pipeline.cascade_monitor import CascadeMonitor
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            if data_source == "Real MNIST Data":
                # Use real MNIST data
                from src.data.real_data import get_mnist_data_for_pipeline
                
                # Get MNIST data with drift
                drifted_data, drifted_targets, feature_names = get_mnist_data_for_pipeline(
                    n_samples_per_step=n_samples_per_step, 
                    n_time_steps=n_time_steps
                )
                
                # Create pipeline for MNIST (10 classes)
                pipeline = MLPipeline(n_features=len(feature_names), n_classes=10)
                
                # Use first time step for training
                X_train = drifted_data[0]
                y_train = drifted_targets[0]
                pipeline.fit(X_train, y_train)
                
            else:
                # Use synthetic data
                from src.data.synthetic_data import SyntheticDataGenerator
                
                # Generate data
                generator = SyntheticDataGenerator(n_samples=1000, n_features=10, n_classes=3)
                X, y = generator.generate_base_data()
                
                # Create pipeline
                pipeline = MLPipeline()
                pipeline.fit(X, y)
                
                # Simulate drifted data
                drifted_data = []
                drifted_targets = []
                for t in range(n_time_steps):
                    drifted_X = X + np.random.normal(0, t * drift_strength / n_time_steps, X.shape)
                    drifted_X = drifted_X + np.random.normal(0, noise_level, drifted_X.shape)
                    drifted_data.append(drifted_X)
                    drifted_targets.append(y)
            
            # Create monitor
            monitor = CascadeMonitor()
            if data_source == "Real MNIST Data":
                monitor.set_reference_data(pd.DataFrame(drifted_data[0], columns=feature_names))
            else:
                monitor.set_reference_data(X)
            
            # Run simulation
            monitoring_history = []
            drift_scores_history = []
            cascade_history = []
            retraining_triggers = []
            
            progress_bar = st.progress(0)
            
            for t in range(n_time_steps):
                # Get current data
                if data_source == "Real MNIST Data":
                    current_data = pd.DataFrame(drifted_data[t], columns=feature_names)
                    current_targets = drifted_targets[t]
                else:
                    current_data = drifted_data[t]
                    current_targets = drifted_targets[t]
                
                monitor.set_current_data(current_data)
                
                # Get predictions
                predictions, probabilities = pipeline.predict(current_data, time_step=t)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(current_targets, predictions),
                    'f1': f1_score(current_targets, predictions, average='weighted'),
                    'precision': precision_score(current_targets, predictions, average='weighted'),
                    'recall': recall_score(current_targets, predictions, average='weighted')
                }
                
                # Update monitoring
                monitoring_entry = monitor.update_monitoring(t, metrics)
                monitoring_history.append(monitoring_entry)
                
                if 'drift_scores' in monitoring_entry:
                    drift_scores_history.append(monitoring_entry['drift_scores'])
                
                if 'cascade_analysis' in monitoring_entry:
                    cascade_history.append(monitoring_entry['cascade_analysis'])
                
                retraining_triggers.extend(monitoring_entry.get('retraining_triggers', []))
                
                # Update progress
                progress_bar.progress((t + 1) / n_time_steps)
            
            # Store results in session state
            st.session_state.monitoring_history = monitoring_history
            st.session_state.drift_scores_history = drift_scores_history
            st.session_state.cascade_history = cascade_history
            st.session_state.retraining_triggers = retraining_triggers
            st.session_state.monitor = monitor
            st.session_state.data_source = data_source
            
            st.success("Simulation completed!")
    
    # Display results if available
    if hasattr(st.session_state, 'monitoring_history') and st.session_state.monitoring_history:
        st.header("📈 Results")
        
        # Show data source info
        if hasattr(st.session_state, 'data_source'):
            st.info(f"Using: {st.session_state.data_source}")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Performance Timeline", "Drift Analysis", "Cascade Effects", 
            "Retraining Triggers", "Summary"
        ])
        
        with tab1:
            st.subheader("Pipeline Performance Over Time")
            fig = dashboard.create_performance_timeline(st.session_state.monitoring_history)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Data Drift Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = dashboard.create_drift_heatmap(st.session_state.drift_scores_history)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if hasattr(st.session_state, 'monitor'):
                    try:
                        drift_summary = st.session_state.monitor.get_drift_summary()
                        if isinstance(drift_summary, dict):
                            st.json(drift_summary)
                        else:
                            st.info(drift_summary)
                    except Exception as e:
                        st.warning(f"Drift summary not available: {e}")
        
        with tab3:
            st.subheader("Cascade Effects")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = dashboard.create_cascade_network(st.session_state.cascade_history)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if hasattr(st.session_state, 'monitor'):
                    cascade_summary = st.session_state.monitor.get_monitoring_summary()
                    if 'cascade_analysis' in cascade_summary:
                        st.metric("Avg Cascade Score", f"{cascade_summary['cascade_analysis']['avg_cascade_score']:.3f}")
                        st.metric("Max Cascade Score", f"{cascade_summary['cascade_analysis']['max_cascade_score']:.3f}")
        
        with tab4:
            st.subheader("Retraining Triggers")
            fig = dashboard.create_retraining_triggers_chart(st.session_state.retraining_triggers)
            st.plotly_chart(fig, use_container_width=True)
            
            if st.session_state.retraining_triggers:
                st.write("Recent triggers:")
                for trigger in st.session_state.retraining_triggers[-5:]:
                    st.write(f"- {trigger['type']}: {trigger.get('metric', 'N/A')} = {trigger.get('value', 'N/A')}")
        
        with tab5:
            st.subheader("Monitoring Summary")
            if hasattr(st.session_state, 'monitor'):
                summary = st.session_state.monitor.get_monitoring_summary()
                st.json(summary)
                
                # Alerts
                alerts = dashboard.create_alert_panel(summary)
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
        st.info("Click 'Run Simulation' to start monitoring the ML pipeline.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Cascades for ML Pipelines** - Simulating and monitoring cascade effects in production ML systems using real MNIST data")


if __name__ == "__main__":
    create_streamlit_dashboard() 