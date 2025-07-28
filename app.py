#!/usr/bin/env python3
"""
Main application entry point for Data Cascades for ML Pipelines.
Enhanced version with all critical gaps implemented.
"""

import sys
import os
import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Data Cascades for ML Pipelines",
    page_icon="📊",
    layout="wide"
)

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the enhanced dashboard
from src.visualization.dashboard import create_enhanced_streamlit_dashboard

if __name__ == "__main__":
    create_enhanced_streamlit_dashboard() 