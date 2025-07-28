#!/usr/bin/env python3
"""
Data Cascades in Long-Running ML Pipelines
Main application entry point.
"""

import streamlit as st
from src.visualization.mnist_dashboard import create_streamlit_dashboard

if __name__ == "__main__":
    create_streamlit_dashboard() 