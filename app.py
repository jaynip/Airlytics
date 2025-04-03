import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Oizom's Smart Airlytics",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import utility modules
from utils.data_processor import upload_and_process_data, load_sample_data
from utils.ai_insights import get_ai_recommendations
from utils.visualization import create_correlation_heatmap, plot_time_series

# Import component modules
from components.dashboard import render_dashboard
from components.report_generator import render_report_generator
from components.map_view import render_map_view

# Import assets
from assets.icons import (
    get_banner_html, 
    get_floating_icons_html, 
    LEAF_ICON, 
    CLOUD_ICON, 
    SUN_ICON, 
    THERMOMETER_ICON,
    WATER_DROP_ICON,
    WIND_ICON,
    PULSE_ANIMATION
)

# Import Oizom logo
from assets.oizom_logo import get_oizom_logo_html

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define color palette
CHATEAU_GREEN = "#4bb051"
GRAY_NURSE = "#dee8dd"
MINE_SHAFT = "#323232"
MOSS_GREEN = "#9cd4a4"
LIGHT_GREEN = "#f0f7f0"
DARK_GREEN = "#3d9542"
ALERT_RED = "#d9534f"
WARNING_YELLOW = "#f0ad4e"
INFO_BLUE = "#5bc0de"

# Add custom styles inspired by oizom.com brand
def local_css():
    st.markdown("""
        <style>
        /* Main container styling */
        .main .block-container {padding-top: 1rem;}
        .stApp {
            background-color: #F8FAF9;
            font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
        }
        
        /* Text styling */
        h1, h2, h3 {
            color: #4bb051;
            font-weight: 600;
        }
        
        /* Panel and container styling */
        .st-emotion-cache-16txtl3 {
            padding: 1rem 1rem 1.5rem;
        }
        .st-emotion-cache-18ni7ap {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        .st-emotion-cache-18ni7ap:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }
        .st-emotion-cache-1v0mbdj {
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            background-color: #FFFFFF;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #4bb051;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #3d9542;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stButton > button:active {
            transform: scale(0.98);
        }
        
        /* Custom sidebar styling */
        .st-emotion-cache-1cypcdb {
            background: linear-gradient(180deg, rgba(75, 176, 81, 0.2) 0%, rgba(255, 255, 255, 1) 100%);
            border-right: 1px solid rgba(75, 176, 81, 0.1);
        }
        .st-emotion-cache-1wmy9hl {
            background: transparent;
        }
        
        /* Custom selection colors */
        ::selection {
            background: rgba(75, 176, 81, 0.3);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: #4bb051;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #3d9542;
        }
        
        /* Custom metrics and KPIs */
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 1.2rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            text-align: center;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #4bb051;
            margin: 0.5rem 0;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 0.5rem;
        }
        
        /* Progress bars and indicators */
        .progress-container {
            width: 100%;
            background-color: #f3f3f3;
            border-radius: 20px;
            margin: 10px 0;
        }
        .progress-bar {
            height: 10px;
            border-radius: 20px;
            transition: width 0.5s ease;
        }
        
        /* Data quality indicator */
        .data-quality-ind {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 6px;
        }
        
        /* Status indicators */
        .status-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            margin: 5px 5px 5px 0;
        }
        .good-status {
            background-color: rgba(75, 176, 81, 0.2);
            color: #4bb051;
        }
        .warning-status {
            background-color: rgba(240, 173, 78, 0.2);
            color: #f0ad4e;
        }
        .alert-status {
            background-color: rgba(217, 83, 79, 0.2);
            color: #d9534f;
        }
        
        /* Section dividers */
        .section-divider {
            border-top: 1px solid rgba(75, 176, 81, 0.2);
            margin: 2rem 0;
        }
        
        /* Chart container styling */
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'loaded_sample' not in st.session_state:
    st.session_state.loaded_sample = False
if 'ai_recommendations' not in st.session_state:
    st.session_state.ai_recommendations = None
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'selected_devices' not in st.session_state:
    st.session_state.selected_devices = []
if 'date_range' not in st.session_state:
    st.session_state.date_range = None
if 'active_view' not in st.session_state:
    st.session_state.active_view = "dashboard"
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'filter_applied' not in st.session_state:
    st.session_state.filter_applied = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# App header with Oizom logo and title
header_container = st.container()
with header_container:
    col1, col2 = st.columns([1, 5])
    
    with col1:
        st.markdown(get_oizom_logo_html(width=120), unsafe_allow_html=True)
        
    with col2:
        st.title("Oizom's Smart Airlytics")
        st.markdown("**AI-powered air quality monitoring and analysis platform**")
        
        # Add a dynamic last update indicator if data is loaded
        if st.session_state.data is not None and st.session_state.last_update:
            st.markdown(f"<p style='color: #666; font-size: 0.8rem;'>📊 Data last updated: {st.session_state.last_update}</p>", 
                      unsafe_allow_html=True)

# Navigation bar will be displayed only after data is loaded
if st.session_state.data is not None:
    st.markdown("""
    <nav class="navbar" style="
        background-color: #4bb051;
        padding: 10px 0;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-radius: 5px;
        display: flex;
        justify-content: space-around;
        align-items: center;
    ">
        <div class="nav-item" style="padding: 0 15px;">
            <a href="#" id="dashboard-link" style="
                color: white;
                text-decoration: none;
                font-weight: bold;
                font-size: 16px;
                display: flex;
                align-items: center;
            ">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-speedometer2" viewBox="0 0 16 16" style="margin-right: 5px;">
                    <path d="M8 4a.5.5 0 0 1 .5.5V6a.5.5 0 0 1-1 0V4.5A.5.5 0 0 1 8 4zM3.732 5.732a.5.5 0 0 1 .707 0l.915.914a.5.5 0 1 1-.708.708l-.914-.915a.5.5 0 0 1 0-.707z"/>
                    <path d="M2 10a.5.5 0 0 1 .5-.5h1.586a.5.5 0 0 1 0 1H2.5A.5.5 0 0 1 2 10zm9.5 0a.5.5 0 0 1 .5-.5h1.5a.5.5 0 0 1 0 1H12a.5.5 0 0 1-.5-.5zm.754-4.246a.389.389 0 0 0-.527-.02L7.547 9.31a.91.91 0 1 0 1.302 1.258l3.434-4.297a.389.389 0 0 0-.029-.518z"/>
                    <path fill-rule="evenodd" d="M0 10a8 8 0 1 1 15.547 2.661c-.442 1.253-1.845 1.602-2.932 1.25C11.309 13.488 9.475 13 8 13c-1.474 0-3.31.488-4.615.911-1.087.352-2.49.003-2.932-1.25A7.988 7.988 0 0 1 0 10zm8-7a7 7 0 0 0-6.603 9.329c.203.575.923.876 1.68.63C4.397 12.533 6.358 12 8 12s3.604.532 4.923.96c.757.245 1.477-.056 1.68-.631A7 7 0 0 0 8 3z"/>
                </svg>
                Dashboard
            </a>
        </div>
        <div class="nav-item" style="padding: 0 15px;">
            <a href="#" id="map-link" style="
                color: white;
                text-decoration: none;
                font-weight: bold;
                font-size: 16px;
                display: flex;
                align-items: center;
            ">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-geo-alt" viewBox="0 0 16 16" style="margin-right: 5px;">
                    <path d="M12.166 8.94c-.524 1.062-1.234 2.12-1.96 3.07A31.493 31.493 0 0 1 8 14.58a31.481 31.481 0 0 1-2.206-2.57c-.726-.95-1.436-2.008-1.96-3.07C3.304 7.867 3 6.862 3 6a5 5 0 0 1 10 0c0 .862-.305 1.867-.834 2.94zM8 16s6-5.686 6-10A6 6 0 0 0 2 6c0 4.314 6 10 6 10z"/>
                    <path d="M8 8a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 1a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/>
                </svg>
                Map View
            </a>
        </div>
        <div class="nav-item" style="padding: 0 15px;">
            <a href="#" id="report-link" style="
                color: white;
                text-decoration: none;
                font-weight: bold;
                font-size: 16px;
                display: flex;
                align-items: center;
            ">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-file-earmark-text" viewBox="0 0 16 16" style="margin-right: 5px;">
                    <path d="M5.5 7a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1h-5zM5 9.5a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0 2a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 0 1h-2a.5.5 0 0 1-.5-.5z"/>
                    <path d="M9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V4.5L9.5 0zm0 1v2A1.5 1.5 0 0 0 11 4.5h2V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1h5.5z"/>
                </svg>
                Reports
            </a>
        </div>
        <div class="nav-item" style="padding: 0 15px;">
            <a href="#" id="analytics-link" style="
                color: white;
                text-decoration: none;
                font-weight: bold;
                font-size: 16px;
                display: flex;
                align-items: center;
            ">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-graph-up" viewBox="0 0 16 16" style="margin-right: 5px;">
                    <path fill-rule="evenodd" d="M0 0h1v15h15v1H0V0Zm14.817 3.113a.5.5 0 0 1 .07.704l-4.5 5.5a.5.5 0 0 1-.74.037L7.06 6.767l-3.656 5.027a.5.5 0 0 1-.808-.588l4-5.5a.5.5 0 0 1 .758-.06l2.609 2.61 4.15-5.073a.5.5 0 0 1 .704-.07Z"/>
                </svg>
                Analytics
            </a>
        </div>
    </nav>

    <script>
    document.getElementById('dashboard-link').addEventListener('click', function() {
        document.querySelector('input[value="Dashboard"]').click();
    });
    document.getElementById('map-link').addEventListener('click', function() {
        document.querySelector('input[value="Map View"]').click();
    });
    document.getElementById('report-link').addEventListener('click', function() {
        document.querySelector('input[value="Report Generator"]').click();
    });
    document.getElementById('analytics-link').addEventListener('click', function() {
        document.querySelector('input[value="Dashboard"]').click();
    });
    </script>
    """, unsafe_allow_html=True)

# Error message display if exists
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    # Add a clear button
    if st.button("Clear Error"):
        st.session_state.error_message = None
        st.rerun()

# Sidebar for controls
with st.sidebar:
    # Oizom logo in sidebar
    st.markdown(get_oizom_logo_html(width=100), unsafe_allow_html=True)
    
    st.header("Controls", divider="green")
    
    # Navigation
    st.subheader("Navigation")
    view_options = ["Dashboard", "Map View", "Report Generator"]
    selected_view = st.radio("Select View", view_options, key="view_selector")
    st.session_state.active_view = selected_view.lower().replace(" ", "_")
    
    # Data import section
    st.subheader("Data Import", divider="green")
    
    upload_option = st.radio(
        "Choose data source", 
        ["Upload Files", "Use Sample Data"],
        key="data_source"
    )
    
    if upload_option == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload air quality data files (CSV/Excel)", 
            accept_multiple_files=True,
            type=['csv', 'xlsx', 'xls'],
            help="Upload your air quality data files in CSV or Excel format"
        )
        
        if uploaded_files:
            process_col1, process_col2 = st.columns([3, 1])
            with process_col1:
                process_button = st.button("Process Files", type="primary", use_container_width=True)
            with process_col2:
                file_count = len(uploaded_files)
                st.info(f"{file_count} file{'s' if file_count > 1 else ''}")
                
            if process_button:
                with st.spinner("Processing data..."):
                    try:
                        # Process uploaded files
                        df = upload_and_process_data(uploaded_files)
                        if df is not None and not df.empty:
                            st.session_state.data = df
                            st.session_state.loaded_sample = False
                            st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.success(f"✅ Successfully processed {len(uploaded_files)} files with {len(df)} records.")
                            
                            # Generate AI recommendations
                            with st.spinner("Generating AI insights..."):
                                recommendations = get_ai_recommendations(df)
                                st.session_state.ai_recommendations = recommendations
                            st.rerun()
                        else:
                            st.session_state.error_message = "❌ No valid data was extracted from the uploaded files."
                            st.rerun()
                    except Exception as e:
                        st.session_state.error_message = f"❌ Error processing files: {str(e)}"
                        logger.error(f"Error processing files: {str(e)}")
                        st.rerun()
    
    elif upload_option == "Use Sample Data":
        if st.button("Load Sample Data", type="primary", use_container_width=True, 
                   disabled=st.session_state.loaded_sample):
            with st.spinner("Loading sample data..."):
                try:
                    # Load sample data
                    df = load_sample_data()
                    if df is not None and not df.empty:
                        st.session_state.data = df
                        st.session_state.loaded_sample = True
                        st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success(f"✅ Successfully loaded sample data with {len(df)} records.")
                        
                        # Generate AI recommendations
                        with st.spinner("Generating AI insights..."):
                            recommendations = get_ai_recommendations(df)
                            st.session_state.ai_recommendations = recommendations
                        st.rerun()
                    else:
                        st.session_state.error_message = "❌ Failed to load sample data."
                        st.rerun()
                except Exception as e:
                    st.session_state.error_message = f"❌ Error loading sample data: {str(e)}"
                    logger.error(f"Error loading sample data: {str(e)}")
                    st.rerun()
        
        if st.session_state.loaded_sample:
            st.success("✅ Sample data loaded successfully")
    
    # Filter controls (only show if data is loaded)
    if st.session_state.data is not None:
        st.subheader("Data Filters", divider="green")
        
        # Date range filter
        if 'Datetime' in st.session_state.data.columns:
            df = st.session_state.data
            min_date = pd.to_datetime(df['Datetime']).min().date()
            max_date = pd.to_datetime(df['Datetime']).max().date()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                st.session_state.date_range = date_range
        
        # Device filter
        if 'Device_ID' in st.session_state.data.columns:
            available_devices = st.session_state.data['Device_ID'].unique().tolist()
            st.session_state.selected_devices = st.multiselect(
                "Select Devices",
                options=available_devices,
                default=available_devices[:min(3, len(available_devices))]
            )
        
        # Pollutant filter if applicable
        if any(col for col in st.session_state.data.columns if col.startswith(('PM', 'NO', 'CO', 'SO', 'O3'))):
            pollutant_cols = [col for col in st.session_state.data.columns 
                            if col.startswith(('PM', 'NO', 'CO', 'SO', 'O3'))]
            
            if 'selected_pollutants' not in st.session_state:
                st.session_state.selected_pollutants = pollutant_cols[:min(3, len(pollutant_cols))]
                
            st.session_state.selected_pollutants = st.multiselect(
                "Select Pollutants",
                options=pollutant_cols,
                default=st.session_state.selected_pollutants
            )
        
        # Apply filters button
        apply_filters_col1, reset_filters_col2 = st.columns([3, 1])
        with apply_filters_col1:
            if st.button("Apply Filters", type="primary", use_container_width=True):
                st.session_state.filter_applied = True
                st.rerun()
        with reset_filters_col2:
            if st.button("Reset", use_container_width=True):
                # Reset to defaults
                if 'Device_ID' in st.session_state.data.columns:
                    available_devices = st.session_state.data['Device_ID'].unique().tolist()
                    st.session_state.selected_devices = available_devices[:min(3, len(available_devices))]
                
                if 'Datetime' in st.session_state.data.columns:
                    df = st.session_state.data
                    min_date = pd.to_datetime(df['Datetime']).min().date()
                    max_date = pd.to_datetime(df['Datetime']).max().date()
                    st.session_state.date_range = (min_date, max_date)
                
                if 'selected_pollutants' in st.session_state:
                    pollutant_cols = [col for col in st.session_state.data.columns 
                                    if col.startswith(('PM', 'NO', 'CO', 'SO', 'O3'))]
                    st.session_state.selected_pollutants = pollutant_cols[:min(3, len(pollutant_cols))]
                
                st.session_state.filter_applied = False
                st.rerun()

# Main content area based on active view
if st.session_state.data is None:
    # Welcome message with enhanced styling
    st.markdown("""
    <div style="background-color: rgba(75, 176, 81, 0.1); border-left: 5px solid #4bb051; padding: 20px; border-radius: 5px; margin: 20px 0;">
        <h3 style="color: #4bb051; margin-top: 0;">Welcome to Oizom's Smart Airlytics</h3>
        <p style="margin: 10px 0; color: #333;">Please upload air quality data files or load sample data to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display app description with improved layout
    st.header("Analyze your air quality data with AI-powered insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        Oizom's Smart Airlytics is an advanced air quality analytics platform that helps you:
        
        * Process and analyze environmental data from various sensors
        * Generate AI-powered insights and recommendations
        * Visualize trends, patterns, and correlations
        * Create comprehensive reports for stakeholders
        
        Get started by uploading your air quality data files or using our sample dataset.
        """)
    
    with col2:
        # Create a cleaner banner with simple SVG icons instead of HTML divs
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(120deg, rgba(75, 176, 81, 0.1) 0%, rgba(255, 255, 255, 0.8) 100%); border-radius: 10px;">
            <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
                <img src="https://img.icons8.com/ios/50/4bb051/leaf.png" alt="Leaf" width="32">
                <img src="https://img.icons8.com/ios/50/4bb051/cloud.png" alt="Cloud" width="32">
                <img src="https://img.icons8.com/ios/50/4bb051/sun.png" alt="Sun" width="32">
                <img src="https://img.icons8.com/ios/50/4bb051/temperature.png" alt="Temperature" width="32">
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature rows with enhanced styling
    st.subheader("Key Features", divider="green")
    
    feature_row1 = st.columns(3)
    
    with feature_row1[0]:
        st.markdown("""
        <div class="metric-card">
            <img src="https://img.icons8.com/ios/50/4bb051/leaf.png" alt="Leaf" width="32">
            <h3>Data Processing</h3>
            <p>Advanced data cleaning and transformation capabilities for accurate analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_row1[1]:
        st.markdown("""
        <div class="metric-card">
            <img src="https://img.icons8.com/ios/50/4bb051/cloud.png" alt="Cloud" width="32">
            <h3>Interactive Dashboards</h3>
            <p>Dynamic visualizations with time-series analysis, correlations, and patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_row1[2]:
        st.markdown("""
        <div class="metric-card">
            <img src="https://img.icons8.com/ios/50/4bb051/sun.png" alt="Sun" width="32">
            <h3>AI-Powered Insights</h3>
            <p>Advanced analytics with predictive modeling and anomaly detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Detailed features section
    st.subheader("Comprehensive Analysis Tools")
    
    feature_row2 = st.columns(3)
    
    with feature_row2[0]:
        st.markdown("""
        <div class="chart-container">
            <h4>Data Analysis</h4>
            <ul>
                <li>Time-series visualization</li>
                <li>Correlation analysis</li>
                <li>Statistical summaries</li>
                <li>Data quality assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_row2[1]:
        st.markdown("""
        <div class="chart-container">
            <h4>AI Insights</h4>
            <ul>
                <li>Pattern recognition</li>
                <li>Anomaly detection</li>
                <li>Trend predictions</li>
                <li>Seasonal analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_row2[2]:
        st.markdown("""
        <div class="chart-container">
            <h4>Geospatial View</h4>
            <ul>
                <li>Location-based analysis</li>
                <li>Pollution hotspots</li>
                <li>Spatial correlations</li>
                <li>Area coverage assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # How to get started
    st.subheader("How to Get Started", divider="green")
    
    start_col1, start_col2 = st.columns([2, 1])
    
    with start_col1:
        st.markdown("""
        1. **Import Data**: Upload your air quality CSV/Excel files or use our sample dataset
        2. **Explore the Dashboard**: Analyze trends, patterns, and correlations in your data
        3. **View on Map**: See geographical distribution of air quality measurements
        4. **Generate Reports**: Create comprehensive reports for stakeholders
        """)
    
    with start_col2:
        st.info("Need help? Check out the sample data option in the sidebar to see the platform in action.")

else:
    # Apply filters if set
    filtered_df = st.session_state.data.copy()
    
    # Apply date filter
    if st.session_state.date_range and len(st.session_state.date_range) == 2:
        start_date, end_date = st.session_state.date_range
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df['Datetime']).dt.date >= start_date) & 
            (pd.to_datetime(filtered_df['Datetime']).dt.date <= end_date)
        ]
    
    # Apply device filter
    if st.session_state.selected_devices:
        filtered_df = filtered_df[filtered_df['Device_ID'].isin(st.session_state.selected_devices)]
    
    # Display different views based on selection
    if st.session_state.active_view == "dashboard":
        render_dashboard(filtered_df, st.session_state.ai_recommendations)
    elif st.session_state.active_view == "map_view":
        render_map_view(filtered_df)
    elif st.session_state.active_view == "report_generator":
        render_report_generator(filtered_df, st.session_state.ai_recommendations)
