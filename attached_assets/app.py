import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Import utility modules
from utils.data_processor import upload_and_process_data, load_sample_data
from utils.ai_insights import get_ai_recommendations
from utils.visualization import create_correlation_heatmap, plot_time_series

# Import component modules
from components.dashboard import render_dashboard
from components.report_generator import generate_report
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
    FLOATING_ANIMATION,
    PULSE_ANIMATION
)

# Import Oizom logo
from assets.oizom_logo import get_oizom_logo_html

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define color palette
CHATEAU_GREEN = "#4bb051"
GRAY_NURSE = "#dee8dd"
MINE_SHAFT = "#323232"
MOSS_GREEN = "#9cd4a4"

# Page configuration
st.set_page_config(
    page_title="Oizom's Smart Airlytics",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom styles inspired by oizom.com brand and aqi.in UI
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
        /* No animations, as requested */
        
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

# App header with Oizom logo and title
header_container = st.container()
with header_container:
    col1, col2 = st.columns([1, 5])
    
    with col1:
        # Logo - use HTML here because it's SVG logo
        st.markdown(get_oizom_logo_html(width=120), unsafe_allow_html=True)
        
    with col2:
        # Title and subtitle - using native Streamlit components
        st.title("Oizom's Smart Airlytics")
        st.markdown("**AI-powered air quality monitoring and analysis platform**")

# Sidebar for controls
with st.sidebar:
    # Oizom logo in sidebar (static, not floating)
    st.markdown(get_oizom_logo_html(width=100), unsafe_allow_html=True)
    
    st.header("Controls")
    
    # Navigation
    st.subheader("Navigation")
    view_options = ["Dashboard", "Map View", "Report Generator"]
    selected_view = st.radio("Select View", view_options, key="view_selector")
    st.session_state.active_view = selected_view.lower().replace(" ", "_")
    
    # Data upload section
    st.subheader("Data Import")
    
    upload_option = st.radio(
        "Choose data source", 
        ["Upload Files", "Use Sample Data"],
        key="data_source"
    )
    
    if upload_option == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload air quality data files (CSV/Excel)", 
            accept_multiple_files=True,
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_files and st.button("Process Files"):
            with st.spinner("Processing data..."):
                try:
                    # Process uploaded files
                    df = upload_and_process_data(uploaded_files)
                    if df is not None and not df.empty:
                        st.session_state.data = df
                        st.session_state.loaded_sample = False
                        st.success(f"✅ Successfully processed {len(uploaded_files)} files with {len(df)} records.")
                        
                        # Generate AI recommendations
                        with st.spinner("Generating AI insights..."):
                            recommendations = get_ai_recommendations(df)
                            st.session_state.ai_recommendations = recommendations
                    else:
                        st.error("❌ No valid data was extracted from the uploaded files.")
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
                    logger.error(f"Error processing files: {str(e)}")
    
    elif upload_option == "Use Sample Data" and not st.session_state.loaded_sample:
        if st.button("Load Sample Data"):
            with st.spinner("Loading sample data..."):
                try:
                    # Load sample data
                    df = load_sample_data()
                    if df is not None and not df.empty:
                        st.session_state.data = df
                        st.session_state.loaded_sample = True
                        st.success(f"✅ Successfully loaded sample data with {len(df)} records.")
                        
                        # Generate AI recommendations
                        with st.spinner("Generating AI insights..."):
                            recommendations = get_ai_recommendations(df)
                            st.session_state.ai_recommendations = recommendations
                    else:
                        st.error("❌ Failed to load sample data.")
                except Exception as e:
                    st.error(f"❌ Error loading sample data: {str(e)}")
                    logger.error(f"Error loading sample data: {str(e)}")
    
    # Filter controls (only show if data is loaded)
    if st.session_state.data is not None:
        st.subheader("Filters")
        
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

# Main content area based on active view
if st.session_state.data is None:
    # Info message with custom styling
    st.markdown("""
    <div style="background-color: rgba(75, 176, 81, 0.1); border-left: 5px solid #4bb051; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <p style="margin: 0; color: #333;">Please upload air quality data files or load sample data to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display app description with Streamlit native components
    st.header("About Oizom's Smart Airlytics")
    
    st.write("Oizom's Smart Airlytics is an AI-powered air quality analytics platform that processes environmental data from sensors to provide actionable insights for better air quality management.")
    
    # Feature rows
    feature_row1 = st.columns(2)
    
    with feature_row1[0]:
        st.markdown("### Data Processing")
        st.info("Handles multiple file formats and performs advanced data cleaning for accurate analysis")
    
    with feature_row1[1]:
        st.markdown("### Interactive Dashboards")
        st.info("View time-series, correlations, and geospatial patterns with intuitive visualizations")
    
    feature_row2 = st.columns(2)
    
    with feature_row2[0]:
        st.markdown("### AI-Powered Insights")
        st.info("Advanced analytics using AI to provide predictive modeling and pattern recognition")
    
    with feature_row2[1]:
        st.markdown("### Comprehensive Reporting")
        st.info("Generate detailed reports for stakeholders with actionable recommendations")
    
    st.markdown("##### To get started, upload your air quality data files or use our sample dataset.")
    
    # Display feature cards using Streamlit native components
    st.subheader("Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.subheader("Data Analysis")
            st.write("• Time-series visualization")
            st.write("• Correlation analysis")
            st.write("• Statistical summaries")
    
    with col2:
        with st.container():
            st.subheader("AI Insights")
            st.write("• Pattern recognition")
            st.write("• Anomaly detection")
            st.write("• Trend predictions")
    
    with col3:
        with st.container():
            st.subheader("Geospatial View")
            st.write("• Location-based analysis")
            st.write("• Pollution hotspots")
            st.write("• Spatial correlations")
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
        generate_report(filtered_df, st.session_state.ai_recommendations)

# Footer with Oizom branding using native Streamlit components
st.markdown("---")

# Use container to keep footer content centered and clean
footer_container = st.container()
with footer_container:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Center align content
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        # Add small Oizom logo
        st.markdown(get_oizom_logo_html(width=80), unsafe_allow_html=True)
        st.caption(f"© {datetime.now().year} Oizom's Smart Airlytics • Powered by Streamlit and Google Gemini")
        st.markdown("</div>", unsafe_allow_html=True)
