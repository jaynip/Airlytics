import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import base64
import io
from utils.visualization import (
    create_correlation_heatmap,
    plot_time_series,
    create_aqi_distribution_chart,
    create_pollutant_comparison,
    create_daily_pattern_chart,
    create_weekly_pattern_chart,
    create_forecast_chart
)

# Set up logging
logger = logging.getLogger(__name__)

def render_report_generator(df, ai_recommendations=None):
    """
    Render the report generator view allowing users to create and export reports.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    ai_recommendations : dict, optional
        AI-generated insights and recommendations
    """
    try:
        if df is None or df.empty:
            st.error("No data available for report generation. Please upload data or use sample data.")
            return
        
        st.header("Air Quality Report Generator", divider="green")
        
        # Report settings sidebar
        report_options = st.container()
        with report_options:
            st.subheader("Report Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Report title and description
                report_title = st.text_input(
                    "Report Title", 
                    value="Air Quality Analysis Report",
                    help="Enter a title for your report"
                )
                
                report_description = st.text_area(
                    "Report Description",
                    value="This report provides an analysis of air quality data collected from monitoring stations.",
                    help="Enter a brief description of your report"
                )
                
                # Report author and organization
                report_author = st.text_input(
                    "Author Name (optional)",
                    value="",
                    help="Enter the name of the report author"
                )
                
                report_org = st.text_input(
                    "Organization (optional)",
                    value="",
                    help="Enter your organization name"
                )
            
            with col2:
                # Report sections to include
                st.markdown("##### Report Sections")
                
                include_summary = st.checkbox("Executive Summary", value=True)
                include_trends = st.checkbox("Trend Analysis", value=True)
                include_pollutants = st.checkbox("Pollutant Details", value=True)
                include_recommendations = st.checkbox("Recommendations", value=True, 
                                                  disabled=ai_recommendations is None)
                include_methodology = st.checkbox("Methodology", value=False)
                
                # Chart and visualization options
                st.markdown("##### Visualizations")
                
                include_overview_charts = st.checkbox("Overview Charts", value=True)
                include_time_series = st.checkbox("Time Series Charts", value=True)
                include_correlations = st.checkbox("Correlation Analysis", value=True)
                include_forecasts = st.checkbox("Forecasts", value=True, 
                                             disabled=ai_recommendations is None or 'forecasts' not in ai_recommendations)
        
        # Report preview
        preview_container = st.container()
        with preview_container:
            st.subheader("Report Preview", divider="green")
            
            # Title and metadata
            report_date = datetime.now().strftime("%Y-%m-%d")
            
            st.markdown(f"# {report_title}")
            
            metadata = []
            metadata.append(f"**Date:** {report_date}")
            if report_author:
                metadata.append(f"**Author:** {report_author}")
            if report_org:
                metadata.append(f"**Organization:** {report_org}")
            
            st.markdown(" | ".join(metadata))
            
            st.markdown("---")
            
            # Description
            if report_description:
                st.markdown(report_description)
                st.markdown("---")
            
            # Executive Summary
            if include_summary:
                st.markdown("## Executive Summary")
                
                # Generate summary based on data
                summary_text = generate_executive_summary(df, ai_recommendations)
                st.markdown(summary_text)
                
                # Key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'AQI' in df.columns:
                        avg_aqi = df['AQI'].mean()
                        aqi_color = get_aqi_color(avg_aqi)
                        st.markdown(
                            f"<div style='text-align: center; padding: 10px; background-color: rgba({','.join(map(str, px.colors.hex_to_rgb(aqi_color)))}, 0.2); "
                            f"border: 1px solid {aqi_color}; border-radius: 5px;'>"
                            f"<h3 style='margin: 0; color: {aqi_color};'>{avg_aqi:.1f}</h3>"
                            f"<p style='margin: 0;'>Average AQI</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                
                with col2:
                    if 'Datetime' in df.columns:
                        date_range = f"{pd.to_datetime(df['Datetime']).min().strftime('%Y-%m-%d')} to {pd.to_datetime(df['Datetime']).max().strftime('%Y-%m-%d')}"
                        days = (pd.to_datetime(df['Datetime']).max() - pd.to_datetime(df['Datetime']).min()).days + 1
                        
                        st.markdown(
                            f"<div style='text-align: center; padding: 10px; background-color: rgba(240, 240, 240, 0.3); "
                            f"border: 1px solid #ddd; border-radius: 5px;'>"
                            f"<h3 style='margin: 0;'>{days}</h3>"
                            f"<p style='margin: 0;'>Days Analyzed</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                
                with col3:
                    records = len(df)
                    
                    st.markdown(
                        f"<div style='text-align: center; padding: 10px; background-color: rgba(240, 240, 240, 0.3); "
                        f"border: 1px solid #ddd; border-radius: 5px;'>"
                        f"<h3 style='margin: 0;'>{records:,}</h3>"
                        f"<p style='margin: 0;'>Data Points</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                st.markdown("---")
            
            # Overview Charts
            if include_overview_charts:
                st.markdown("## Overview")
                
                if 'AQI' in df.columns:
                    # AQI Distribution
                    aqi_dist_fig = create_aqi_distribution_chart(df)
                    if aqi_dist_fig:
                        st.plotly_chart(aqi_dist_fig, use_container_width=True)
                
                # Pollutant comparison if multiple devices
                if 'Device_ID' in df.columns and df['Device_ID'].nunique() > 1:
                    pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
                    if pollutant_cols:
                        comp_fig = create_pollutant_comparison(df, pollutants=pollutant_cols[:3])
                        if comp_fig:
                            st.plotly_chart(comp_fig, use_container_width=True)
                
                st.markdown("---")
            
            # Trend Analysis
            if include_trends:
                st.markdown("## Trend Analysis")
                
                if 'Datetime' in df.columns:
                    # Time series graphs
                    if include_time_series:
                        if 'AQI' in df.columns:
                            ts_param = 'AQI'
                        elif 'PM2.5' in df.columns:
                            ts_param = 'PM2.5'
                        else:
                            pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
                            ts_param = next((p for p in pollutant_cols), None)
                        
                        if ts_param:
                            ts_fig = plot_time_series(df, parameters=[ts_param], rolling_window=24, add_trend=True)
                            if ts_fig:
                                st.plotly_chart(ts_fig, use_container_width=True)
                    
                    # Daily and weekly patterns
                    main_param = 'AQI' if 'AQI' in df.columns else ('PM2.5' if 'PM2.5' in df.columns else next(iter([col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]), None))
                    
                    if main_param:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            daily_fig = create_daily_pattern_chart(df, main_param)
                            if daily_fig:
                                st.plotly_chart(daily_fig, use_container_width=True)
                        
                        with col2:
                            weekly_fig = create_weekly_pattern_chart(df, main_param)
                            if weekly_fig:
                                st.plotly_chart(weekly_fig, use_container_width=True)
                    
                    # Correlations
                    if include_correlations:
                        correlation_cols = [col for col in df.columns if col in 
                                         ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'AQI', 
                                           'Temperature', 'Humidity', 'Pressure']]
                        
                        if len(correlation_cols) >= 2:
                            st.markdown("### Parameter Correlations")
                            corr_fig = create_correlation_heatmap(df, selected_columns=correlation_cols)
                            if corr_fig:
                                st.plotly_chart(corr_fig, use_container_width=True)
                            
                            # Add correlation insights
                            if (ai_recommendations and 'correlations' in ai_recommendations and 
                                'insights' in ai_recommendations['correlations'] and 
                                ai_recommendations['correlations']['insights']):
                                
                                st.markdown("#### Key Correlation Findings:")
                                for insight in ai_recommendations['correlations']['insights']:
                                    st.markdown(f"- {insight}")
                    
                    # Forecasts
                    if include_forecasts and ai_recommendations and 'forecasts' in ai_recommendations:
                        st.markdown("### Forecasts")
                        
                        available_forecasts = [p for p in ['AQI', 'PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2'] 
                                           if p in ai_recommendations['forecasts']]
                        
                        if available_forecasts:
                            forecast_param = available_forecasts[0]  # Use first available forecast
                            forecast_fig = create_forecast_chart(df, ai_recommendations['forecasts'], forecast_param)
                            
                            if forecast_fig:
                                st.plotly_chart(forecast_fig, use_container_width=True)
                
                st.markdown("---")
            
            # Pollutant Details
            if include_pollutants:
                st.markdown("## Pollutant Analysis")
                
                pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
                
                for pollutant in pollutant_cols:
                    st.markdown(f"### {pollutant}")
                    
                    # Pollutant statistics
                    avg_value = df[pollutant].mean()
                    max_value = df[pollutant].max()
                    min_value = df[pollutant].min()
                    
                    # Get threshold and exceedance rate
                    thresholds = {
                        'PM2.5': 35.0,  # μg/m³
                        'PM10': 50.0,   # μg/m³
                        'NO2': 100.0,   # ppb
                        'O3': 70.0,     # ppb
                        'CO': 9.0,      # ppm
                        'SO2': 75.0     # ppb
                    }
                    
                    threshold = thresholds.get(pollutant, 0)
                    exceedance_rate = (df[pollutant] > threshold).mean() * 100 if threshold > 0 else 0
                    
                    # Units for each pollutant
                    units = {
                        'PM2.5': 'μg/m³',
                        'PM10': 'μg/m³',
                        'NO2': 'ppb',
                        'O3': 'ppb',
                        'CO': 'ppm',
                        'SO2': 'ppb'
                    }
                    
                    unit = units.get(pollutant, '')
                    
                    # Display metrics in a nice grid
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Average", f"{avg_value:.2f} {unit}")
                    
                    with col2:
                        st.metric("Maximum", f"{max_value:.2f} {unit}")
                    
                    with col3:
                        st.metric("Minimum", f"{min_value:.2f} {unit}")
                    
                    with col4:
                        if threshold > 0:
                            st.metric("Threshold Exceedance", f"{exceedance_rate:.1f}%")
                    
                    # Time series for this pollutant
                    if 'Datetime' in df.columns:
                        ts_fig = plot_time_series(df, parameters=[pollutant], rolling_window=24)
                        if ts_fig:
                            st.plotly_chart(ts_fig, use_container_width=True)
                    
                    # Health impact info if available
                    if (ai_recommendations and 'health_impact' in ai_recommendations and 
                        'pollutant_impacts' in ai_recommendations['health_impact'] and
                        pollutant in ai_recommendations['health_impact']['pollutant_impacts']):
                        
                        health_info = ai_recommendations['health_impact']['pollutant_impacts'][pollutant]
                        
                        if 'health_effects' in health_info and health_info['health_effects']:
                            st.markdown("#### Health Effects")
                            for effect in health_info['health_effects']:
                                st.markdown(f"- {effect}")
                    
                    st.markdown("---")
            
            # Recommendations
            if include_recommendations and ai_recommendations and 'recommendations' in ai_recommendations:
                st.markdown("## Recommendations")
                
                recommendations = ai_recommendations['recommendations']
                
                if recommendations:
                    # Group by priority
                    high_priority = [r for r in recommendations if r.get('priority') == 'high']
                    med_priority = [r for r in recommendations if r.get('priority') == 'medium']
                    low_priority = [r for r in recommendations if r.get('priority') == 'low']
                    
                    # Display prioritized recommendations
                    if high_priority:
                        st.markdown("### High Priority Actions")
                        for i, rec in enumerate(high_priority, 1):
                            st.markdown(f"{i}. {rec.get('message', '')}")
                    
                    if med_priority:
                        st.markdown("### Medium Priority Actions")
                        for i, rec in enumerate(med_priority, 1):
                            st.markdown(f"{i}. {rec.get('message', '')}")
                    
                    if low_priority:
                        st.markdown("### Additional Recommendations")
                        for i, rec in enumerate(low_priority, 1):
                            st.markdown(f"{i}. {rec.get('message', '')}")
                
                st.markdown("---")
            
            # Methodology
            if include_methodology:
                st.markdown("## Methodology")
                
                st.markdown("""
                ### Data Collection
                Air quality data was collected using monitoring stations equipped with sensors for various pollutants including particulate matter (PM2.5, PM10), gases (NO2, O3, CO, SO2), and meteorological parameters (temperature, humidity, pressure).
                
                ### Data Processing
                The raw data underwent quality control procedures including:
                - Removal of outliers and invalid readings
                - Filling of missing values using appropriate interpolation methods
                - Aggregation to standardized time intervals
                - Calculation of air quality indices
                
                ### Analysis Methods
                The analysis includes:
                - Temporal trend analysis using moving averages and regression
                - Correlation analysis between different parameters
                - Pattern detection for daily and weekly variations
                - Statistical significance testing
                """)
                
                st.markdown("---")
        
        # Export options
        export_container = st.container()
        with export_container:
            st.subheader("Export Report", divider="green")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                export_format = st.radio(
                    "Choose Export Format",
                    options=["HTML", "PDF"],
                    index=0,
                    horizontal=True
                )
                
                include_raw_data = st.checkbox("Include raw data as CSV attachment", value=False)
            
            with col2:
                export_button = st.button("Generate Report", type="primary", use_container_width=True)
                
                if export_button:
                    # Generate and export report
                    with st.spinner("Generating report..."):
                        if export_format == "HTML":
                            html_content = generate_html_report(
                                df, 
                                ai_recommendations, 
                                report_title, 
                                report_description,
                                report_author,
                                report_org,
                                include_summary,
                                include_trends,
                                include_pollutants,
                                include_recommendations,
                                include_methodology,
                                include_overview_charts,
                                include_time_series,
                                include_correlations,
                                include_forecasts
                            )
                            
                            # Create download link
                            b64_html = base64.b64encode(html_content.encode()).decode()
                            # Process the filename safely outside the f-string
                            safe_filename = report_title.replace(" ", "_")
                            href = f'<a href="data:text/html;base64,{b64_html}" download="{safe_filename}.html">Download HTML Report</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        else:
                            st.info("PDF export is being implemented. Please use HTML export for now.")
                        
                        # Include raw data if requested
                        if include_raw_data:
                            csv = df.to_csv(index=False).encode()
                            b64_csv = base64.b64encode(csv).decode()
                            href_csv = f'<a href="data:text/csv;base64,{b64_csv}" download="air_quality_data.csv">Download Raw Data (CSV)</a>'
                            st.markdown(href_csv, unsafe_allow_html=True)
                    
                    st.success("Report generated successfully!")
        
    except Exception as e:
        logger.error(f"Error rendering report generator: {str(e)}")
        st.error(f"An error occurred while rendering the report generator: {str(e)}")

def generate_executive_summary(df, ai_recommendations=None):
    """
    Generate an executive summary based on the data and AI recommendations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    ai_recommendations : dict, optional
        AI-generated insights and recommendations
        
    Returns:
    --------
    str
        Formatted executive summary text
    """
    summary = []
    
    try:
        # Data overview
        if 'Datetime' in df.columns:
            date_range = f"{pd.to_datetime(df['Datetime']).min().strftime('%Y-%m-%d')} to {pd.to_datetime(df['Datetime']).max().strftime('%Y-%m-%d')}"
            days = (pd.to_datetime(df['Datetime']).max() - pd.to_datetime(df['Datetime']).min()).days + 1
            summary.append(f"This report analyzes air quality data collected over {days} days from {date_range}.")
        else:
            summary.append(f"This report analyzes air quality data containing {len(df)} records.")
        
        # Device information
        if 'Device_ID' in df.columns:
            device_count = df['Device_ID'].nunique()
            device_txt = "devices" if device_count > 1 else "device"
            summary.append(f"Data was collected from {device_count} monitoring {device_txt}.")
        
        # Overall air quality
        if 'AQI' in df.columns:
            avg_aqi = df['AQI'].mean()
            max_aqi = df['AQI'].max()
            
            # AQI category
            if avg_aqi <= 50:
                aqi_category = "Good"
                health_implication = "satisfactory, with minimal health risks"
            elif avg_aqi <= 100:
                aqi_category = "Moderate"
                health_implication = "acceptable, though there may be health concerns for a small number of sensitive individuals"
            elif avg_aqi <= 150:
                aqi_category = "Unhealthy for Sensitive Groups"
                health_implication = "concerning for sensitive groups, who may experience health effects"
            elif avg_aqi <= 200:
                aqi_category = "Unhealthy"
                health_implication = "concerning, with potential health effects for the general population"
            elif avg_aqi <= 300:
                aqi_category = "Very Unhealthy"
                health_implication = "a health risk for the general population, with more serious effects for sensitive groups"
            else:
                aqi_category = "Hazardous"
                health_implication = "a serious health risk for everyone"
                
            summary.append(f"The average Air Quality Index (AQI) was {avg_aqi:.1f}, categorized as **{aqi_category}**. This indicates air quality that is {health_implication}.")
            summary.append(f"The maximum recorded AQI was {max_aqi:.1f}.")
        
        # Pollutant information
        pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
        
        if pollutant_cols:
            # Check exceedance rates
            thresholds = {
                'PM2.5': 35.0,  # μg/m³
                'PM10': 50.0,   # μg/m³
                'NO2': 100.0,   # ppb
                'O3': 70.0,     # ppb
                'CO': 9.0,      # ppm
                'SO2': 75.0     # ppb
            }
            
            exceedance_data = []
            for pollutant in pollutant_cols:
                if pollutant in thresholds:
                    exceedance_rate = (df[pollutant] > thresholds[pollutant]).mean() * 100
                    if exceedance_rate > 5:  # Only mention if significant
                        exceedance_data.append((pollutant, exceedance_rate))
            
            if exceedance_data:
                exceedance_data.sort(key=lambda x: x[1], reverse=True)
                pollutant, rate = exceedance_data[0]
                summary.append(f"**{pollutant}** was the most problematic pollutant, exceeding recommended limits {rate:.1f}% of the time.")
                
                if len(exceedance_data) > 1:
                    pollutant2, rate2 = exceedance_data[1]
                    summary.append(f"**{pollutant2}** also showed elevated levels, exceeding limits {rate2:.1f}% of the time.")
        
        # Key insights from AI recommendations
        if ai_recommendations and 'general_insights' in ai_recommendations and ai_recommendations['general_insights']:
            summary.append("\n**Key Insights:**")
            
            # Add up to 3 key insights
            for insight in ai_recommendations['general_insights'][:3]:
                summary.append(f"- {insight}")
        
        # Top recommendations if available
        if ai_recommendations and 'recommendations' in ai_recommendations and ai_recommendations['recommendations']:
            high_priority = [r for r in ai_recommendations['recommendations'] if r.get('priority') == 'high']
            
            if high_priority:
                summary.append("\n**Priority Recommendations:**")
                for i, rec in enumerate(high_priority[:2], 1):  # List top 2 high priority recommendations
                    summary.append(f"- {rec.get('message', '')}")
        
        return "\n\n".join(summary)
        
    except Exception as e:
        logger.error(f"Error generating executive summary: {str(e)}")
        return "An error occurred while generating the executive summary."

def get_aqi_color(aqi_value):
    """
    Get the color corresponding to an AQI value.
    
    Parameters:
    -----------
    aqi_value : float
        The AQI value
        
    Returns:
    --------
    str
        Hex color code for the AQI value
    """
    if aqi_value <= 50:
        return "#00e400"  # Green - Good
    elif aqi_value <= 100:
        return "#ffff00"  # Yellow - Moderate
    elif aqi_value <= 150:
        return "#ff7e00"  # Orange - Unhealthy for Sensitive Groups
    elif aqi_value <= 200:
        return "#ff0000"  # Red - Unhealthy
    elif aqi_value <= 300:
        return "#8f3f97"  # Purple - Very Unhealthy
    else:
        return "#7e0023"  # Maroon - Hazardous

def generate_html_report(df, ai_recommendations, title, description, author, organization,
                        include_summary, include_trends, include_pollutants, include_recommendations,
                        include_methodology, include_overview_charts, include_time_series,
                        include_correlations, include_forecasts):
    """
    Generate an HTML report based on the selected options.
    
    Parameters:
    -----------
    Various parameters controlling report content
        
    Returns:
    --------
    str
        HTML content of the report
    """
    try:
        # Start building HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: #4bb051;
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                }}
                h1 {{
                    border-bottom: 2px solid #4bb051;
                    padding-bottom: 10px;
                }}
                h2 {{
                    border-bottom: 1px solid #dee8dd;
                    padding-bottom: 5px;
                }}
                .metadata {{
                    color: #666;
                    font-size: 0.9em;
                    margin-bottom: 20px;
                }}
                .executive-summary {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-left: 5px solid #4bb051;
                    margin: 20px 0;
                }}
                .metric-row {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    flex: 1;
                    min-width: 200px;
                    background-color: #fff;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #4bb051;
                    margin: 10px 0;
                }}
                .metric-label {{
                    color: #666;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin: 20px 0;
                    border: 1px solid #ddd;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .chart-container {{
                    margin: 30px 0;
                }}
                .recommendation {{
                    background-color: #f0f7f0;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                }}
                .high-priority {{
                    border-left: 5px solid #d9534f;
                }}
                .medium-priority {{
                    border-left: 5px solid #f0ad4e;
                }}
                .low-priority {{
                    border-left: 5px solid #5bc0de;
                }}
                .footer {{
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                    font-size: 0.8em;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="metadata">
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d")}
        """
        
        # Add author and organization if provided
        if author:
            html_content += f" | Author: {author}"
        if organization:
            html_content += f" | Organization: {organization}"
            
        html_content += """
                </p>
            </div>
        """
        
        # Add description
        if description:
            html_content += f"<p>{description}</p>"
        
        # Executive Summary
        if include_summary:
            summary_text = generate_executive_summary(df, ai_recommendations)
            # Process summary text outside the f-string
            processed_summary = summary_text.replace("\n\n", "<br><br>")
            
            html_content += f"""
            <h2>Executive Summary</h2>
            <div class="executive-summary">
                {processed_summary}
            </div>
            
            <div class="metric-row">
            """
            
            # Add key metrics
            if 'AQI' in df.columns:
                avg_aqi = df['AQI'].mean()
                aqi_color = get_aqi_color(avg_aqi)
                html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">Average AQI</div>
                    <div class="metric-value" style="color: {aqi_color};">{avg_aqi:.1f}</div>
                </div>
                """
            
            if 'Datetime' in df.columns:
                days = (pd.to_datetime(df['Datetime']).max() - pd.to_datetime(df['Datetime']).min()).days + 1
                html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">Days Analyzed</div>
                    <div class="metric-value">{days}</div>
                </div>
                """
            
            records = len(df)
            html_content += f"""
            <div class="metric-card">
                <div class="metric-label">Data Points</div>
                <div class="metric-value">{records:,}</div>
            </div>
            """
            
            html_content += """
            </div>
            """
        
        # Overview Charts
        if include_overview_charts:
            html_content += """
            <h2>Overview</h2>
            <p>This section provides an overview of the air quality data through key visualizations.</p>
            
            <div class="chart-container">
                <img src="data:image/png;base64,PLACEHOLDER_AQI_DISTRIBUTION" alt="AQI Distribution Chart">
                <p><em>Figure 1: Distribution of Air Quality Index (AQI) readings.</em></p>
            </div>
            
            <div class="chart-container">
                <img src="data:image/png;base64,PLACEHOLDER_POLLUTANT_COMPARISON" alt="Pollutant Comparison Chart">
                <p><em>Figure 2: Comparison of average pollutant concentrations across monitoring devices.</em></p>
            </div>
            """
        
        # Trend Analysis
        if include_trends:
            html_content += """
            <h2>Trend Analysis</h2>
            <p>This section examines temporal patterns and trends in the air quality data.</p>
            """
            
            if include_time_series:
                html_content += """
                <div class="chart-container">
                    <img src="data:image/png;base64,PLACEHOLDER_TIME_SERIES" alt="Time Series Chart">
                    <p><em>Figure 3: Time series analysis showing air quality trends over the monitored period.</em></p>
                </div>
                """
            
            html_content += """
            <h3>Daily and Weekly Patterns</h3>
            <div class="chart-container">
                <img src="data:image/png;base64,PLACEHOLDER_DAILY_PATTERN" alt="Daily Pattern Chart">
                <p><em>Figure 4: Daily pattern showing how air quality varies by hour of day.</em></p>
            </div>
            
            <div class="chart-container">
                <img src="data:image/png;base64,PLACEHOLDER_WEEKLY_PATTERN" alt="Weekly Pattern Chart">
                <p><em>Figure 5: Weekly pattern showing air quality variations by day of week.</em></p>
            </div>
            """
            
            if include_correlations:
                html_content += """
                <h3>Parameter Correlations</h3>
                <p>This analysis shows how different air quality parameters relate to each other and to meteorological conditions.</p>
                
                <div class="chart-container">
                    <img src="data:image/png;base64,PLACEHOLDER_CORRELATION_HEATMAP" alt="Correlation Heatmap">
                    <p><em>Figure 6: Correlation matrix showing relationships between different parameters.</em></p>
                </div>
                """
                
                # Add correlation insights
                if (ai_recommendations and 'correlations' in ai_recommendations and 
                    'insights' in ai_recommendations['correlations'] and 
                    ai_recommendations['correlations']['insights']):
                    
                    html_content += """
                    <h4>Key Correlation Findings:</h4>
                    <ul>
                    """
                    
                    for insight in ai_recommendations['correlations']['insights']:
                        html_content += f"<li>{insight}</li>"
                    
                    html_content += """
                    </ul>
                    """
            
            if include_forecasts and ai_recommendations and 'forecasts' in ai_recommendations:
                html_content += """
                <h3>Forecasts</h3>
                <p>Based on historical patterns, this section provides forecasts for future air quality levels.</p>
                
                <div class="chart-container">
                    <img src="data:image/png;base64,PLACEHOLDER_FORECAST" alt="Air Quality Forecast">
                    <p><em>Figure 7: Forecast of future air quality based on historical patterns.</em></p>
                </div>
                """
        
        # Pollutant Details
        if include_pollutants:
            html_content += """
            <h2>Pollutant Analysis</h2>
            <p>This section provides detailed analysis of individual pollutants monitored in this study.</p>
            """
            
            pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
            
            for i, pollutant in enumerate(pollutant_cols, 1):
                avg_value = df[pollutant].mean()
                max_value = df[pollutant].max()
                min_value = df[pollutant].min()
                
                # Units for each pollutant
                units = {
                    'PM2.5': 'μg/m³',
                    'PM10': 'μg/m³',
                    'NO2': 'ppb',
                    'O3': 'ppb',
                    'CO': 'ppm',
                    'SO2': 'ppb'
                }
                
                unit = units.get(pollutant, '')
                
                # Get threshold and exceedance rate
                thresholds = {
                    'PM2.5': 35.0,
                    'PM10': 50.0,
                    'NO2': 100.0,
                    'O3': 70.0,
                    'CO': 9.0,
                    'SO2': 75.0
                }
                
                threshold = thresholds.get(pollutant, 0)
                exceedance_rate = (df[pollutant] > threshold).mean() * 100 if threshold > 0 else 0
                
                html_content += f"""
                <h3>{pollutant}</h3>
                
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-label">Average</div>
                        <div class="metric-value">{avg_value:.2f} {unit}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Maximum</div>
                        <div class="metric-value">{max_value:.2f} {unit}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Minimum</div>
                        <div class="metric-value">{min_value:.2f} {unit}</div>
                    </div>
                """
                
                if threshold > 0:
                    html_content += f"""
                    <div class="metric-card">
                        <div class="metric-label">Threshold Exceedance</div>
                        <div class="metric-value">{exceedance_rate:.1f}%</div>
                        <div class="metric-label">Threshold: {threshold} {unit}</div>
                    </div>
                    """
                
                html_content += """
                </div>
                
                <div class="chart-container">
                    <img src="data:image/png;base64,PLACEHOLDER_POLLUTANT_TIME_SERIES" alt="Pollutant Time Series">
                    <p><em>Figure {}: Time series analysis of {} levels over the monitored period.</em></p>
                </div>
                """.format(7 + i, pollutant)
                
                # Health impact info if available
                if (ai_recommendations and 'health_impact' in ai_recommendations and 
                    'pollutant_impacts' in ai_recommendations['health_impact'] and
                    pollutant in ai_recommendations['health_impact']['pollutant_impacts']):
                    
                    health_info = ai_recommendations['health_impact']['pollutant_impacts'][pollutant]
                    
                    if 'health_effects' in health_info and health_info['health_effects']:
                        html_content += """
                        <h4>Health Effects</h4>
                        <ul>
                        """
                        
                        for effect in health_info['health_effects']:
                            html_content += f"<li>{effect}</li>"
                        
                        html_content += """
                        </ul>
                        """
        
        # Recommendations
        if include_recommendations and ai_recommendations and 'recommendations' in ai_recommendations:
            html_content += """
            <h2>Recommendations</h2>
            <p>Based on the analysis of the air quality data, the following recommendations are provided to improve air quality and minimize health impacts.</p>
            """
            
            recommendations = ai_recommendations['recommendations']
            
            if recommendations:
                # Group by priority
                high_priority = [r for r in recommendations if r.get('priority') == 'high']
                med_priority = [r for r in recommendations if r.get('priority') == 'medium']
                low_priority = [r for r in recommendations if r.get('priority') == 'low']
                
                # Display prioritized recommendations
                if high_priority:
                    html_content += """
                    <h3>High Priority Actions</h3>
                    """
                    
                    for i, rec in enumerate(high_priority, 1):
                        html_content += f"""
                        <div class="recommendation high-priority">
                            <strong>{i}.</strong> {rec.get('message', '')}
                        </div>
                        """
                
                if med_priority:
                    html_content += """
                    <h3>Medium Priority Actions</h3>
                    """
                    
                    for i, rec in enumerate(med_priority, 1):
                        html_content += f"""
                        <div class="recommendation medium-priority">
                            <strong>{i}.</strong> {rec.get('message', '')}
                        </div>
                        """
                
                if low_priority:
                    html_content += """
                    <h3>Additional Recommendations</h3>
                    """
                    
                    for i, rec in enumerate(low_priority, 1):
                        html_content += f"""
                        <div class="recommendation low-priority">
                            <strong>{i}.</strong> {rec.get('message', '')}
                        </div>
                        """
        
        # Methodology
        if include_methodology:
            html_content += """
            <h2>Methodology</h2>
            
            <h3>Data Collection</h3>
            <p>Air quality data was collected using monitoring stations equipped with sensors for various pollutants including particulate matter (PM2.5, PM10), gases (NO2, O3, CO, SO2), and meteorological parameters (temperature, humidity, pressure).</p>
            
            <h3>Data Processing</h3>
            <p>The raw data underwent quality control procedures including:</p>
            <ul>
                <li>Removal of outliers and invalid readings</li>
                <li>Filling of missing values using appropriate interpolation methods</li>
                <li>Aggregation to standardized time intervals</li>
                <li>Calculation of air quality indices</li>
            </ul>
            
            <h3>Analysis Methods</h3>
            <p>The analysis includes:</p>
            <ul>
                <li>Temporal trend analysis using moving averages and regression</li>
                <li>Correlation analysis between different parameters</li>
                <li>Pattern detection for daily and weekly variations</li>
                <li>Statistical significance testing</li>
            </ul>
            """
        
        # Footer
        html_content += f"""
            <div class="footer">
                <p>This report was generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} using Oizom's Smart Airlytics platform.</p>
                <p>© {datetime.now().year} Oizom. All rights reserved.</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        return f"<html><body><h1>Error Generating Report</h1><p>{str(e)}</p></body></html>"
