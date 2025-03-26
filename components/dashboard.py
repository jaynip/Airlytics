import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# Import utility functions
from utils.visualization import (
    create_correlation_heatmap, 
    plot_time_series,
    create_aqi_distribution_chart,
    create_pollutant_comparison,
    create_daily_pattern_chart,
    create_weekly_pattern_chart,
    create_forecast_chart,
    create_pollutant_box_plot
)

# Set up logging
logger = logging.getLogger(__name__)

def render_dashboard(df, ai_recommendations=None):
    """
    Render the main dashboard view with charts, KPIs, and insights.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    ai_recommendations : dict, optional
        AI-generated insights and recommendations
    """
    try:
        if df is None or df.empty:
            st.error("No data available for dashboard. Please upload data or use sample data.")
            return
        
        # Determine data time range
        if 'Datetime' in df.columns:
            start_date = pd.to_datetime(df['Datetime']).min()
            end_date = pd.to_datetime(df['Datetime']).max()
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            time_span = (end_date - start_date).days + 1
        else:
            date_range = "Unknown date range"
            time_span = 0
        
        # Dashboard header
        st.header("Air Quality Dashboard", divider="green")
        
        # Top metrics row
        metrics_container = st.container()
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            
            # Average AQI
            with col1:
                if 'AQI' in df.columns:
                    avg_aqi = df['AQI'].mean()
                    aqi_delta = None
                    
                    # Calculate change from previous period if enough data
                    if 'Datetime' in df.columns and time_span > 2:
                        mid_point = start_date + (end_date - start_date) / 2
                        recent_aqi = df[pd.to_datetime(df['Datetime']) > mid_point]['AQI'].mean()
                        earlier_aqi = df[pd.to_datetime(df['Datetime']) <= mid_point]['AQI'].mean()
                        
                        if not np.isnan(recent_aqi) and not np.isnan(earlier_aqi) and earlier_aqi != 0:
                            aqi_delta = f"{((recent_aqi - earlier_aqi) / earlier_aqi) * 100:.1f}%"
                    
                    # Determine AQI text color based on value
                    if avg_aqi <= 50:
                        aqi_color = "green"
                    elif avg_aqi <= 100:
                        aqi_color = "orange"
                    else:
                        aqi_color = "red"
                    
                    st.metric(
                        "Average AQI", 
                        f"{avg_aqi:.1f}", 
                        delta=aqi_delta,
                        delta_color="inverse"
                    )
                else:
                    st.metric("Average AQI", "N/A")
            
            # Dominant Pollutant
            with col2:
                pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
                
                if pollutant_cols:
                    # Calculate each pollutant's average as percentage of its typical threshold
                    thresholds = {
                        'PM2.5': 35.0,  # μg/m³
                        'PM10': 50.0,   # μg/m³
                        'NO2': 100.0,   # ppb
                        'O3': 70.0,     # ppb
                        'CO': 9.0,      # ppm
                        'SO2': 75.0     # ppb
                    }
                    
                    pollutant_ratios = {}
                    for pollutant in pollutant_cols:
                        if pollutant in thresholds:
                            avg_value = df[pollutant].mean()
                            ratio = avg_value / thresholds[pollutant]
                            pollutant_ratios[pollutant] = ratio
                    
                    if pollutant_ratios:
                        dominant_pollutant = max(pollutant_ratios.items(), key=lambda x: x[1])[0]
                        dom_poll_val = df[dominant_pollutant].mean()
                        dom_poll_ratio = pollutant_ratios[dominant_pollutant]
                        
                        st.metric(
                            "Dominant Pollutant", 
                            f"{dominant_pollutant}", 
                            f"{dom_poll_val:.1f} ({dom_poll_ratio:.1f}x threshold)"
                        )
                    else:
                        st.metric("Dominant Pollutant", "N/A")
                else:
                    st.metric("Dominant Pollutant", "N/A")
            
            # Data Timespan
            with col3:
                if 'Datetime' in df.columns:
                    st.metric(
                        "Time Period", 
                        f"{time_span} days", 
                        f"{len(df)} records"
                    )
                else:
                    st.metric("Time Period", "Unknown", f"{len(df)} records")
            
            # Location Count
            with col4:
                if 'Device_ID' in df.columns:
                    device_count = df['Device_ID'].nunique()
                    st.metric(
                        "Monitoring Locations", 
                        f"{device_count}",
                        f"{len(df) / device_count:.1f} records/location"
                    )
                else:
                    st.metric("Monitoring Locations", "N/A")
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time Analysis", "Pollutant Details", "AI Insights"])
        
        # TAB 1: Overview
        with tab1:
            # Time series and AQI distribution side by side
            chart_row1 = st.container()
            with chart_row1:
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Time series of AQI or main pollutant
                    if 'Datetime' in df.columns:
                        if 'AQI' in df.columns:
                            ts_param = 'AQI'
                        elif 'PM2.5' in df.columns:
                            ts_param = 'PM2.5'
                        else:
                            # Use first available pollutant
                            ts_param = next((p for p in pollutant_cols), None)
                            
                        if ts_param:
                            ts_fig = plot_time_series(df, parameters=[ts_param], rolling_window=5)
                            if ts_fig:
                                st.plotly_chart(ts_fig, use_container_width=True, key="overview_ts_fig_1")
                            else:
                                st.error(f"Could not create time series chart for {ts_param}")
                        else:
                            st.info("No suitable parameters found for time series chart")
                    else:
                        st.info("Time data not available for time series chart")
                
                with col2:
                    # AQI Distribution
                    if 'AQI' in df.columns:
                        aqi_dist_fig = create_aqi_distribution_chart(df)
                        if aqi_dist_fig:
                            st.plotly_chart(aqi_dist_fig, use_container_width=True, key="aqi_dist_fig")
                        else:
                            st.error("Could not create AQI distribution chart")
                    else:
                        st.info("AQI data not available for distribution chart")
            
            # Pollutant comparison and correlation
            chart_row2 = st.container()
            with chart_row2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pollutant comparison across devices
                    if 'Device_ID' in df.columns and df['Device_ID'].nunique() > 1:
                        pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
                        if pollutant_cols:
                            comp_fig = create_pollutant_comparison(df, pollutants=pollutant_cols[:3])
                            if comp_fig:
                                st.plotly_chart(comp_fig, use_container_width=True)
                            else:
                                st.error("Could not create pollutant comparison chart")
                        else:
                            st.info("No pollutant data available for comparison")
                    else:
                        # If only one device, show box plots instead
                        pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
                        if pollutant_cols:
                            box_fig = create_pollutant_box_plot(df, pollutants=pollutant_cols[:5])
                            if box_fig:
                                st.plotly_chart(box_fig, use_container_width=True)
                            else:
                                st.error("Could not create pollutant box plot")
                        else:
                            st.info("No pollutant data available for box plot")
                
                with col2:
                    # Correlation heatmap
                    correlation_cols = [col for col in df.columns if col in 
                                      ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'AQI', 
                                       'Temperature', 'Humidity', 'Pressure']]
                    
                    if len(correlation_cols) >= 2:
                        corr_fig = create_correlation_heatmap(df, selected_columns=correlation_cols)
                        if corr_fig:
                            st.plotly_chart(corr_fig, use_container_width=True)
                        else:
                            st.error("Could not create correlation heatmap")
                    else:
                        st.info("Not enough parameters available for correlation analysis")
        
        # TAB 2: Time Analysis
        with tab2:
            if 'Datetime' in df.columns:
                # Time pattern selector
                pattern_container = st.container()
                with pattern_container:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Parameter selection
                        available_params = [col for col in df.columns if col in 
                                         ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'AQI']]
                        
                        if available_params:
                            default_param = 'AQI' if 'AQI' in available_params else available_params[0]
                            selected_param = st.selectbox(
                                "Select Parameter",
                                options=available_params,
                                index=available_params.index(default_param)
                            )
                            
                            # Time window selection
                            time_windows = ["Raw Data", "Hourly Average", "Daily Average", "Weekly Average"]
                            selected_window = st.radio("Time Aggregation", time_windows)
                            
                            # Show trend line option
                            show_trend = st.checkbox("Show Trend Line", value=True)
                            
                            # Download data button
                            st.download_button(
                                "Download Data",
                                data=df.to_csv(index=False).encode('utf-8'),
                                file_name=f"air_quality_data_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No parameters available for time analysis")
                    
                    with col2:
                        # Create aggregated view based on selection
                        if 'selected_param' in locals() and selected_param:
                            # Prepare data based on selected time window
                            df_copy = df.copy()
                            df_copy['Datetime'] = pd.to_datetime(df_copy['Datetime'])
                            
                            if selected_window == "Hourly Average":
                                df_agg = df_copy.set_index('Datetime').resample('H').mean().reset_index()
                                window_size = 12  # For 12-hour rolling average
                            elif selected_window == "Daily Average":
                                df_agg = df_copy.set_index('Datetime').resample('D').mean().reset_index()
                                window_size = 7  # For 7-day rolling average
                            elif selected_window == "Weekly Average":
                                df_agg = df_copy.set_index('Datetime').resample('W').mean().reset_index()
                                window_size = 4  # For 4-week rolling average
                            else:
                                df_agg = df_copy
                                window_size = 24  # For raw data, use 24-point rolling average
                            
                            # Create time series chart
                            rolling_window = window_size if len(df_agg) > window_size else None
                            ts_fig = plot_time_series(
                                df_agg, 
                                parameters=[selected_param],
                                rolling_window=rolling_window,
                                add_trend=show_trend
                            )
                            
                            if ts_fig:
                                st.plotly_chart(ts_fig, use_container_width=True)
                            else:
                                st.error(f"Could not create time series chart for {selected_param}")
                
                # Daily and weekly patterns
                patterns_container = st.container()
                with patterns_container:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Daily pattern
                        if 'selected_param' in locals() and selected_param:
                            daily_fig = create_daily_pattern_chart(df, selected_param)
                            if daily_fig:
                                st.plotly_chart(daily_fig, use_container_width=True)
                            else:
                                st.error(f"Could not create daily pattern chart for {selected_param}")
                    
                    with col2:
                        # Weekly pattern
                        if 'selected_param' in locals() and selected_param:
                            weekly_fig = create_weekly_pattern_chart(df, selected_param)
                            if weekly_fig:
                                st.plotly_chart(weekly_fig, use_container_width=True)
                            else:
                                st.error(f"Could not create weekly pattern chart for {selected_param}")
                
                # Forecast section if AI recommendations available
                if ai_recommendations and 'forecasts' in ai_recommendations:
                    forecast_container = st.container()
                    with forecast_container:
                        st.subheader("Forecasts", divider="green")
                        
                        available_forecasts = [p for p in ['AQI', 'PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2'] 
                                             if p in ai_recommendations['forecasts']]
                        
                        if available_forecasts:
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                forecast_param = st.selectbox(
                                    "Select Parameter for Forecast",
                                    options=available_forecasts,
                                    index=0
                                )
                                
                                # Display forecast metadata
                                if 'metadata' in ai_recommendations['forecasts']:
                                    meta = ai_recommendations['forecasts']['metadata']
                                    st.info(
                                        f"Forecast for next {meta.get('forecast_hours', 24)} hours\n\n"
                                        f"Generated: {meta.get('generated_at', 'Unknown')}"
                                    )
                            
                            with col2:
                                # Create forecast chart
                                if forecast_param in ai_recommendations['forecasts']:
                                    forecast_fig = create_forecast_chart(
                                        df, 
                                        ai_recommendations['forecasts'], 
                                        forecast_param
                                    )
                                    
                                    if forecast_fig:
                                        st.plotly_chart(forecast_fig, use_container_width=True)
                                    else:
                                        st.error(f"Could not create forecast chart for {forecast_param}")
                        else:
                            st.info("No forecast data available")
            else:
                st.info("Time data not available for temporal analysis")
        
        # TAB 3: Pollutant Details
        with tab3:
            pollutant_tabs_container = st.container()
            with pollutant_tabs_container:
                # Get available pollutants
                available_pollutants = [col for col in df.columns if col in 
                                     ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
                
                if available_pollutants:
                    pollutant_tabs = st.tabs(available_pollutants)
                    
                    for i, pollutant in enumerate(available_pollutants):
                        with pollutant_tabs[i]:
                            # Pollutant info and statistics
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                # Key statistics
                                avg_value = df[pollutant].mean()
                                max_value = df[pollutant].max()
                                std_value = df[pollutant].std()
                                
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
                                
                                # Display metrics
                                st.metric("Average", f"{avg_value:.2f} {unit}")
                                st.metric("Maximum", f"{max_value:.2f} {unit}")
                                st.metric("Standard Deviation", f"{std_value:.2f} {unit}")
                                
                                if threshold > 0:
                                    st.metric(
                                        "Threshold Exceedance", 
                                        f"{exceedance_rate:.1f}%",
                                        f"Threshold: {threshold} {unit}"
                                    )
                                
                                # Health impact info if available
                                if (ai_recommendations and 'health_impact' in ai_recommendations and 
                                    'pollutant_impacts' in ai_recommendations['health_impact'] and
                                    pollutant in ai_recommendations['health_impact']['pollutant_impacts']):
                                    
                                    health_info = ai_recommendations['health_impact']['pollutant_impacts'][pollutant]
                                    
                                    if 'health_effects' in health_info and health_info['health_effects']:
                                        st.markdown("##### Health Effects")
                                        for effect in health_info['health_effects']:
                                            st.markdown(f"- {effect}")
                            
                            with col2:
                                # Time series of this pollutant
                                if 'Datetime' in df.columns:
                                    ts_fig = plot_time_series(
                                        df, 
                                        parameters=[pollutant], 
                                        rolling_window=24,
                                        add_trend=True
                                    )
                                    
                                    if ts_fig:
                                        st.plotly_chart(ts_fig, use_container_width=True)
                                    else:
                                        st.error(f"Could not create time series chart for {pollutant}")
                            
                            # Additional charts for this pollutant
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Daily pattern
                                if 'Datetime' in df.columns:
                                    daily_fig = create_daily_pattern_chart(df, pollutant)
                                    if daily_fig:
                                        st.plotly_chart(daily_fig, use_container_width=True)
                                    else:
                                        st.error(f"Could not create daily pattern chart for {pollutant}")
                            
                            with col2:
                                # Device comparison if multiple devices
                                if 'Device_ID' in df.columns and df['Device_ID'].nunique() > 1:
                                    comp_fig = create_pollutant_comparison(df, pollutants=[pollutant])
                                    if comp_fig:
                                        st.plotly_chart(comp_fig, use_container_width=True)
                                    else:
                                        st.error(f"Could not create device comparison for {pollutant}")
                                else:
                                    # Weekly pattern if only one device
                                    if 'Datetime' in df.columns:
                                        weekly_fig = create_weekly_pattern_chart(df, pollutant)
                                        if weekly_fig:
                                            st.plotly_chart(weekly_fig, use_container_width=True)
                                        else:
                                            st.error(f"Could not create weekly pattern chart for {pollutant}")
                else:
                    st.info("No pollutant data available for detailed analysis")
        
        # TAB 4: AI Insights
        with tab4:
            if ai_recommendations:
                insights_container = st.container()
                with insights_container:
                    # Display general insights
                    if 'general_insights' in ai_recommendations and ai_recommendations['general_insights']:
                        st.subheader("Key Insights", divider="green")
                        
                        for insight in ai_recommendations['general_insights']:
                            st.markdown(f"- {insight}")
                    
                    # Display recommendations
                    if 'recommendations' in ai_recommendations and ai_recommendations['recommendations']:
                        st.subheader("Recommendations", divider="green")
                        
                        # Group recommendations by priority
                        high_priority = [r for r in ai_recommendations['recommendations'] 
                                        if r.get('priority') == 'high']
                        med_priority = [r for r in ai_recommendations['recommendations'] 
                                        if r.get('priority') == 'medium']
                        low_priority = [r for r in ai_recommendations['recommendations'] 
                                        if r.get('priority') == 'low']
                        
                        # Display high priority recommendations
                        if high_priority:
                            st.markdown("##### High Priority")
                            for rec in high_priority:
                                st.markdown(
                                    f"<div style='background-color: rgba(217, 83, 79, 0.15); border-left: 4px solid #d9534f; "
                                    f"padding: 10px; margin-bottom: 10px;'>{rec.get('message', '')}</div>", 
                                    unsafe_allow_html=True
                                )
                        
                        # Display medium priority recommendations
                        if med_priority:
                            st.markdown("##### Medium Priority")
                            for rec in med_priority:
                                st.markdown(
                                    f"<div style='background-color: rgba(240, 173, 78, 0.15); border-left: 4px solid #f0ad4e; "
                                    f"padding: 10px; margin-bottom: 10px;'>{rec.get('message', '')}</div>", 
                                    unsafe_allow_html=True
                                )
                        
                        # Display low priority recommendations
                        if low_priority:
                            st.markdown("##### Low Priority")
                            for rec in low_priority:
                                st.markdown(
                                    f"<div style='background-color: rgba(91, 192, 222, 0.15); border-left: 4px solid #5bc0de; "
                                    f"padding: 10px; margin-bottom: 10px;'>{rec.get('message', '')}</div>", 
                                    unsafe_allow_html=True
                                )
                    
                    # Display anomalies if detected
                    if 'anomalies' in ai_recommendations and ai_recommendations['anomalies']:
                        st.subheader("Detected Anomalies", divider="green")
                        
                        # Filter out error messages
                        anomalies = [a for a in ai_recommendations['anomalies'] 
                                   if a.get('type') not in ['error', 'message']]
                        
                        # Display anomalies table if there are any
                        if anomalies:
                            # Create a more readable table of anomalies
                            anomaly_data = []
                            
                            for anomaly in anomalies:
                                anomaly_type = anomaly.get('type', 'Unknown')
                                pollutant = anomaly.get('pollutant', 'Unknown')
                                count = anomaly.get('count', 0)
                                message = anomaly.get('message', '')
                                
                                anomaly_data.append({
                                    'Pollutant': pollutant,
                                    'Type': anomaly_type.title(),
                                    'Count': count,
                                    'Description': message
                                })
                            
                            if anomaly_data:
                                st.dataframe(
                                    pd.DataFrame(anomaly_data),
                                    column_config={
                                        'Pollutant': st.column_config.TextColumn("Pollutant"),
                                        'Type': st.column_config.TextColumn("Anomaly Type"),
                                        'Count': st.column_config.NumberColumn("Count"),
                                        'Description': st.column_config.TextColumn("Description", width="large")
                                    },
                                    hide_index=True,
                                    use_container_width=True
                                )
                        else:
                            st.info("No significant anomalies detected in the data")
                    
                    # Display correlation insights
                    if ('correlations' in ai_recommendations and 
                        'insights' in ai_recommendations['correlations'] and 
                        ai_recommendations['correlations']['insights']):
                        
                        st.subheader("Correlation Insights", divider="green")
                        
                        for insight in ai_recommendations['correlations']['insights']:
                            st.markdown(f"- {insight}")
                    
                    # Display health impact assessment
                    if 'health_impact' in ai_recommendations:
                        st.subheader("Health Impact Assessment", divider="green")
                        
                        health_impact = ai_recommendations['health_impact']
                        
                        # Overall health risk
                        if 'overall_risk' in health_impact:
                            risk = health_impact['overall_risk']
                            
                            # Determine color based on risk level
                            risk_colors = {
                                'Low': 'green',
                                'Moderate': 'orange',
                                'Elevated': 'orange',
                                'High': 'red',
                                'Very High': 'red',
                                'Hazardous': 'red'
                            }
                            
                            risk_color = risk_colors.get(risk.get('level'), 'gray')
                            
                            st.markdown(
                                f"<div style='background-color: rgba({risk_color=='green' and '0,128,0' or risk_color=='orange' and '255,165,0' or risk_color=='red' and '255,0,0' or '128,128,128'}, 0.15); "
                                f"border-left: 4px solid {risk_color}; padding: 15px; margin-bottom: 15px; border-radius: 5px;'>"
                                f"<h5 style='margin-top: 0;'>Overall Health Risk: {risk.get('level')}</h5>"
                                f"<p>{risk.get('description', '')}</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        
                        # Vulnerable groups
                        if 'vulnerable_groups' in health_impact and health_impact['vulnerable_groups']:
                            st.markdown("##### Vulnerable Groups")
                            
                            for group in health_impact['vulnerable_groups']:
                                st.markdown(
                                    f"<div style='background-color: #f8f9fa; padding: 10px; "
                                    f"margin-bottom: 8px; border-radius: 5px;'>"
                                    f"<strong>{group.get('group', '')}</strong>: {group.get('concern', '')}"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
            else:
                st.info("AI recommendations not available. Please process the data to generate insights.")
                
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        st.error(f"An error occurred while rendering the dashboard: {str(e)}")
