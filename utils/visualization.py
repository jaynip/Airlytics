import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import logging

# Set up logging
logger = logging.getLogger(__name__)

def create_correlation_heatmap(df, selected_columns=None):
    """
    Create a correlation heatmap for the selected columns in the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    selected_columns : list, optional
        List of column names to include in the correlation heatmap
        
    Returns:
    --------
    plotly.graph_objects.Figure
        A plotly figure object containing the correlation heatmap
    """
    try:
        if df is None or df.empty:
            logger.error("Cannot create correlation heatmap: DataFrame is empty or None")
            return None
            
        # If no columns specified, use all numeric columns
        if selected_columns is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Filter out non-relevant columns
            excluded_patterns = ['ID', 'Latitude', 'Longitude']
            selected_columns = [col for col in numeric_cols 
                               if not any(pattern in col for pattern in excluded_patterns)]
        
        # Calculate correlation matrix
        corr_matrix = df[selected_columns].corr().round(2)
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title="Parameter Correlation Matrix"
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            width=800,
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1<br>Strong<br>Negative", "-0.5<br>Moderate<br>Negative", "0<br>No<br>Correlation", "0.5<br>Moderate<br>Positive", "1<br>Strong<br>Positive"],
            )
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate="<b>%{y} vs %{x}</b><br>Correlation Coefficient: %{z:.2f}<extra></extra>"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")
        return None

def plot_time_series(df, parameters=None, device_ids=None, rolling_window=None, add_trend=False):
    """
    Create a time series plot for selected parameters.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    parameters : list, optional
        List of parameter names to plot
    device_ids : list, optional
        List of device IDs to include
    rolling_window : int, optional
        Window size for rolling average (None for no smoothing)
    add_trend : bool, optional
        Whether to add a trend line
        
    Returns:
    --------
    plotly.graph_objects.Figure
        A plotly figure object containing the time series plot
    """
    try:
        if df is None or df.empty or 'Datetime' not in df.columns:
            logger.error("Cannot create time series plot: Invalid data")
            return None
            
        # Make a copy of the dataframe to avoid modifying the original
        plot_df = df.copy()
        
        # Convert datetime column to datetime type if it's not already
        plot_df['Datetime'] = pd.to_datetime(plot_df['Datetime'])
        
        # Filter by device ID if specified
        if device_ids:
            if 'Device_ID' in plot_df.columns:
                plot_df = plot_df[plot_df['Device_ID'].isin(device_ids)]
            else:
                logger.warning("Device_ID column not found, ignoring device filter")
        
        # Default to AQI or PM2.5 if parameters not specified
        if not parameters:
            if 'AQI' in plot_df.columns:
                parameters = ['AQI']
            elif 'PM2.5' in plot_df.columns:
                parameters = ['PM2.5']
            else:
                # Use first available numeric column
                numeric_cols = plot_df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    parameters = [numeric_cols[0]]
                else:
                    logger.error("No numeric columns available for time series plot")
                    return None
        
        # Create figure
        fig = go.Figure()
        
        # Group by device if multiple devices
        has_multiple_devices = 'Device_ID' in plot_df.columns and plot_df['Device_ID'].nunique() > 1
        
        if has_multiple_devices:
            # Sort data by datetime
            plot_df = plot_df.sort_values('Datetime')
            
            for device in plot_df['Device_ID'].unique():
                device_data = plot_df[plot_df['Device_ID'] == device]
                
                for param in parameters:
                    if param in device_data.columns:
                        # Apply rolling average if specified
                        if rolling_window and len(device_data) > rolling_window:
                            smoothed_data = device_data[param].rolling(window=rolling_window).mean()
                            fig.add_trace(
                                go.Scatter(
                                    x=device_data['Datetime'],
                                    y=smoothed_data,
                                    mode='lines',
                                    name=f"{device} - {param} (Smoothed)",
                                    hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
                                )
                            )
                        else:
                            fig.add_trace(
                                go.Scatter(
                                    x=device_data['Datetime'],
                                    y=device_data[param],
                                    mode='lines',
                                    name=f"{device} - {param}",
                                    hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
                                )
                            )
                            
                        # Add trend line if requested
                        if add_trend and len(device_data) > 2:
                            x_numeric = np.arange(len(device_data))
                            y_values = device_data[param].values
                            mask = ~np.isnan(y_values)
                            if np.sum(mask) > 1:  # Need at least 2 points for a line
                                coeffs = np.polyfit(x_numeric[mask], y_values[mask], 1)
                                trend_line = coeffs[0] * x_numeric + coeffs[1]
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=device_data['Datetime'],
                                        y=trend_line,
                                        mode='lines',
                                        line=dict(dash='dash', width=1),
                                        name=f"{device} - {param} Trend",
                                        hoverinfo='skip'
                                    )
                                )
        else:
            # Sort data by datetime
            plot_df = plot_df.sort_values('Datetime')
            
            for param in parameters:
                if param in plot_df.columns:
                    # Apply rolling average if specified
                    if rolling_window and len(plot_df) > rolling_window:
                        smoothed_data = plot_df[param].rolling(window=rolling_window).mean()
                        fig.add_trace(
                            go.Scatter(
                                x=plot_df['Datetime'],
                                y=smoothed_data,
                                mode='lines',
                                name=f"{param} (Smoothed)",
                                hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
                            )
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=plot_df['Datetime'],
                                y=plot_df[param],
                                mode='lines',
                                name=f"{param}",
                                hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
                            )
                        )
                        
                    # Add trend line if requested
                    if add_trend and len(plot_df) > 2:
                        x_numeric = np.arange(len(plot_df))
                        y_values = plot_df[param].values
                        mask = ~np.isnan(y_values)
                        if np.sum(mask) > 1:  # Need at least 2 points for a line
                            coeffs = np.polyfit(x_numeric[mask], y_values[mask], 1)
                            trend_line = coeffs[0] * x_numeric + coeffs[1]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=plot_df['Datetime'],
                                    y=trend_line,
                                    mode='lines',
                                    line=dict(dash='dash', width=1),
                                    name=f"{param} Trend",
                                    hoverinfo='skip'
                                )
                            )
        
        # Update layout
        chart_title = "Time Series: " + ", ".join(parameters)
        if device_ids and len(device_ids) < 4:  # Only add devices to title if not too many
            chart_title += f" ({', '.join(device_ids)})"
            
        y_axis_title = parameters[0] if len(parameters) == 1 else "Value"
        
        fig.update_layout(
            title=chart_title,
            xaxis_title="Date & Time",
            yaxis_title=y_axis_title,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        # Add AQI color bands if plotting AQI
        if len(parameters) == 1 and parameters[0] == 'AQI':
            # AQI color bands
            aqi_bands = [
                {"name": "Good", "min": 0, "max": 50, "color": "rgba(0, 228, 0, 0.2)"},
                {"name": "Moderate", "min": 51, "max": 100, "color": "rgba(255, 255, 0, 0.2)"},
                {"name": "Unhealthy for Sensitive Groups", "min": 101, "max": 150, "color": "rgba(255, 126, 0, 0.2)"},
                {"name": "Unhealthy", "min": 151, "max": 200, "color": "rgba(255, 0, 0, 0.2)"},
                {"name": "Very Unhealthy", "min": 201, "max": 300, "color": "rgba(143, 63, 151, 0.2)"},
                {"name": "Hazardous", "min": 301, "max": 500, "color": "rgba(126, 0, 35, 0.2)"}
            ]
            
            for band in aqi_bands:
                fig.add_shape(
                    type="rect",
                    x0=plot_df['Datetime'].min(),
                    x1=plot_df['Datetime'].max(),
                    y0=band["min"],
                    y1=band["max"],
                    fillcolor=band["color"],
                    line=dict(width=0),
                    layer="below",
                    name=band["name"]
                )
                
            # Add annotations for AQI categories
            y_pos = plot_df[parameters[0]].max() * 0.9
            x_pos = plot_df['Datetime'].min()
            
            for band in aqi_bands:
                fig.add_annotation(
                    x=x_pos,
                    y=((band["min"] + band["max"]) / 2),
                    text=band["name"],
                    showarrow=False,
                    font=dict(size=10),
                    align="left"
                )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating time series plot: {str(e)}")
        return None

def create_aqi_distribution_chart(df):
    """
    Create a stacked bar chart showing the distribution of AQI categories.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        A plotly figure object containing the AQI distribution chart
    """
    try:
        if df is None or df.empty or 'AQI' not in df.columns:
            logger.error("Cannot create AQI distribution chart: AQI data not available")
            return None
            
        # Define AQI categories and colors
        aqi_categories = [
            {"name": "Good", "min": 0, "max": 50, "color": "#00e400"},
            {"name": "Moderate", "min": 51, "max": 100, "color": "#ffff00"},
            {"name": "Unhealthy for Sensitive Groups", "min": 101, "max": 150, "color": "#ff7e00"},
            {"name": "Unhealthy", "min": 151, "max": 200, "color": "#ff0000"},
            {"name": "Very Unhealthy", "min": 201, "max": 300, "color": "#8f3f97"},
            {"name": "Hazardous", "min": 301, "max": 500, "color": "#7e0023"}
        ]
        
        # Categorize data
        df_copy = df.copy()
        df_copy['AQI_Category'] = pd.cut(
            df_copy['AQI'],
            bins=[cat["min"] for cat in aqi_categories] + [float('inf')],
            labels=[cat["name"] for cat in aqi_categories],
            right=False
        )
        
        # Count by category
        category_counts = df_copy['AQI_Category'].value_counts().sort_index()
        
        # Create custom color map matching the categories
        color_map = {cat["name"]: cat["color"] for cat in aqi_categories}
        ordered_colors = [color_map[cat] for cat in category_counts.index]
        
        # Calculate percentages
        total = category_counts.sum()
        category_percents = (category_counts / total * 100).round(1)
        
        # Create combined labels
        labels = [f"{cat} ({count} samples, {pct}%)" 
                for cat, count, pct in zip(category_counts.index, category_counts, category_percents)]
        
        # Create bar chart
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            color=category_counts.index,
            color_discrete_map=color_map,
            title="Air Quality Index (AQI) Distribution",
            labels={'x': 'AQI Category', 'y': 'Count'},
            text=category_percents.apply(lambda x: f"{x:.1f}%")
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title=None,
            yaxis_title="Number of Readings",
            legend_title=None,
            showlegend=False
        )
        
        # Update traces to show percentages on top of bars
        fig.update_traces(
            textposition='outside',
            hovertemplate='%{x}<br>Count: %{y}<br>Percentage: %{text}<extra></extra>'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating AQI distribution chart: {str(e)}")
        return None

def create_pollutant_comparison(df, pollutants=None):
    """
    Create a grouped bar chart comparing average pollutant levels across devices.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    pollutants : list, optional
        List of pollutant names to compare
        
    Returns:
    --------
    plotly.graph_objects.Figure
        A plotly figure object containing the pollutant comparison chart
    """
    try:
        if df is None or df.empty or 'Device_ID' not in df.columns:
            logger.error("Cannot create pollutant comparison: Invalid data or no device information")
            return None
            
        # Default pollutants if not specified
        if not pollutants:
            all_pollutants = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
            pollutants = all_pollutants[:min(3, len(all_pollutants))]  # Use up to 3 pollutants by default
            
            if not pollutants:
                logger.error("No pollutant data available for comparison")
                return None
        
        # Calculate average by device
        device_pollutants = df.groupby('Device_ID')[pollutants].mean().reset_index()
        
        # Create the plot
        fig = px.bar(
            device_pollutants, 
            x='Device_ID', 
            y=pollutants,
            barmode='group',
            title="Average Pollutant Levels by Device",
            labels={'value': 'Concentration', 'variable': 'Pollutant'}
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating pollutant comparison chart: {str(e)}")
        return None

def create_daily_pattern_chart(df, parameter):
    """
    Create a chart showing daily patterns for a specific parameter.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    parameter : str
        The parameter to analyze (e.g., 'PM2.5', 'AQI')
        
    Returns:
    --------
    plotly.graph_objects.Figure
        A plotly figure object containing the daily pattern chart
    """
    try:
        if df is None or df.empty or 'Datetime' not in df.columns or parameter not in df.columns:
            logger.error(f"Cannot create daily pattern chart: Missing data or {parameter} column")
            return None
            
        # Extract hour of day
        df_copy = df.copy()
        df_copy['Hour'] = pd.to_datetime(df_copy['Datetime']).dt.hour
        
        # Calculate hourly average
        hourly_avg = df_copy.groupby('Hour')[parameter].mean().reset_index()
        
        # Create line chart
        fig = px.line(
            hourly_avg,
            x='Hour',
            y=parameter,
            title=f"Daily Pattern: {parameter} by Hour of Day",
            markers=True
        )
        
        # Update x-axis to show all hours
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(24)),
                ticktext=[f"{h:02d}:00" for h in range(24)]
            ),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified"
        )
        
        # Add area under the curve
        fig.add_trace(
            go.Scatter(
                x=hourly_avg['Hour'],
                y=hourly_avg[parameter],
                fill='tozeroy',
                fillcolor='rgba(75, 176, 81, 0.2)',
                line=dict(color='rgba(75, 176, 81, 1)'),
                name=parameter,
                hovertemplate='<b>%{x}:00</b><br>%{y:.2f}<extra></extra>'
            )
        )
        
        # Remove original trace
        fig.data = [fig.data[1]]
        
        # Add annotations for peak and low periods
        peak_hour = hourly_avg.loc[hourly_avg[parameter].idxmax()]
        low_hour = hourly_avg.loc[hourly_avg[parameter].idxmin()]
        
        fig.add_annotation(
            x=peak_hour['Hour'],
            y=peak_hour[parameter],
            text=f"Peak: {peak_hour[parameter]:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30
        )
        
        fig.add_annotation(
            x=low_hour['Hour'],
            y=low_hour[parameter],
            text=f"Low: {low_hour[parameter]:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=30
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating daily pattern chart: {str(e)}")
        return None

def create_pollutant_box_plot(df, pollutants=None):
    """
    Create box plots for pollutant distributions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    pollutants : list, optional
        List of pollutant names to include
        
    Returns:
    --------
    plotly.graph_objects.Figure
        A plotly figure object containing the box plots
    """
    try:
        if df is None or df.empty:
            logger.error("Cannot create pollutant box plot: Invalid data")
            return None
            
        # Default pollutants if not specified
        if not pollutants:
            all_pollutants = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
            pollutants = all_pollutants[:min(6, len(all_pollutants))]  # Use up to 6 pollutants by default
            
            if not pollutants:
                logger.error("No pollutant data available for box plot")
                return None
        
        # Create a long-format DataFrame for box plot
        plot_data = []
        for pollutant in pollutants:
            if pollutant in df.columns:
                # Get the data for this pollutant
                values = df[pollutant].dropna().tolist()
                pollutant_data = [{'Pollutant': pollutant, 'Value': value} for value in values]
                plot_data.extend(pollutant_data)
        
        if not plot_data:
            logger.error("No valid data for box plot")
            return None
            
        plot_df = pd.DataFrame(plot_data)
        
        # Create box plot
        fig = px.box(
            plot_df,
            x='Pollutant',
            y='Value',
            color='Pollutant',
            title="Pollutant Distributions",
            points="outliers"
        )
        
        # Update layout
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
            yaxis_title="Concentration"
        )
        
        # Add explanatory annotation
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            text="Box plots show the median (line), interquartile range (box), and range (whiskers) of pollutant values.",
            showarrow=False,
            font=dict(size=10),
            align="center"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating pollutant box plot: {str(e)}")
        return None

def create_weekly_pattern_chart(df, parameter):
    """
    Create a chart showing weekly patterns for a specific parameter.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    parameter : str
        The parameter to analyze (e.g., 'PM2.5', 'AQI')
        
    Returns:
    --------
    plotly.graph_objects.Figure
        A plotly figure object containing the weekly pattern chart
    """
    try:
        if df is None or df.empty or 'Datetime' not in df.columns or parameter not in df.columns:
            logger.error(f"Cannot create weekly pattern chart: Missing data or {parameter} column")
            return None
            
        # Extract day of week
        df_copy = df.copy()
        df_copy['Day'] = pd.to_datetime(df_copy['Datetime']).dt.dayofweek
        
        # Map day numbers to names
        day_names = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        df_copy['DayName'] = df_copy['Day'].map(day_names)
        
        # Calculate daily average
        daily_avg = df_copy.groupby('DayName')[parameter].mean().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ).reset_index()
        
        # Create bar chart
        fig = px.bar(
            daily_avg,
            x='DayName',
            y=parameter,
            title=f"Weekly Pattern: {parameter} by Day of Week",
            color=parameter,
            color_continuous_scale='Viridis'
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title=None,
            coloraxis_showscale=False,
            hovermode="x unified"
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
        )
        
        # Add weekday/weekend annotation
        weekday_avg = daily_avg[daily_avg['DayName'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])][parameter].mean()
        weekend_avg = daily_avg[daily_avg['DayName'].isin(['Saturday', 'Sunday'])][parameter].mean()
        difference = ((weekend_avg - weekday_avg) / weekday_avg) * 100
        
        annotation_text = f"Weekend {parameter} is "
        if abs(difference) < 5:
            annotation_text += f"similar to weekdays (Δ = {difference:.1f}%)"
        elif difference > 0:
            annotation_text += f"{difference:.1f}% higher than weekdays"
        else:
            annotation_text += f"{abs(difference):.1f}% lower than weekdays"
            
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            text=annotation_text,
            showarrow=False,
            font=dict(size=12),
            align="center"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating weekly pattern chart: {str(e)}")
        return None

def create_forecast_chart(df, forecast_data, parameter):
    """
    Create a chart showing historical data and forecast for a parameter.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The historical air quality data
    forecast_data : dict
        Forecast data from AI insights
    parameter : str
        The parameter to plot (e.g., 'PM2.5', 'AQI')
        
    Returns:
    --------
    plotly.graph_objects.Figure
        A plotly figure object containing the forecast chart
    """
    try:
        if (df is None or df.empty or 'Datetime' not in df.columns or parameter not in df.columns or
                forecast_data is None or parameter not in forecast_data):
            logger.error(f"Cannot create forecast chart: Missing data for {parameter}")
            return None
            
        # Get forecast data
        forecast = forecast_data[parameter]
        
        # Format timestamps from forecast
        forecast_timestamps = [pd.to_datetime(ts) for ts in forecast['timestamps']]
        forecast_values = forecast['values']
        
        # Get historical data (last 7 days)
        df_copy = df.copy()
        df_copy['Datetime'] = pd.to_datetime(df_copy['Datetime'])
        df_copy = df_copy.sort_values('Datetime')
        
        # Get last week of historical data
        last_date = df_copy['Datetime'].max()
        week_ago = last_date - pd.Timedelta(days=7)
        historical_data = df_copy[df_copy['Datetime'] >= week_ago]
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=historical_data['Datetime'],
                y=historical_data[parameter],
                mode='lines',
                name='Historical Data',
                line=dict(color='rgba(31, 119, 180, 0.8)', width=2),
                hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
            )
        )
        
        # Add forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_timestamps,
                y=forecast_values,
                mode='lines',
                name='Forecast',
                line=dict(color='rgba(255, 127, 14, 0.8)', width=2, dash='dot'),
                hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
            )
        )
        
        # Add confidence interval if available
        if 'confidence' in forecast and forecast['confidence'] in ['medium', 'low']:
            # Create a simple uncertainty band (±15% for medium confidence, ±25% for low)
            uncertainty = 0.15 if forecast['confidence'] == 'medium' else 0.25
            upper_bound = [val * (1 + uncertainty) for val in forecast_values]
            lower_bound = [val * (1 - uncertainty) for val in forecast_values]
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_timestamps + forecast_timestamps[::-1],
                    y=upper_bound + lower_bound[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255, 127, 14, 0)'),
                    name='Confidence Interval',
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        
        # Add vertical line at current time
        fig.add_shape(
            type="line",
            x0=last_date,
            x1=last_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="black", width=1, dash="dash"),
        )
        
        fig.add_annotation(
            x=last_date,
            y=0.01,
            yref="paper",
            text="Now",
            showarrow=False,
            textangle=-90,
            xanchor="left"
        )
        
        # Update layout
        fig.update_layout(
            title=f"{parameter} Forecast",
            xaxis_title="Date & Time",
            yaxis_title=parameter,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating forecast chart: {str(e)}")
        return None
