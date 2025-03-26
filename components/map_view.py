import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)

def render_map_view(df):
    """
    Render the map view showing geographical distribution of air quality data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    """
    try:
        if df is None or df.empty:
            st.error("No data available for map view. Please upload data or use sample data.")
            return
        
        # Check if location data is available
        if not all(col in df.columns for col in ['Latitude', 'Longitude']):
            st.error("Location data (Latitude, Longitude) not found in the dataset. Map view cannot be displayed.")
            return
        
        st.header("Air Quality Map View", divider="green")
        
        # Map controls
        controls_container = st.container()
        with controls_container:
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                # Select parameter to display on map
                available_params = [col for col in df.columns if col in 
                                  ['AQI', 'PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 
                                   'Temperature', 'Humidity', 'Pressure']]
                
                default_param = 'AQI' if 'AQI' in available_params else 'PM2.5' if 'PM2.5' in available_params else available_params[0]
                
                map_parameter = st.selectbox(
                    "Select Parameter to Display",
                    options=available_params,
                    index=available_params.index(default_param) if default_param in available_params else 0
                )
            
            with col2:
                # Map style
                map_style = st.selectbox(
                    "Map Style",
                    options=["Open Street Map", "Carto", "White Background", "Satellite"],
                    index=0
                )
                
                # Convert to plotly mapbox style
                mapbox_style_dict = {
                    "Open Street Map": "open-street-map",
                    "Carto": "carto-positron",
                    "White Background": "white-bg",
                    "Satellite": "satellite"
                }
                mapbox_style = mapbox_style_dict[map_style]
            
            with col3:
                # Time aggregation
                if 'Datetime' in df.columns:
                    time_options = ["Latest", "Daily Average", "All Data"]
                    time_selection = st.selectbox(
                        "Time Selection",
                        options=time_options,
                        index=0
                    )
                else:
                    time_selection = "All Data"  # Default if no datetime
        
        # Process data for map
        map_data = prepare_map_data(df, map_parameter, time_selection)
        
        # Create map
        map_container = st.container()
        with map_container:
            if map_data is not None and not map_data.empty:
                map_fig = create_map_visualization(map_data, map_parameter, mapbox_style)
                st.plotly_chart(map_fig, use_container_width=True)
            else:
                st.error("Unable to create map. Please check your data.")
        
        # Device locations table
        devices_container = st.container()
        with devices_container:
            st.subheader("Monitoring Locations")
            
            if 'Device_ID' in map_data.columns:
                # Group by device and get latest location
                device_locations = map_data.groupby('Device_ID').agg({
                    'Latitude': 'mean',
                    'Longitude': 'mean',
                    map_parameter: 'mean'
                }).reset_index()
                
                # Add a color column based on parameter value
                device_locations['Color'] = categorize_value(device_locations[map_parameter], map_parameter)
                
                # Create a dataframe for display
                display_df = device_locations[['Device_ID', 'Latitude', 'Longitude', map_parameter, 'Color']].copy()
                
                # Format columns for better display
                st.dataframe(
                    display_df,
                    column_config={
                        "Device_ID": st.column_config.TextColumn("Device ID"),
                        "Latitude": st.column_config.NumberColumn("Latitude", format="%.5f"),
                        "Longitude": st.column_config.NumberColumn("Longitude", format="%.5f"),
                        map_parameter: st.column_config.NumberColumn(map_parameter, format="%.2f"),
                        "Color": st.column_config.Column("Status", 
                                                      help="Color-coded status based on parameter value")
                    },
                    hide_index=True,
                    use_container_width=True
                )
        
        # Additional visualizations
        viz_container = st.container()
        with viz_container:
            st.subheader("Geographical Analysis", divider="green")
            
            # Tabs for different visualizations
            tab1, tab2 = st.tabs(["Parameter Distribution by Location", "Spatial Clustering"])
            
            with tab1:
                # Bar chart showing parameter value by location
                if 'Device_ID' in map_data.columns:
                    location_dist_fig = create_location_distribution_chart(map_data, map_parameter)
                    if location_dist_fig:
                        st.plotly_chart(location_dist_fig, use_container_width=True)
                    else:
                        st.info("Unable to create location distribution chart.")
            
            with tab2:
                # Clustering of locations by parameter values
                if len(map_data) > 5 and 'Device_ID' in map_data.columns:
                    cluster_fig = create_spatial_cluster_chart(map_data, map_parameter)
                    if cluster_fig:
                        st.plotly_chart(cluster_fig, use_container_width=True)
                    else:
                        st.info("Unable to create spatial clustering chart. Insufficient data points.")
                else:
                    st.info("Spatial clustering requires at least 6 data points.")
        
    except Exception as e:
        logger.error(f"Error rendering map view: {str(e)}")
        st.error(f"An error occurred while rendering the map view: {str(e)}")

def prepare_map_data(df, parameter, time_selection):
    """
    Prepare data for map visualization based on time selection.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    parameter : str
        Parameter to display on map
    time_selection : str
        Time selection option ('Latest', 'Daily Average', 'All Data')
        
    Returns:
    --------
    pandas.DataFrame
        Processed data for map visualization
    """
    try:
        # Make a copy to avoid modifying original
        map_data = df.copy()
        
        # Ensure parameter exists
        if parameter not in map_data.columns:
            logger.error(f"Parameter {parameter} not found in data")
            return None
        
        # Process based on time selection
        if time_selection == "Latest" and 'Datetime' in map_data.columns:
            # Get the latest data for each device
            map_data['Datetime'] = pd.to_datetime(map_data['Datetime'])
            
            # If Device_ID exists, get latest by device
            if 'Device_ID' in map_data.columns:
                # Group by device and get the latest records
                latest_times = map_data.groupby('Device_ID')['Datetime'].max().reset_index()
                map_data = pd.merge(
                    map_data, 
                    latest_times, 
                    on=['Device_ID', 'Datetime'], 
                    how='inner'
                )
            else:
                # Get the overall latest time
                latest_time = map_data['Datetime'].max()
                map_data = map_data[map_data['Datetime'] == latest_time]
                
        elif time_selection == "Daily Average" and 'Datetime' in map_data.columns:
            # Calculate daily average for each location
            map_data['Date'] = pd.to_datetime(map_data['Datetime']).dt.date
            
            # If Device_ID exists, group by device and date
            if 'Device_ID' in map_data.columns:
                map_data = map_data.groupby(['Device_ID', 'Date', 'Latitude', 'Longitude']).agg({
                    parameter: 'mean'
                }).reset_index()
            else:
                # Group by location and date
                map_data = map_data.groupby(['Date', 'Latitude', 'Longitude']).agg({
                    parameter: 'mean'
                }).reset_index()
        
        # Ensure data has required columns
        required_cols = ['Latitude', 'Longitude', parameter]
        if not all(col in map_data.columns for col in required_cols):
            logger.error(f"Required columns missing in processed data")
            return None
        
        return map_data
        
    except Exception as e:
        logger.error(f"Error preparing map data: {str(e)}")
        return None

def create_map_visualization(map_data, parameter, mapbox_style):
    """
    Create a map visualization of the air quality data.
    
    Parameters:
    -----------
    map_data : pandas.DataFrame
        Processed data for map visualization
    parameter : str
        Parameter to display on map
    mapbox_style : str
        Mapbox style for the map background
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with map visualization
    """
    try:
        # Determine color scale based on parameter
        if parameter == 'AQI':
            color_scale = [
                [0, '#00e400'],     # Green - Good
                [0.2, '#ffff00'],   # Yellow - Moderate
                [0.4, '#ff7e00'],   # Orange - Unhealthy for Sensitive Groups
                [0.6, '#ff0000'],   # Red - Unhealthy
                [0.8, '#8f3f97'],   # Purple - Very Unhealthy
                [1.0, '#7e0023']    # Maroon - Hazardous
            ]
            color_title = "AQI"
        elif parameter in ['PM2.5', 'PM10']:
            color_scale = 'YlOrRd'  # Yellow-Orange-Red
            color_title = f"{parameter} (µg/m³)"
        elif parameter in ['Temperature']:
            color_scale = 'RdYlBu_r'  # Red-Yellow-Blue reversed
            color_title = f"{parameter} (°C)"
        elif parameter in ['Humidity']:
            color_scale = 'Blues'  # Blues
            color_title = f"{parameter} (%)"
        else:
            color_scale = 'Viridis'  # Default colorscale
            color_title = parameter
        
        # Create figure
        fig = px.scatter_mapbox(
            map_data,
            lat='Latitude',
            lon='Longitude',
            color=parameter,
            size=parameter,
            size_max=15,
            zoom=10,
            color_continuous_scale=color_scale,
            hover_name='Device_ID' if 'Device_ID' in map_data.columns else None,
            hover_data={
                'Latitude': ':.5f',
                'Longitude': ':.5f',
                parameter: ':.2f',
                'Date': True if 'Date' in map_data.columns else False,
                'Datetime': True if 'Datetime' in map_data.columns else False
            },
            title=f"{parameter} Geographical Distribution"
        )
        
        # Update layout
        fig.update_layout(
            mapbox_style=mapbox_style,
            height=600,
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title=color_title,
                tickfont=dict(size=12),
                titlefont=dict(size=14)
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating map visualization: {str(e)}")
        return None

def categorize_value(values, parameter):
    """
    Categorize parameter values for color coding.
    
    Parameters:
    -----------
    values : pandas.Series
        Series of values to categorize
    parameter : str
        Parameter name
        
    Returns:
    --------
    pandas.Series
        Series of categories
    """
    categories = []
    
    if parameter == 'AQI':
        for value in values:
            if value <= 50:
                categories.append("Good")
            elif value <= 100:
                categories.append("Moderate")
            elif value <= 150:
                categories.append("Unhealthy for Sensitive Groups")
            elif value <= 200:
                categories.append("Unhealthy")
            elif value <= 300:
                categories.append("Very Unhealthy")
            else:
                categories.append("Hazardous")
    
    elif parameter == 'PM2.5':
        for value in values:
            if value <= 12:
                categories.append("Good")
            elif value <= 35:
                categories.append("Moderate")
            elif value <= 55:
                categories.append("Unhealthy for Sensitive Groups")
            elif value <= 150:
                categories.append("Unhealthy")
            else:
                categories.append("Hazardous")
    
    elif parameter == 'PM10':
        for value in values:
            if value <= 54:
                categories.append("Good")
            elif value <= 154:
                categories.append("Moderate")
            elif value <= 254:
                categories.append("Unhealthy for Sensitive Groups")
            elif value <= 354:
                categories.append("Unhealthy")
            else:
                categories.append("Hazardous")
    
    elif parameter == 'O3':
        for value in values:
            if value <= 54:
                categories.append("Good")
            elif value <= 70:
                categories.append("Moderate")
            elif value <= 85:
                categories.append("Unhealthy for Sensitive Groups")
            elif value <= 105:
                categories.append("Unhealthy")
            else:
                categories.append("Hazardous")
    
    elif parameter == 'NO2':
        for value in values:
            if value <= 53:
                categories.append("Good")
            elif value <= 100:
                categories.append("Moderate")
            elif value <= 360:
                categories.append("Unhealthy for Sensitive Groups")
            elif value <= 649:
                categories.append("Unhealthy")
            else:
                categories.append("Hazardous")
    
    elif parameter == 'CO':
        for value in values:
            if value <= 4.4:
                categories.append("Good")
            elif value <= 9.4:
                categories.append("Moderate")
            elif value <= 12.4:
                categories.append("Unhealthy for Sensitive Groups")
            elif value <= 15.4:
                categories.append("Unhealthy")
            else:
                categories.append("Hazardous")
    
    elif parameter == 'SO2':
        for value in values:
            if value <= 35:
                categories.append("Good")
            elif value <= 75:
                categories.append("Moderate")
            elif value <= 185:
                categories.append("Unhealthy for Sensitive Groups")
            elif value <= 304:
                categories.append("Unhealthy")
            else:
                categories.append("Hazardous")
    
    elif parameter == 'Temperature':
        for value in values:
            if value <= 10:
                categories.append("Cold")
            elif value <= 20:
                categories.append("Cool")
            elif value <= 30:
                categories.append("Warm")
            else:
                categories.append("Hot")
    
    elif parameter == 'Humidity':
        for value in values:
            if value <= 30:
                categories.append("Very Dry")
            elif value <= 50:
                categories.append("Dry")
            elif value <= 70:
                categories.append("Moderate")
            else:
                categories.append("Humid")
    
    else:
        # Generic categorization for other parameters
        for value in values:
            if value <= np.percentile(values, 25):
                categories.append("Low")
            elif value <= np.percentile(values, 50):
                categories.append("Medium-Low")
            elif value <= np.percentile(values, 75):
                categories.append("Medium-High")
            else:
                categories.append("High")
    
    return categories

def create_location_distribution_chart(map_data, parameter):
    """
    Create a bar chart showing parameter distribution by location.
    
    Parameters:
    -----------
    map_data : pandas.DataFrame
        Processed data for map visualization
    parameter : str
        Parameter to display
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with location distribution chart
    """
    try:
        if 'Device_ID' not in map_data.columns:
            return None
        
        # Group by device and calculate mean
        location_data = map_data.groupby('Device_ID')[parameter].mean().reset_index()
        
        # Sort by parameter value
        location_data = location_data.sort_values(parameter, ascending=False)
        
        # Create color based on parameter value
        location_data['Color'] = categorize_value(location_data[parameter], parameter)
        
        # Get color map for categories
        if parameter == 'AQI':
            color_map = {
                "Good": "#00e400",
                "Moderate": "#ffff00",
                "Unhealthy for Sensitive Groups": "#ff7e00",
                "Unhealthy": "#ff0000",
                "Very Unhealthy": "#8f3f97",
                "Hazardous": "#7e0023"
            }
        else:
            # Generate a color scale for other parameters
            colors = px.colors.sequential.Viridis
            categories = location_data['Color'].unique()
            color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
        
        # Create figure
        fig = px.bar(
            location_data,
            x='Device_ID',
            y=parameter,
            color='Color',
            color_discrete_map=color_map,
            title=f"{parameter} by Location",
            labels={'Device_ID': 'Device/Location', parameter: parameter}
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Device/Location",
            yaxis_title=parameter,
            legend_title="Status",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified"
        )
        
        # Add threshold lines for relevant parameters
        thresholds = {
            'AQI': [50, 100, 150, 200, 300],
            'PM2.5': [12, 35, 55, 150],
            'PM10': [54, 154, 254, 354],
            'O3': [54, 70, 85, 105],
            'NO2': [53, 100, 360, 649],
            'CO': [4.4, 9.4, 12.4, 15.4],
            'SO2': [35, 75, 185, 304]
        }
        
        if parameter in thresholds:
            for threshold in thresholds[parameter]:
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(location_data) - 0.5,
                    y0=threshold,
                    y1=threshold,
                    line=dict(
                        color="rgba(0, 0, 0, 0.5)",
                        width=1,
                        dash="dash",
                    )
                )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating location distribution chart: {str(e)}")
        return None

def create_spatial_cluster_chart(map_data, parameter):
    """
    Create a scatter plot showing spatial clustering of locations.
    
    Parameters:
    -----------
    map_data : pandas.DataFrame
        Processed data for map visualization
    parameter : str
        Parameter to display
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with spatial cluster chart
    """
    try:
        if len(map_data) < 5:
            return None
        
        # Group by device location if multiple records per device
        if 'Device_ID' in map_data.columns:
            location_data = map_data.groupby('Device_ID').agg({
                'Latitude': 'mean',
                'Longitude': 'mean',
                parameter: 'mean'
            }).reset_index()
        else:
            location_data = map_data[['Latitude', 'Longitude', parameter]].drop_duplicates()
        
        # Create color based on parameter value
        location_data['Category'] = categorize_value(location_data[parameter], parameter)
        
        # Create figure
        fig = px.scatter(
            location_data,
            x='Longitude',
            y='Latitude',
            color=parameter,
            size=parameter,
            hover_name='Device_ID' if 'Device_ID' in location_data.columns else None,
            hover_data={
                'Latitude': ':.5f',
                'Longitude': ':.5f',
                parameter: ':.2f',
                'Category': True
            },
            title=f"Spatial Distribution of {parameter}",
            labels={
                'Longitude': 'Longitude',
                'Latitude': 'Latitude'
            }
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="closest",
            xaxis=dict(tickformat='.4f'),
            yaxis=dict(tickformat='.4f')
        )
        
        # Add annotations for extreme values
        if len(location_data) > 2:
            max_val_idx = location_data[parameter].idxmax()
            max_val_loc = location_data.iloc[max_val_idx]
            
            fig.add_annotation(
                x=max_val_loc['Longitude'],
                y=max_val_loc['Latitude'],
                text=f"Highest: {max_val_loc[parameter]:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
            
            min_val_idx = location_data[parameter].idxmin()
            min_val_loc = location_data.iloc[min_val_idx]
            
            fig.add_annotation(
                x=min_val_loc['Longitude'],
                y=min_val_loc['Latitude'],
                text=f"Lowest: {min_val_loc[parameter]:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=40
            )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating spatial cluster chart: {str(e)}")
        return None
