import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False)
def prepare_map_data(df, parameter, time_selection):
    """
    Prepare data for map visualization with caching for better performance
    """
    try:
        map_data = df.copy()
        
        if parameter not in map_data.columns:
            st.error(f"Selected parameter '{parameter}' not found in data columns.")
            logger.error(f"Parameter {parameter} not found in data")
            return None
            
        if not pd.api.types.is_numeric_dtype(map_data[parameter]):
            st.error(f"Selected parameter '{parameter}' must be numeric.")
            logger.error(f"Parameter {parameter} is not numeric")
            return None

        # Process based on time selection
        if time_selection == "Latest" and 'Datetime' in map_data.columns:
            map_data['Datetime'] = pd.to_datetime(map_data['Datetime'])
            if 'Device_ID' in map_data.columns:
                map_data = map_data.loc[map_data.groupby('Device_ID')['Datetime'].idxmax()]
            else:
                latest_time = map_data['Datetime'].max()
                map_data = map_data[map_data['Datetime'] == latest_time]
                
        elif time_selection == "Daily Average" and 'Datetime' in map_data.columns:
            map_data['Date'] = pd.to_datetime(map_data['Datetime']).dt.date
            if 'Device_ID' in map_data.columns:
                map_data = map_data.groupby(['Device_ID', 'Date', 'Latitude', 'Longitude'])[parameter].mean().reset_index()
            else:
                map_data = map_data.groupby(['Date', 'Latitude', 'Longitude'])[parameter].mean().reset_index()

        # Validate coordinates and values
        valid_coords = (
            map_data['Latitude'].between(-90, 90) & 
            map_data['Longitude'].between(-180, 180) &
            pd.notna(map_data['Latitude']) &
            pd.notna(map_data['Longitude']) &
            pd.notna(map_data[parameter])
        )
        
        if not valid_coords.any():
            st.error("No valid coordinates available for mapping after filtering.")
            return None
        
        return map_data[valid_coords][['Device_ID', 'Latitude', 'Longitude', parameter]].copy()
        
    except Exception as e:
        logger.error(f"Error preparing map data: {str(e)}", exc_info=True)
        st.error(f"Error processing map data: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def create_map_visualization(map_data, parameter):
    """
    Create a lightweight map visualization using scatter_mapbox
    """
    try:
        if map_data.empty:
            logger.error("Empty data provided for map visualization")
            return None

        # Log the data for debugging
        logger.info(f"Map data for visualization:\n{map_data[['Latitude', 'Longitude', parameter]].to_string()}")

        # Determine color scale based on parameter
        if parameter in ['AQI', 'ACI']:
            color_scale = [
                [0, 'green'],     # Good
                [0.2, 'yellow'],  # Moderate
                [0.4, 'orange'],  # Unhealthy for Sensitive Groups
                [0.6, 'red'],     # Unhealthy
                [0.8, 'purple'],  # Very Unhealthy
                [1.0, 'maroon']   # Hazardous
            ]
            color_title = parameter
        elif parameter in ['PM2.5', 'PM10']:
            color_scale = [
                [0, 'green'],     # Good
                [0.2, 'yellow'],  # Moderate
                [0.4, 'orange'],  # Unhealthy for Sensitive Groups
                [0.6, 'red'],     # Unhealthy
                [0.8, 'purple'],  # Very Unhealthy
                [1.0, 'maroon']   # Hazardous
            ]
            color_title = f"{parameter} (µg/m³)"
        elif parameter in ['Temperature']:
            color_scale = 'RdYlBu_r'
            color_title = f"{parameter} (°C)"
        elif parameter in ['Humidity']:
            color_scale = 'Blues'
            color_title = f"{parameter} (%)"
        else:
            color_scale = 'Viridis'
            color_title = parameter

        # Calculate the center of the map
        center_lat = map_data['Latitude'].mean()
        center_lon = map_data['Longitude'].mean()

        # Create the map using scatter_mapbox
        fig = px.scatter_mapbox(
            map_data,
            lat='Latitude',
            lon='Longitude',
            color=parameter,
            size=np.log1p(map_data[parameter]),
            size_max=15,
            zoom=10,
            center=dict(lat=center_lat, lon=center_lon),
            color_continuous_scale=color_scale,
            hover_name='Device_ID' if 'Device_ID' in map_data.columns else None,
            hover_data={
                'Latitude': False,
                'Longitude': False,
                parameter: True,
                'Device_ID': True if 'Device_ID' in map_data.columns else False
            },
            title=f"Air Quality Map - {parameter}"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            height=500,
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title=dict(text=color_title, font=dict(size=12)),
                tickfont=dict(size=10)
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating map visualization: {str(e)}", exc_info=True)
        st.error(f"Error generating map: {str(e)}")
        return None

def categorize_value(values, parameter):
    """
    Categorize parameter values for color coding.
    """
    categories = []
    for value in values:
        if parameter in ['AQI', 'ACI']:
            if value <= 50: categories.append("Good")
            elif value <= 100: categories.append("Moderate")
            elif value <= 150: categories.append("Unhealthy for Sensitive Groups")
            elif value <= 200: categories.append("Unhealthy")
            elif value <= 300: categories.append("Very Unhealthy")
            else: categories.append("Hazardous")
        elif parameter == 'PM2.5':
            if value <= 12: categories.append("Good")
            elif value <= 35: categories.append("Moderate")
            elif value <= 55: categories.append("Unhealthy for Sensitive Groups")
            elif value <= 150: categories.append("Unhealthy")
            else: categories.append("Hazardous")
        elif parameter == 'PM10':
            if value <= 54: categories.append("Good")
            elif value <= 154: categories.append("Moderate")
            elif value <= 254: categories.append("Unhealthy for Sensitive Groups")
            elif value <= 354: categories.append("Unhealthy")
            else: categories.append("Hazardous")
        elif parameter == 'O3':
            if value <= 54: categories.append("Good")
            elif value <= 70: categories.append("Moderate")
            elif value <= 85: categories.append("Unhealthy for Sensitive Groups")
            elif value <= 105: categories.append("Unhealthy")
            else: categories.append("Hazardous")
        elif parameter == 'NO2':
            if value <= 53: categories.append("Good")
            elif value <= 100: categories.append("Moderate")
            elif value <= 360: categories.append("Unhealthy for Sensitive Groups")
            elif value <= 649: categories.append("Unhealthy")
            else: categories.append("Hazardous")
        elif parameter == 'CO':
            if value <= 4.4: categories.append("Good")
            elif value <= 9.4: categories.append("Moderate")
            elif value <= 12.4: categories.append("Unhealthy for Sensitive Groups")
            elif value <= 15.4: categories.append("Unhealthy")
            else: categories.append("Hazardous")
        elif parameter == 'SO2':
            if value <= 35: categories.append("Good")
            elif value <= 75: categories.append("Moderate")
            elif value <= 185: categories.append("Unhealthy for Sensitive Groups")
            elif value <= 304: categories.append("Unhealthy")
            else: categories.append("Hazardous")
        elif parameter == 'Temperature':
            if value <= 10: categories.append("Cold")
            elif value <= 20: categories.append("Cool")
            elif value <= 30: categories.append("Warm")
            else: categories.append("Hot")
        elif parameter == 'Humidity':
            if value <= 30: categories.append("Very Dry")
            elif value <= 50: categories.append("Dry")
            elif value <= 70: categories.append("Moderate")
            else: categories.append("Humid")
        else:
            if value <= np.percentile(values, 25): categories.append("Low")
            elif value <= np.percentile(values, 50): categories.append("Medium-Low")
            elif value <= np.percentile(values, 75): categories.append("Medium-High")
            else: categories.append("High")
    return categories

def create_location_distribution_chart(map_data, parameter):
    """
    Create a bar chart showing parameter distribution by location.
    """
    try:
        if 'Device_ID' not in map_data.columns:
            return None
        
        location_data = map_data.groupby('Device_ID')[parameter].mean().reset_index()
        location_data = location_data.sort_values(parameter, ascending=False)
        location_data['Color'] = categorize_value(location_data[parameter], parameter)
        
        color_map = {
            "Good": "#00e400", "Moderate": "#ffff00", "Unhealthy for Sensitive Groups": "#ff7e00",
            "Unhealthy": "#ff0000", "Very Unhealthy": "#8f3f97", "Hazardous": "#7e0023"
        } if parameter in ['AQI', 'ACI'] else {
            cat: px.colors.sequential.Viridis[i % len(px.colors.sequential.Viridis)] 
            for i, cat in enumerate(location_data['Color'].unique())
        }
                                    
        fig = px.bar(
            location_data,
            x='Device_ID',
            y=parameter,
            color='Color',
            color_discrete_map=color_map,
            title=f"{parameter} by Location",
            labels={'Device_ID': 'Device/Location', parameter: parameter}
        )
        
        fig.update_layout(
            xaxis_title="Device/Location",
            yaxis_title=parameter,
            legend_title="Status",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified"
        )
        
        thresholds = {
            'AQI': [50, 100, 150, 200, 300], 
            'ACI': [50, 100, 150, 200, 300],
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
                    x1=len(location_data)-0.5,
                    y0=threshold,
                    y1=threshold,
                    line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dash")
                )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating location distribution chart: {str(e)}")
        return None

def create_spatial_cluster_chart(map_data, parameter):
    """
    Create a scatter plot showing spatial clustering of locations.
    """
    try:
        if len(map_data) < 3:
            return None
            
        location_data = (
            map_data.groupby('Device_ID')
            .agg({'Latitude': 'mean', 'Longitude': 'mean', parameter: 'mean'})
            .reset_index() if 'Device_ID' in map_data.columns 
            else map_data[['Latitude', 'Longitude', parameter]].drop_duplicates()
        )
                        
        location_data['Category'] = categorize_value(location_data[parameter], parameter)
        
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
            labels={'Longitude': 'Longitude', 'Latitude': 'Latitude'}
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="closest",
            xaxis=dict(tickformat='.4f'),
            yaxis=dict(tickformat='.4f')
        )
        
        if len(location_data) > 2:
            max_val_loc = location_data.iloc[location_data[parameter].idxmax()]
            fig.add_annotation(
                x=max_val_loc['Longitude'],
                y=max_val_loc['Latitude'],
                text=f"Highest: {max_val_loc[parameter]:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
                             
            min_val_loc = location_data.iloc[location_data[parameter].idxmin()]
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

def render_map_view(df):
    """
    Main function to render the map view with all optimizations
    """
    try:
        # Early validation
        if df is None or df.empty:
            st.error("No data available for map view.")
            return

        if not all(col in df.columns for col in ['Latitude', 'Longitude']):
            st.error("Missing required location data (Latitude/Longitude).")
            st.write("Debug: DataFrame columns:", df.columns.tolist())
            st.write("Debug: First few rows of DataFrame:", df.head())
            return
            
        st.header("Air Quality Map View", divider="green")

        # Controls
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                available_params = [col for col in df.columns if col in 
                                  ['AQI', 'PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 
                                   'Temperature', 'Humidity', 'Pressure', 'ACI']]
                if not available_params:
                    st.error("No valid parameters found.")
                    return
                    
                map_parameter = st.selectbox(
                    "Select Parameter to Display",
                    options=available_params,
                    index=0
                )
            
            with col2:
                # Remove projection selection since we're using scatter_mapbox
                st.write("")  # Placeholder to maintain layout
            
            with col3:
                time_selection = st.selectbox(
                    "Time Selection",
                    options=["Latest", "Daily Average", "All Data"],
                    index=2 if 'Datetime' not in df.columns else 0
                )

        # Process data
        processed_data = prepare_map_data(df, map_parameter, time_selection)
        
        if processed_data is None or processed_data.empty:
            st.error("No valid data available after processing.")
            return

        # Debug: Display the processed data
        st.write("Debug: Processed data for map visualization:")
        st.write(processed_data[['Device_ID', 'Latitude', 'Longitude', map_parameter]])

        # Display map
        map_fig = create_map_visualization(processed_data, map_parameter)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.error("Failed to generate map visualization.")

        # Display monitoring locations
        with st.container():
            st.subheader("Monitoring Locations")
            if 'Device_ID' in processed_data.columns:
                display_data = processed_data[['Device_ID', 'Latitude', 'Longitude', map_parameter]].copy()
                display_data['Status'] = categorize_value(display_data[map_parameter], map_parameter)
                st.dataframe(
                    display_data,
                    column_config={
                        "Device_ID": "Device ID",
                        "Latitude": st.column_config.NumberColumn(format="%.5f"),
                        "Longitude": st.column_config.NumberColumn(format="%.5f"),
                        map_parameter: st.column_config.NumberColumn(format="%.2f"),
                        "Status": None
                    },
                    hide_index=True,
                    use_container_width=True
                )

        # Additional visualizations
        with st.container():
            st.subheader("Geographical Analysis", divider="green")
            tab1, tab2 = st.tabs(["Parameter Distribution", "Spatial Clustering"])
            
            with tab1:
                if 'Device_ID' in processed_data.columns:
                    fig = create_location_distribution_chart(processed_data, map_parameter)
                    if fig: st.plotly_chart(fig, use_container_width=True)
                    else: st.info("Unable to create distribution chart.")
            
            with tab2:
                if len(processed_data) >= 3:
                    fig = create_spatial_cluster_chart(processed_data, map_parameter)
                    if fig: st.plotly_chart(fig, use_container_width=True)
                    else: st.info("Unable to create clustering chart.")
                else:
                    st.info("Need at least 3 locations for clustering.")

    except Exception as e:
        logger.error(f"Map view error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")