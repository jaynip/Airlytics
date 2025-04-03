import pandas as pd
import numpy as np
import logging
import os
import re
import geohash2
from datetime import datetime, timedelta
import random
from io import BytesIO

# Set up logging
logger = logging.getLogger(__name__)

def upload_and_process_data(uploaded_files):
    """
    Process uploaded files (CSV or Excel) and convert them to a pandas DataFrame.
    
    Parameters:
    -----------
    uploaded_files : list
        List of uploaded files from st.file_uploader
        
    Returns:
    --------
    pandas.DataFrame
        Processed and cleaned DataFrame
    """
    if not uploaded_files:
        logger.warning("No files were uploaded")
        return None
    
    all_data = []
    all_columns = set()
    
    # Define device coordinates mapping (latitude, longitude)
    device_coordinates = {
        523005: (23.02909505, 72.49078965),
        523011: (23.0309101, 72.5088321),
        523047: (23.0692036, 72.5653925),
        523082: (23.0850114, 72.5751516),
        523093: (23.096915, 72.527362),
        524037: (23.04836488, 72.68863108),
        524046: (23.0428964, 72.4749039),
        524049: (23.0777287, 72.5056656),
        524062: (23.12348122, 72.53853052),
        524089: (23.02815923, 72.50001528),
        524091: (23.0087287, 72.4551301),
    }
    device_geohash = {device_id: geohash2.encode(lat, lon, precision=7) for device_id, (lat, lon) in device_coordinates.items()}
    
    for file in uploaded_files:
        try:
            file_name = file.name
            file_content = BytesIO(file.getvalue())
            
            # Load file based on extension
            file_extension = file_name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(file_content, encoding='utf-8-sig', delimiter=',', dtype={'To Date': str})
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(file_content, dtype={'To Date': str})
            else:
                logger.warning(f"⚠️ Unsupported file format: {file_name}")
                continue

            df.columns = df.columns.str.strip()
            logger.info(f"Loaded columns for {file_name}: {list(df.columns)}")

            # Datetime conversion
            if "To Date" in df.columns:
                df.rename(columns={"To Date": "Datetime"}, inplace=True)
                df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d-%m-%Y %H:%M", errors="coerce", dayfirst=True)
                if df["Datetime"].isna().any():
                    logger.warning(f"⚠️ Some Datetime values invalid in {file_name}; attempting fallback parsing.")
                    df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=True, errors="coerce")
                df["Time"] = df["Datetime"].dt.strftime('%H:%M:%S')
                if df["Datetime"].isna().all():
                    logger.error(f"❌ All Datetime values are NaT in {file_name}. Skipping.")
                    continue
                logger.info(f"✅ Datetime column converted successfully in {file_name}")
            else:
                logger.error(f"❌ 'To Date' column not found in {file_name}. Skipping.")
                continue

            # Rename columns
            column_mapping = {
                'PM₂.₅ (µg/m³)': 'PM2.5', 'PM₁₀ (µg/m³)': 'PM10', 'PM₁ (µg/m³ )': 'PM1',
                'PM₁₀₀ (µg/m³ )': 'PM100', 'R. Humidity (%)': 'Humidity',
                'Temperature (°C)': 'Temperature', 'wind direction (degree)': 'Wind_Direction',
                'wind speed (m/s)': 'Wind_Speed'
            }
            df.rename(columns=column_mapping, inplace=True)
            all_columns.update(df.columns)
            logger.info(f"Columns after renaming in {file_name}: {list(df.columns)}")

            # Extract Device_ID from filename (e.g., AQ0523005.csv -> 523005)
            device_id_match = re.search(r'AQ0(\d{6})', file_name)
            if not device_id_match:
                logger.warning(f"⚠️ Could not extract Device_ID from {file_name}. Using filename as Device_ID.")
                df['Device_ID'] = os.path.splitext(file_name)[0]
            else:
                device_id = int(device_id_match.group(1))
                df['Device_ID'] = device_id
                if device_id in device_coordinates:
                    df['Geohash'] = device_geohash[device_id]
                    logger.info(f"✅ Geohash assigned in {file_name}: {df['Geohash'].iloc[0]}")
                else:
                    logger.warning(f"⚠️ Device_ID {device_id} from {file_name} not in coordinates mapping. Geohash will be NaN.")
                    df['Geohash'] = pd.NA

            # Convert objects to numeric
            df.replace('-', pd.NA, inplace=True)
            for col in df.select_dtypes(include=['object']).columns:
                if col not in ['Time', 'Device_ID', 'Geohash']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Validate key columns
            if 'PM2.5' in df.columns:
                invalid_pm25 = df[(df['PM2.5'] < 0) | (df['PM2.5'] > 500)].index
                df.loc[invalid_pm25, 'PM2.5'] = pd.NA
                if len(invalid_pm25) > 0:
                    logger.warning(f"⚠️ {len(invalid_pm25)} rows with invalid PM2.5 values in {file_name}; set to NaN.")
            if 'Wind_Direction' in df.columns:
                invalid_wind = df[(df['Wind_Direction'] < 0) | (df['Wind_Direction'] > 360)].index
                df.loc[invalid_wind, 'Wind_Direction'] = pd.NA
                if len(invalid_wind) > 0:
                    logger.warning(f"⚠️ {len(invalid_wind)} rows with invalid Wind_Direction values in {file_name}; set to NaN.")

            # Fill missing values for continuous variables
            for col in ['Temperature', 'Humidity', 'Wind_Speed']:
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].ffill().bfill()

            # Fill calibration factors
            cf_columns = ['P1_CF', 'P2_CF', 'P3_CF', 'P4_CF', 'HUM_CF', 'TEMP_CF']
            for col in cf_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(100.0)

            all_data.append(df)
            logger.info(f"✅ Successfully processed file: {file_name}")
            
        except Exception as e:
            logger.error(f"❌ Error processing file {file.name}: {str(e)}")
            continue
    
    if all_data:
        # Ensure consistent columns across all DataFrames
        for i in range(len(all_data)):
            missing_cols = all_columns - set(all_data[i].columns)
            for col in missing_cols:
                all_data[i][col] = pd.NA
        
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by device ID and datetime
        if 'Datetime' in combined_df.columns and 'Device_ID' in combined_df.columns:
            combined_df.sort_values(by=['Device_ID', 'Datetime'], ascending=[True, True], inplace=True)
        
        # Check for devices without coordinates
        if 'Device_ID' in combined_df.columns:
            unknown_devices = set(combined_df['Device_ID'].unique()) - set(device_coordinates.keys())
            if unknown_devices:
                logger.warning(f"⚠️ Unknown devices without coordinates: {unknown_devices}")
        
        # Final processing
        combined_df = final_data_processing(combined_df)
        logger.info("✅ Data merged successfully.")
        return combined_df
    
    logger.error("❌ No valid data processed.")
    return None

def load_sample_data():
    """
    Load sample air quality data for demonstration purposes.
    
    Returns:
    --------
    pandas.DataFrame
        Sample dataset with air quality measurements
    """
    try:
        # Define device coordinates mapping (latitude, longitude)
        device_coordinates = {
            523005: (23.02909505, 72.49078965),
            523011: (23.0309101, 72.5088321),
            523047: (23.0692036, 72.5653925),
            523082: (23.0850114, 72.5751516),
            523093: (23.096915, 72.527362),
            524037: (23.04836488, 72.68863108),
            524046: (23.0428964, 72.4749039),
            524049: (23.0777287, 72.5056656),
            524062: (23.12348122, 72.53853052),
            524089: (23.02815923, 72.50001528),
            524091: (23.0087287, 72.4551301),
        }
        
        # Base date for the time series
        base_date = datetime.now() - timedelta(days=30)
        num_days = 30
        hours_per_day = 24
        
        # Create a time series
        dates = [base_date + timedelta(days=d, hours=h) 
                 for d in range(num_days) 
                 for h in range(hours_per_day)]
        
        # Use device IDs from coordinates
        device_ids = list(device_coordinates.keys())
        
        rows = []
        
        for date in dates:
            for device_id in device_ids:
                # Get fixed coordinates
                lat, lon = device_coordinates[device_id]
                
                # Base values for each pollutant
                pm25_base = 35.0  # μg/m³
                pm10_base = 65.0  # μg/m³
                no2_base = 40.0   # ppb
                co_base = 1.0     # ppm
                o3_base = 30.0    # ppb
                so2_base = 20.0   # ppb
                
                # Add time-of-day variation
                hour = date.hour
                if 7 <= hour <= 10 or 17 <= hour <= 20:  # Rush hours
                    time_factor = 1.5
                elif 0 <= hour <= 5:  # Night
                    time_factor = 0.7
                else:
                    time_factor = 1.0
                
                # Add day-of-week variation
                weekday = date.weekday()
                day_factor = 0.8 if weekday >= 5 else 1.0
                
                # Add device-specific variation
                if device_id in [523005, 524046]:  # Moderate pollution
                    device_factor = 1.2
                elif device_id == 523011:  # High pollution
                    device_factor = 1.8
                elif device_id == 523047:  # Low pollution
                    device_factor = 0.7
                else:  # Variable pollution
                    device_factor = 1.3
                
                # Add random noise
                noise = lambda: np.random.normal(1, 0.15)
                
                # Calculate final values
                pm25 = pm25_base * time_factor * day_factor * device_factor * noise()
                pm10 = pm10_base * time_factor * day_factor * device_factor * noise()
                no2 = no2_base * time_factor * day_factor * device_factor * noise()
                co = co_base * time_factor * day_factor * device_factor * noise()
                o3 = o3_base * (2 - time_factor) * day_factor * device_factor * noise()
                so2 = so2_base * time_factor * day_factor * device_factor * noise()
                
                # Add environmental data
                temperature = 25 + 10 * np.sin(np.pi * hour / 12) * noise() * 0.3
                humidity = 60 + 20 * np.cos(np.pi * hour / 12) * noise() * 0.3
                pressure = 1013 + np.random.normal(0, 2)
                
                # Calculate AQI
                aqi = calculate_aqi(pm25, pm10, no2, o3, co, so2)
                
                row = {
                    'Datetime': date,
                    'Device_ID': device_id,
                    'Latitude': lat,
                    'Longitude': lon,
                    'PM2.5': round(pm25, 1),
                    'PM10': round(pm10, 1),
                    'NO2': round(no2, 1),
                    'CO': round(co, 2),
                    'O3': round(o3, 1),
                    'SO2': round(so2, 1),
                    'Temperature': round(temperature, 1),
                    'Humidity': round(humidity, 1),
                    'Pressure': round(pressure, 1),
                    'AQI': round(aqi, 0)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        logger.info("✅ Sample data loaded successfully")
        return df
    
    except Exception as e:
        logger.error(f"❌ Error creating sample data: {str(e)}")
        return None

def clean_column_name(col_name):
    """
    Clean and standardize column names
    
    Parameters:
    -----------
    col_name : str
        Original column name
        
    Returns:
    --------
    str
        Cleaned column name
    """
    col_name = str(col_name).strip()
    replacements = {
        'pm2.5': 'PM2.5', 'pm25': 'PM2.5', 'pm 2.5': 'PM2.5', 'pm_2.5': 'PM2.5',
        'pm10': 'PM10', 'pm 10': 'PM10', 'pm_10': 'PM10',
        'no2': 'NO2', 'no_2': 'NO2',
        'so2': 'SO2', 'so_2': 'SO2',
        'co2': 'CO2', 'co_2': 'CO2', 'co': 'CO',
        'o3': 'O3', 'ozone': 'O3',
        'temp': 'Temperature', 'temperature': 'Temperature',
        'humidity': 'Humidity', 'rh': 'Humidity',
        'pressure': 'Pressure',
        'datetime': 'Datetime', 'date_time': 'Datetime', 'time': 'Datetime', 'timestamp': 'Datetime',
        'latitude': 'Latitude', 'lat': 'Latitude',
        'longitude': 'Longitude', 'long': 'Longitude', 'lon': 'Longitude',
        'device': 'Device_ID', 'deviceid': 'Device_ID', 'device_id': 'Device_ID',
        'station': 'Device_ID', 'stationid': 'Device_ID', 'station_id': 'Device_ID',
        'aqi': 'AQI', 'air_quality_index': 'AQI'
    }
    lower_col = col_name.lower()
    for key, value in replacements.items():
        if key == lower_col or key in lower_col:
            return value
    return col_name.replace(' ', '_')

def standardize_datetime_column(df):
    """
    Ensure DataFrame has a correctly formatted DateTime column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with standardized DateTime column
    """
    datetime_cols = [col for col in df.columns if 
                    any(kw in col.lower() for kw in ['time', 'date', 'datetime', 'timestamp'])]
    
    if 'Datetime' in df.columns:
        datetime_col = 'Datetime'
    elif datetime_cols:
        datetime_col = datetime_cols[0]
        df = df.rename(columns={datetime_col: 'Datetime'})
    else:
        logger.warning("⚠️ No datetime column found in the data")
        return df
    
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    except Exception as e:
        logger.error(f"❌ Error converting Datetime column: {str(e)}")
        date_formats = [
            '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
            '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'
        ]
        for fmt in date_formats:
            try:
                df['Datetime'] = pd.to_datetime(df['Datetime'], format=fmt)
                break
            except:
                continue
    
    return df

def find_and_standardize_pollutant_columns(df):
    """
    Identify and standardize pollutant column names
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns:
    --------
    list
        List of standardized pollutant column names
    """
    pollutant_keywords = {
        'pm2.5': 'PM2.5', 'pm25': 'PM2.5', 'pm10': 'PM10',
        'no2': 'NO2', 'so2': 'SO2', 'co': 'CO', 'o3': 'O3'
    }
    pollutant_columns = []
    for col in df.columns:
        col_lower = col.lower()
        for keyword, standard_name in pollutant_keywords.items():
            if keyword in col_lower and standard_name not in df.columns:
                df.rename(columns={col: standard_name}, inplace=True)
                pollutant_columns.append(standard_name)
                break
    return pollutant_columns

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with handled missing values
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].rolling(window=3, min_periods=1).mean())
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            df[col] = df[col].fillna(df[col].mean())
        elif df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    return df

def final_data_processing(df):
    device_coordinates = {
        523005: (23.02909505, 72.49078965),
        523011: (23.0309101, 72.5088321),
        523047: (23.0692036, 72.5653925),
        523082: (23.0850114, 72.5751516),
        523093: (23.096915, 72.527362),
        524037: (23.04836488, 72.68863108),
        524046: (23.0428964, 72.4749039),
        524049: (23.0777287, 72.5056656),
        524062: (23.12348122, 72.53853052),
        524089: (23.02815923, 72.50001528),
        524091: (23.0087287, 72.4551301),
    }
    
    if 'Datetime' in df.columns:
        df = df.sort_values('Datetime')
    
    # Add Latitude and Longitude based on Device_ID
    if 'Device_ID' in df.columns:
        # Ensure Device_ID is integer
        df['Device_ID'] = pd.to_numeric(df['Device_ID'], errors='coerce').astype('Int64')
        
        # Add Latitude and Longitude
        df['Latitude'] = None
        df['Longitude'] = None
        for device_id, (lat, lon) in device_coordinates.items():
            mask = df['Device_ID'] == device_id
            if mask.any():
                df.loc[mask, 'Latitude'] = lat
                df.loc[mask, 'Longitude'] = lon
                logger.info(f"Assigned coordinates ({lat}, {lon}) to Device_ID {device_id}")
        
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        
        # Log rows with missing coordinates
        missing_coords = df[df['Latitude'].isna() | df['Longitude'].isna()]
        if not missing_coords.empty:
            logger.warning(f"⚠️ {len(missing_coords)} rows have missing Latitude/Longitude values.")
            logger.warning(f"Missing coordinates for Device_IDs: {missing_coords['Device_ID'].unique()}")
    
    # Calculate AQI if needed pollutant columns exist and AQI doesn't
    if 'AQI' not in df.columns:
        required_columns = ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']
        if all(col in df.columns for col in required_columns):
            df['AQI'] = df.apply(
                lambda row: calculate_aqi(
                    row['PM2.5'], row['PM10'], row['NO2'], 
                    row['O3'], row['CO'], row['SO2']
                ),
                axis=1
            )
        elif 'PM2.5' in df.columns:
            df['AQI'] = df['PM2.5'] * 4.2  # Simplified conversion factor
    
    # Convert column types
    for col in df.columns:
        if col not in ['Datetime', 'Device_ID'] and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add timezone info if missing
    if 'Datetime' in df.columns and df['Datetime'].dt.tz is None:
        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC')
    
    return df

def calculate_aqi(pm25, pm10, no2, o3, co, so2):
    """
    Calculate a simplified Air Quality Index based on multiple pollutants.
    
    Parameters:
    -----------
    pm25, pm10, no2, o3, co, so2 : float
        Pollutant concentrations
        
    Returns:
    --------
    float
        Calculated AQI value
    """
    pm25_aqi = min(500, max(0, pm25 * 2.1))
    pm10_aqi = min(500, max(0, pm10 * 1.0))
    no2_aqi = min(500, max(0, no2 * 1.9))
    o3_aqi = min(500, max(0, o3 * 1.8))
    co_aqi = min(500, max(0, co * 50))
    so2_aqi = min(500, max(0, so2 * 2.1))
    return max(pm25_aqi, pm10_aqi, no2_aqi, o3_aqi, co_aqi, so2_aqi)

# New Streamlit App Code Added Below
import pandas as pd
import streamlit as st

# Function to load latitude and longitude mappings from the mapping file
def load_mapping(mapping_file):
    try:
        mapping_df = pd.read_csv(mapping_file)
        if 'device' not in mapping_df.columns or 'latitude' not in mapping_df.columns or 'longitude' not in mapping_df.columns:
            st.error("Mapping file must contain 'device', 'latitude', and 'longitude' columns.")
            return None
        return mapping_df.set_index('device')[['latitude', 'longitude']]
    except Exception as e:
        st.error(f"Error loading mapping file: {e}")
        return None

# Function to process the uploaded file and merge latitude and longitude
def process_file(uploaded_file, mapping_df):
    try:
        # Detect file format and load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Ensure the 'device' column exists
        if 'device' not in df.columns:
            st.error("Uploaded file must contain a 'device' column.")
            return None
        
        # Merge data with mapping information
        df = df.merge(mapping_df, on='device', how='left')

        # Handling missing coordinates
        missing_count = df['latitude'].isna().sum()
        if missing_count > 0:
            st.warning(f"{missing_count} devices could not be assigned latitude/longitude.")
        
        return df

    except Exception as e:
        st.error(f"Error processing the file: {e}")
        return None

# Streamlit UI
st.title("Device Location Mapper")

# Upload mapping file
st.sidebar.header("Upload Mapping File")
mapping_file = st.sidebar.file_uploader("Upload CSV mapping file", type=['csv'])

if mapping_file:
    mapping_df = load_mapping(mapping_file)
    if mapping_df is not None:
        st.sidebar.success("Mapping file loaded successfully.")

        # Upload data file
        st.sidebar.header("Upload Data File")
        uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=['csv', 'xls', 'xlsx'])

        if uploaded_file:
            processed_df = process_file(uploaded_file, mapping_df)

            if processed_df is not None:
                st.success("File processed successfully!")
                st.write(processed_df)

                # Provide download link for processed file
                st.download_button(
                    label="Download Processed File",
                    data=processed_df.to_csv(index=False).encode('utf-8'),
                    file_name="processed_data.csv",
                    mime="text/csv"
                )