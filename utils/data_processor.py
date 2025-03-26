import pandas as pd
import numpy as np
import logging
import os
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
    
    all_dataframes = []
    
    for file in uploaded_files:
        try:
            # Read file content into a BytesIO object
            file_content = BytesIO(file.getvalue())
            
            # Determine file type and read accordingly
            file_name = file.name.lower()
            
            if file_name.endswith('.csv'):
                # Try different encodings and delimiters for CSV files
                try:
                    df = pd.read_csv(file_content, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        # Reset pointer to the beginning of the file
                        file_content.seek(0)
                        df = pd.read_csv(file_content, encoding='latin1')
                    except:
                        # Reset pointer and try with different delimiter
                        file_content.seek(0)
                        df = pd.read_csv(file_content, encoding='utf-8', delimiter=';')
                        
            elif file_name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_content)
            else:
                logger.error(f"Unsupported file format: {file_name}")
                continue
                
            # Clean up column names - remove spaces, special chars, standardize
            df.columns = [clean_column_name(col) for col in df.columns]
            
            # Ensure DateTime column exists and is properly formatted
            df = standardize_datetime_column(df)
            
            # Ensure Device_ID column exists
            if 'Device_ID' not in df.columns and 'device_id' not in df.columns:
                # Try to find a column that might contain device IDs
                device_col = next((col for col in df.columns if 
                                  any(kw in col.lower() for kw in ['device', 'id', 'sensor', 'station'])), None)
                
                if device_col:
                    df = df.rename(columns={device_col: 'Device_ID'})
                else:
                    # If no device column found, use filename as device id
                    device_id = os.path.splitext(file.name)[0]
                    df['Device_ID'] = device_id
            elif 'device_id' in df.columns:
                df = df.rename(columns={'device_id': 'Device_ID'})
                
            # Standardize pollutant columns if they exist
            pollutant_columns = find_and_standardize_pollutant_columns(df)
            
            # Handle missing values
            df = handle_missing_values(df)
            
            # Remove duplicate records
            df = df.drop_duplicates()
            
            all_dataframes.append(df)
            logger.info(f"Successfully processed file: {file.name}")
            
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {str(e)}")
            continue
    
    if not all_dataframes:
        logger.error("No dataframes were successfully created")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Final processing
    combined_df = final_data_processing(combined_df)
    
    return combined_df

def load_sample_data():
    """
    Load sample air quality data for demonstration purposes.
    
    Returns:
    --------
    pandas.DataFrame
        Sample dataset with air quality measurements
    """
    try:
        # Create a synthetic dataset for demonstration
        # Base date for the time series
        base_date = datetime.now() - timedelta(days=30)
        
        # Number of days and hours per day for the sample data
        num_days = 30
        hours_per_day = 24
        
        # Create a time series
        dates = [base_date + timedelta(days=d, hours=h) 
                for d in range(num_days) 
                for h in range(hours_per_day)]
        
        # Create multiple devices
        device_ids = ['OIZOM001', 'OIZOM002', 'OIZOM003', 'OIZOM004']
        
        # Create location data (latitude/longitude)
        locations = {
            'OIZOM001': (19.0760, 72.8777),  # Mumbai
            'OIZOM002': (28.7041, 77.1025),  # Delhi
            'OIZOM003': (12.9716, 77.5946),  # Bangalore
            'OIZOM004': (22.5726, 88.3639)   # Kolkata
        }
        
        # Create the dataset
        rows = []
        
        for date in dates:
            for device_id in device_ids:
                # Base values for each pollutant
                pm25_base = 35.0  # μg/m³
                pm10_base = 65.0  # μg/m³
                no2_base = 40.0   # ppb
                co_base = 1.0     # ppm
                o3_base = 30.0    # ppb
                so2_base = 20.0   # ppb
                
                # Add time-of-day variation
                hour = date.hour
                # More pollution during morning and evening rush hours
                if 7 <= hour <= 10 or 17 <= hour <= 20:
                    time_factor = 1.5
                # Less pollution at night
                elif 0 <= hour <= 5:
                    time_factor = 0.7
                else:
                    time_factor = 1.0
                
                # Add day-of-week variation (weekends have less pollution)
                weekday = date.weekday()
                day_factor = 0.8 if weekday >= 5 else 1.0  # Weekend vs weekday
                
                # Add device-specific variation
                if device_id == 'OIZOM001':  # Mumbai - moderate pollution
                    device_factor = 1.2
                elif device_id == 'OIZOM002':  # Delhi - high pollution
                    device_factor = 1.8
                elif device_id == 'OIZOM003':  # Bangalore - low pollution
                    device_factor = 0.7
                else:  # Kolkata - variable pollution
                    device_factor = 1.3
                
                # Add random noise
                noise = lambda: np.random.normal(1, 0.15)
                
                # Calculate final values with variations
                pm25 = pm25_base * time_factor * day_factor * device_factor * noise()
                pm10 = pm10_base * time_factor * day_factor * device_factor * noise()
                no2 = no2_base * time_factor * day_factor * device_factor * noise()
                co = co_base * time_factor * day_factor * device_factor * noise()
                o3 = o3_base * (2 - time_factor) * day_factor * device_factor * noise()  # O3 tends to be higher in midday
                so2 = so2_base * time_factor * day_factor * device_factor * noise()
                
                # Get location
                lat, lon = locations[device_id]
                
                # Add some randomness to location (as if the device is moving slightly)
                lat += np.random.normal(0, 0.002)
                lon += np.random.normal(0, 0.002)
                
                # Add temperature, humidity, and pressure data
                temperature = 25 + 10 * np.sin(np.pi * hour / 12) * noise() * 0.3  # Daily temperature cycle
                humidity = 60 + 20 * np.cos(np.pi * hour / 12) * noise() * 0.3  # Daily humidity cycle
                pressure = 1013 + np.random.normal(0, 2)  # Atmospheric pressure in hPa
                
                # Calculate air quality index (simplified)
                aqi = calculate_aqi(pm25, pm10, no2, o3, co, so2)
                
                # Add the data point
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
        
        # Create the DataFrame
        df = pd.DataFrame(rows)
        
        # Ensure datetime column is properly formatted
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        logger.info("Sample data loaded successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error creating sample data: {str(e)}")
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
    # Replace spaces and special characters
    col_name = str(col_name).strip()
    
    # Common air quality parameter mappings
    replacements = {
        'pm2.5': 'PM2.5',
        'pm25': 'PM2.5',
        'pm 2.5': 'PM2.5',
        'pm_2.5': 'PM2.5',
        'pm10': 'PM10',
        'pm 10': 'PM10',
        'pm_10': 'PM10',
        'no2': 'NO2',
        'no_2': 'NO2',
        'so2': 'SO2',
        'so_2': 'SO2',
        'co2': 'CO2',
        'co_2': 'CO2',
        'co': 'CO',
        'o3': 'O3',
        'ozone': 'O3',
        'temp': 'Temperature',
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'rh': 'Humidity',
        'pressure': 'Pressure',
        'datetime': 'Datetime',
        'date_time': 'Datetime',
        'time': 'Datetime',
        'timestamp': 'Datetime',
        'latitude': 'Latitude',
        'lat': 'Latitude',
        'longitude': 'Longitude',
        'long': 'Longitude',
        'lon': 'Longitude',
        'device': 'Device_ID',
        'deviceid': 'Device_ID',
        'device_id': 'Device_ID',
        'station': 'Device_ID',
        'stationid': 'Device_ID',
        'station_id': 'Device_ID',
        'aqi': 'AQI',
        'air_quality_index': 'AQI'
    }
    
    # Check for matching keys in the replacements dictionary
    lower_col = col_name.lower()
    for key, value in replacements.items():
        if key == lower_col or key in lower_col:
            return value
    
    # If no match found, return the original with underscores replacing spaces
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
    # Identify potential datetime columns
    datetime_cols = [col for col in df.columns if 
                    any(kw in col.lower() for kw in ['time', 'date', 'datetime', 'timestamp'])]
    
    if 'Datetime' in df.columns:
        datetime_col = 'Datetime'
    elif datetime_cols:
        datetime_col = datetime_cols[0]
        df = df.rename(columns={datetime_col: 'Datetime'})
    else:
        # No datetime column found, return as is
        logger.warning("No datetime column found in the data")
        return df
    
    # Convert to datetime format
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    except Exception as e:
        logger.error(f"Error converting Datetime column: {str(e)}")
        # Try common datetime formats
        date_formats = [
            '%Y-%m-%d %H:%M:%S', 
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%d-%m-%Y %H:%M:%S',
            '%m-%d-%Y %H:%M:%S',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y'
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
        'pm2.5': 'PM2.5',
        'pm25': 'PM2.5',
        'pm10': 'PM10',
        'no2': 'NO2',
        'so2': 'SO2',
        'co': 'CO',
        'o3': 'O3'
    }
    
    pollutant_columns = []
    
    # Find and rename pollutant columns
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
    # Fill missing data based on column type
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # For numerical columns, use rolling mean with window size 3
            # This preserves trends better than a global mean
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].rolling(window=3, min_periods=1).mean())
                
                # If there are still NaNs (e.g., at the beginning), use forward/backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
                # If still NaNs, use global mean
                df[col] = df[col].fillna(df[col].mean())
        else:
            # For non-numeric columns, use the most frequent value
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    return df

def final_data_processing(df):
    """
    Perform final processing on the combined DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        Final processed DataFrame
    """
    # Sort by datetime
    if 'Datetime' in df.columns:
        df = df.sort_values('Datetime')
    
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
            # Simplified AQI based on PM2.5 only
            df['AQI'] = df['PM2.5'] * 4.2  # Simplified conversion factor
    
    # Convert column types if needed
    for col in df.columns:
        if col not in ['Datetime', 'Device_ID'] and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add timezone info if missing
    if 'Datetime' in df.columns and df['Datetime'].dt.tz is None:
        # Assume UTC if no timezone info
        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC')
    
    return df

def calculate_aqi(pm25, pm10, no2, o3, co, so2):
    """
    Calculate a simplified Air Quality Index based on multiple pollutants.
    This is a simplified version for demonstration.
    
    Parameters:
    -----------
    pm25, pm10, no2, o3, co, so2 : float
        Pollutant concentrations
        
    Returns:
    --------
    float
        Calculated AQI value
    """
    # AQI calculation is normally complex with breakpoints for each pollutant
    # This is a simplified version based predominantly on PM2.5 with influence from other pollutants
    
    # PM2.5 AQI (simplified)
    pm25_aqi = min(500, max(0, pm25 * 2.1))
    
    # PM10 AQI (simplified)
    pm10_aqi = min(500, max(0, pm10 * 1.0))
    
    # NO2 AQI (simplified)
    no2_aqi = min(500, max(0, no2 * 1.9))
    
    # O3 AQI (simplified)
    o3_aqi = min(500, max(0, o3 * 1.8))
    
    # CO AQI (simplified)
    co_aqi = min(500, max(0, co * 50))
    
    # SO2 AQI (simplified)
    so2_aqi = min(500, max(0, so2 * 2.1))
    
    # Take the maximum AQI value (as per standard practice)
    aqi = max(pm25_aqi, pm10_aqi, no2_aqi, o3_aqi, co_aqi, so2_aqi)
    
    return aqi
