import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import google.generativeai as genai

# Set up logging
logger = logging.getLogger(__name__)

def get_ai_recommendations(df):
    """
    Generate AI-powered insights and recommendations based on the air quality data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
        
    Returns:
    --------
    dict
        A dictionary containing various insights and recommendations
    """
    try:
        if df is None or df.empty:
            logger.error("Cannot generate AI insights: DataFrame is empty or None")
            return None
            
        # Initialize insights dictionary
        insights = {
            'general_insights': [],
            'pollutant_trends': {},
            'anomalies': [],
            'correlations': {},
            'recommendations': []
        }
        
        # Generate general insights
        insights['general_insights'] = generate_general_insights(df)
        
        # Analyze pollutant trends
        insights['pollutant_trends'] = analyze_pollutant_trends(df)
        
        # Detect anomalies
        insights['anomalies'] = detect_anomalies(df)
        
        # Find correlations between parameters
        insights['correlations'] = find_correlations(df)
        
        # Generate recommendations
        insights['recommendations'] = generate_recommendations(df, insights)
        
        # Add forecast data if datetime column is available
        if 'Datetime' in df.columns:
            insights['forecasts'] = generate_forecasts(df)
            
        # Generate health impact assessment
        insights['health_impact'] = assess_health_impact(df)
        
        # Generate device performance metrics
        insights['device_performance'] = assess_device_performance(df)
        
        logger.info("AI insights generated successfully")
        return insights
        
    except Exception as e:
        logger.error(f"Error generating AI insights: {str(e)}")
        # Return a basic structure with an error message
        return {
            'general_insights': ['Error generating AI insights'],
            'pollutant_trends': {},
            'anomalies': [],
            'correlations': {},
            'recommendations': ['Unable to provide recommendations due to an error in analysis']
        }

def generate_general_insights(df):
    """
    Generate general insights about the air quality data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
        
    Returns:
    --------
    list
        A list of general insights
    """
    insights = []
    
    try:
        # Get date range
        if 'Datetime' in df.columns:
            start_date = pd.to_datetime(df['Datetime']).min().strftime('%Y-%m-%d')
            end_date = pd.to_datetime(df['Datetime']).max().strftime('%Y-%m-%d')
            date_range = f"{start_date} to {end_date}"
            insights.append(f"Data spans from {date_range}")
            
            # Calculate data completeness
            total_days = (pd.to_datetime(df['Datetime']).max() - pd.to_datetime(df['Datetime']).min()).days + 1
            unique_days = df['Datetime'].dt.date.nunique()
            completeness = (unique_days / total_days) * 100
            insights.append(f"Data completeness: {completeness:.1f}% ({unique_days} out of {total_days} days)")
            
        # Get device count
        if 'Device_ID' in df.columns:
            device_count = df['Device_ID'].nunique()
            insights.append(f"Data collected from {device_count} devices")
            
        # AQI statistics if available
        if 'AQI' in df.columns:
            avg_aqi = df['AQI'].mean()
            max_aqi = df['AQI'].max()
            min_aqi = df['AQI'].min()
            
            # Categorize overall air quality
            if avg_aqi <= 50:
                quality = "Good"
            elif avg_aqi <= 100:
                quality = "Moderate"
            elif avg_aqi <= 150:
                quality = "Unhealthy for Sensitive Groups"
            elif avg_aqi <= 200:
                quality = "Unhealthy"
            elif avg_aqi <= 300:
                quality = "Very Unhealthy"
            else:
                quality = "Hazardous"
                
            insights.append(f"Overall air quality is {quality} (Average AQI: {avg_aqi:.1f})")
            insights.append(f"Maximum AQI recorded: {max_aqi:.1f}, Minimum AQI recorded: {min_aqi:.1f}")
            
        # Identify most problematic pollutants
        pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
        if pollutant_cols:
            # Calculate percentage of time each pollutant exceeds healthy thresholds
            thresholds = {
                'PM2.5': 35.0,  # μg/m³
                'PM10': 50.0,   # μg/m³
                'NO2': 100.0,   # ppb
                'O3': 70.0,     # ppb
                'CO': 9.0,      # ppm
                'SO2': 75.0     # ppb
            }
            
            exceedance_rates = {}
            for col in pollutant_cols:
                if col in thresholds and col in df.columns:
                    exceedance_rate = (df[col] > thresholds[col]).mean() * 100
                    exceedance_rates[col] = exceedance_rate
            
            if exceedance_rates:
                # Sort by exceedance rate
                sorted_rates = sorted(exceedance_rates.items(), key=lambda x: x[1], reverse=True)
                top_pollutant, top_rate = sorted_rates[0]
                
                if top_rate > 5:  # Only mention if it's significant
                    insights.append(f"{top_pollutant} is the most problematic pollutant, exceeding healthy levels {top_rate:.1f}% of the time")
                    
                    # Add secondary pollutant if available
                    if len(sorted_rates) > 1:
                        second_pollutant, second_rate = sorted_rates[1]
                        if second_rate > 5:
                            insights.append(f"{second_pollutant} is also concerning, exceeding healthy levels {second_rate:.1f}% of the time")
            
        # Check for regional variations if location data is available
        if all(col in df.columns for col in ['Latitude', 'Longitude', 'Device_ID']):
            if df['Device_ID'].nunique() > 1:
                insights.append("Regional variations are present in the data, consider using the map view for spatial analysis")
                
    except Exception as e:
        logger.error(f"Error generating general insights: {str(e)}")
        insights.append("Unable to generate complete insights due to an error in the analysis")
        
    return insights

def analyze_pollutant_trends(df):
    """
    Analyze trends in pollutant concentrations over time
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
        
    Returns:
    --------
    dict
        A dictionary containing trend analyses for different pollutants
    """
    trends = {}
    
    try:
        # Check if datetime column exists
        if 'Datetime' not in df.columns:
            return {'error': 'No datetime information available for trend analysis'}
            
        # Get list of pollutant columns
        pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'AQI']]
        
        if not pollutant_cols:
            return {'error': 'No pollutant data found for trend analysis'}
            
        # Resample to daily averages for trend analysis
        df_daily = df.copy()
        df_daily['Date'] = pd.to_datetime(df['Datetime']).dt.date
        df_daily = df_daily.groupby('Date').mean(numeric_only=True).reset_index()
        
        # Calculate trends for each pollutant
        for pollutant in pollutant_cols:
            if pollutant in df_daily.columns:
                # Get recent trend (last 7 days if available)
                days_available = len(df_daily)
                window = min(7, days_available)
                
                if days_available >= 3:  # Need at least 3 days for meaningful trend
                    recent_data = df_daily.tail(window)
                    
                    # Calculate simple linear trend
                    X = np.array(range(len(recent_data))).reshape(-1, 1)
                    y = recent_data[pollutant].values
                    
                    # Fit linear regression
                    model = LinearRegression()
                    model.fit(X, y)
                    slope = model.coef_[0]
                    
                    # Interpret trend
                    if abs(slope) < 0.01 * np.mean(y):  # Threshold for "stable"
                        trend_type = "stable"
                    elif slope > 0:
                        trend_type = "increasing"
                    else:
                        trend_type = "decreasing"
                    
                    # Calculate percent change if applicable
                    if len(recent_data) > 1:
                        first_value = recent_data[pollutant].iloc[0]
                        last_value = recent_data[pollutant].iloc[-1]
                        
                        if first_value > 0:
                            percent_change = ((last_value - first_value) / first_value) * 100
                        else:
                            percent_change = 0
                    else:
                        percent_change = 0
                    
                    # Store trend information
                    trends[pollutant] = {
                        'trend': trend_type,
                        'slope': float(slope),
                        'percent_change': float(percent_change),
                        'recent_average': float(recent_data[pollutant].mean()),
                        'window_days': window
                    }
        
        # Analyze daily and weekly patterns if enough data
        if days_available >= 7 and 'Datetime' in df.columns:
            df['Hour'] = pd.to_datetime(df['Datetime']).dt.hour
            df['Day'] = pd.to_datetime(df['Datetime']).dt.dayofweek  # 0=Monday, 6=Sunday
            
            # Daily patterns (by hour)
            hourly_patterns = {}
            for pollutant in pollutant_cols:
                if pollutant in df.columns:
                    hourly_avg = df.groupby('Hour')[pollutant].mean().to_dict()
                    
                    # Find peak hours (top 3)
                    peak_hours = sorted(hourly_avg.items(), key=lambda x: x[1], reverse=True)[:3]
                    peak_hours = [(int(hour), float(val)) for hour, val in peak_hours]
                    
                    # Find lowest hours (bottom 3)
                    low_hours = sorted(hourly_avg.items(), key=lambda x: x[1])[:3]
                    low_hours = [(int(hour), float(val)) for hour, val in low_hours]
                    
                    hourly_patterns[pollutant] = {
                        'peak_hours': peak_hours,
                        'lowest_hours': low_hours
                    }
            
            trends['daily_patterns'] = hourly_patterns
            
            # Weekly patterns (by day)
            weekly_patterns = {}
            for pollutant in pollutant_cols:
                if pollutant in df.columns:
                    daily_avg = df.groupby('Day')[pollutant].mean().to_dict()
                    
                    # Map numeric days to names for better readability
                    day_names = {
                        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                    }
                    
                    # Convert to day names
                    named_daily_avg = {day_names[day]: float(val) for day, val in daily_avg.items()}
                    
                    # Find peak and lowest days
                    peak_day = max(daily_avg.items(), key=lambda x: x[1])
                    peak_day_name = day_names[peak_day[0]]
                    peak_value = float(peak_day[1])
                    
                    low_day = min(daily_avg.items(), key=lambda x: x[1])
                    low_day_name = day_names[low_day[0]]
                    low_value = float(low_day[1])
                    
                    weekly_patterns[pollutant] = {
                        'daily_averages': named_daily_avg,
                        'peak_day': {'day': peak_day_name, 'value': peak_value},
                        'lowest_day': {'day': low_day_name, 'value': low_value}
                    }
            
            trends['weekly_patterns'] = weekly_patterns

    except Exception as e:
        logger.error(f"Error analyzing pollutant trends: {str(e)}")
        trends['error'] = f"Failed to complete trend analysis: {str(e)}"
        
    return trends

def detect_anomalies(df):
    """
    Detect anomalies in the air quality data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
        
    Returns:
    --------
    list
        A list of detected anomalies with descriptions
    """
    anomalies = []
    
    try:
        # Check if we have enough data
        if df is None or len(df) < 10:
            return [{"message": "Insufficient data for anomaly detection"}]
            
        # Get pollutant columns
        pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'AQI']]
        
        if not pollutant_cols:
            return [{"message": "No pollutant data found for anomaly detection"}]
        
        # Process each pollutant separately
        for pollutant in pollutant_cols:
            if pollutant in df.columns and df[pollutant].notna().sum() > 10:
                # Extract the data
                data = df[pollutant].dropna().values.reshape(-1, 1)
                
                # Standardize the data
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
                
                # Use Isolation Forest for anomaly detection
                model = IsolationForest(contamination=0.05, random_state=42)
                preds = model.fit_predict(data_scaled)
                
                # Extract anomalies
                anomaly_indices = np.where(preds == -1)[0]
                
                if len(anomaly_indices) > 0:
                    # Calculate statistics
                    pollutant_avg = df[pollutant].mean()
                    pollutant_std = df[pollutant].std()
                    
                    # Get anomaly values
                    anomaly_values = df[pollutant].iloc[anomaly_indices].values
                    
                    # Group anomalies by severity
                    extreme_high = [val for val in anomaly_values if val > pollutant_avg + 3*pollutant_std]
                    high = [val for val in anomaly_values if pollutant_avg + 2*pollutant_std < val <= pollutant_avg + 3*pollutant_std]
                    low = [val for val in anomaly_values if pollutant_avg - 3*pollutant_std < val <= pollutant_avg - 2*pollutant_std]
                    extreme_low = [val for val in anomaly_values if val <= pollutant_avg - 3*pollutant_std]
                    
                    # Add insights based on anomalies
                    if extreme_high:
                        anomalies.append({
                            "pollutant": pollutant,
                            "type": "extreme_high",
                            "count": len(extreme_high),
                            "avg_value": float(sum(extreme_high) / len(extreme_high)),
                            "max_value": float(max(extreme_high)),
                            "message": f"Detected {len(extreme_high)} extremely high {pollutant} values"
                        })
                    
                    if high:
                        anomalies.append({
                            "pollutant": pollutant,
                            "type": "high",
                            "count": len(high),
                            "avg_value": float(sum(high) / len(high)),
                            "message": f"Detected {len(high)} abnormally high {pollutant} values"
                        })
                        
                    if low or extreme_low:
                        anomalies.append({
                            "pollutant": pollutant,
                            "type": "low",
                            "count": len(low) + len(extreme_low),
                            "message": f"Detected {len(low) + len(extreme_low)} abnormally low {pollutant} values"
                        })
        
        # Check for temporal anomalies if datetime is available
        if 'Datetime' in df.columns and len(df) > 48:  # Need at least 2 days of data
            # Group by hour and calculate the average
            df['Hour'] = pd.to_datetime(df['Datetime']).dt.hour
            hourly_avg = df.groupby('Hour').mean(numeric_only=True)
            
            for pollutant in pollutant_cols:
                if pollutant in hourly_avg.columns:
                    # Calculate the difference between consecutive hours
                    hourly_diff = hourly_avg[pollutant].diff().abs()
                    
                    # Find hours with unusual jumps (more than 2x the average difference)
                    avg_diff = hourly_diff.mean()
                    unusual_jumps = hourly_diff[hourly_diff > 2 * avg_diff]
                    
                    if not unusual_jumps.empty:
                        for hour, diff in unusual_jumps.items():
                            prev_hour = (hour - 1) % 24
                            anomalies.append({
                                "pollutant": pollutant,
                                "type": "temporal",
                                "hours": f"{prev_hour:02d}:00 to {hour:02d}:00",
                                "jump_value": float(diff),
                                "message": f"Unusual jump in {pollutant} between {prev_hour:02d}:00 and {hour:02d}:00"
                            })
    
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        anomalies.append({
            "type": "error",
            "message": f"Failed to complete anomaly detection: {str(e)}"
        })
        
    return anomalies

def find_correlations(df):
    """
    Find correlations between different parameters in the data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
        
    Returns:
    --------
    dict
        A dictionary containing correlation analyses
    """
    correlations = {}
    
    try:
        # Check if we have enough data
        if df is None or len(df) < 10:
            return {'error': 'Insufficient data for correlation analysis'}
            
        # Get relevant columns for correlation analysis
        relevant_cols = [col for col in df.columns if col in 
                        ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'AQI', 
                         'Temperature', 'Humidity', 'Pressure']]
        
        if len(relevant_cols) < 2:
            return {'error': 'Not enough parameters for correlation analysis'}
            
        # Calculate correlation matrix
        corr_matrix = df[relevant_cols].corr().round(2)
        
        # Convert to dictionary format
        correlations['matrix'] = corr_matrix.to_dict()
        
        # Find strong correlations (positive and negative)
        strong_positive = []
        strong_negative = []
        moderate_positive = []
        moderate_negative = []
        
        for i in range(len(relevant_cols)):
            for j in range(i+1, len(relevant_cols)):
                param1 = relevant_cols[i]
                param2 = relevant_cols[j]
                corr_value = corr_matrix.iloc[i, j]
                
                # Skip if correlation is NaN
                if pd.isna(corr_value):
                    continue
                    
                # Categorize by strength and direction
                if corr_value >= 0.7:
                    strong_positive.append({
                        'param1': param1,
                        'param2': param2,
                        'correlation': float(corr_value)
                    })
                elif corr_value >= 0.5:
                    moderate_positive.append({
                        'param1': param1,
                        'param2': param2,
                        'correlation': float(corr_value)
                    })
                elif corr_value <= -0.7:
                    strong_negative.append({
                        'param1': param1,
                        'param2': param2,
                        'correlation': float(corr_value)
                    })
                elif corr_value <= -0.5:
                    moderate_negative.append({
                        'param1': param1,
                        'param2': param2,
                        'correlation': float(corr_value)
                    })
        
        # Store the categorized correlations
        correlations['strong_positive'] = strong_positive
        correlations['strong_negative'] = strong_negative
        correlations['moderate_positive'] = moderate_positive
        correlations['moderate_negative'] = moderate_negative
        
        # Add explanations for interesting correlations
        correlations['insights'] = []
        
        # Check for temperature and pollutant correlations
        if 'Temperature' in relevant_cols:
            for pollutant in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']:
                if pollutant in relevant_cols:
                    corr_value = corr_matrix.loc['Temperature', pollutant]
                    
                    if not pd.isna(corr_value):
                        if corr_value > 0.5:
                            correlations['insights'].append(
                                f"Higher temperatures are associated with higher {pollutant} levels (correlation: {corr_value:.2f})"
                            )
                        elif corr_value < -0.5:
                            correlations['insights'].append(
                                f"Higher temperatures are associated with lower {pollutant} levels (correlation: {corr_value:.2f})"
                            )
                            
        # Check for humidity and pollutant correlations
        if 'Humidity' in relevant_cols:
            for pollutant in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']:
                if pollutant in relevant_cols:
                    corr_value = corr_matrix.loc['Humidity', pollutant]
                    
                    if not pd.isna(corr_value):
                        if corr_value > 0.5:
                            correlations['insights'].append(
                                f"Higher humidity is associated with higher {pollutant} levels (correlation: {corr_value:.2f})"
                            )
                        elif corr_value < -0.5:
                            correlations['insights'].append(
                                f"Higher humidity is associated with lower {pollutant} levels (correlation: {corr_value:.2f})"
                            )
                            
        # Check correlation between primary pollutants
        primary_pollutants = ['PM2.5', 'PM10', 'NO2', 'CO']
        available_primaries = [p for p in primary_pollutants if p in relevant_cols]
        
        for i in range(len(available_primaries)):
            for j in range(i+1, len(available_primaries)):
                pollutant1 = available_primaries[i]
                pollutant2 = available_primaries[j]
                corr_value = corr_matrix.loc[pollutant1, pollutant2]
                
                if not pd.isna(corr_value) and abs(corr_value) > 0.7:
                    correlations['insights'].append(
                        f"{pollutant1} and {pollutant2} show a strong {'positive' if corr_value > 0 else 'negative'} correlation ({corr_value:.2f}), suggesting a common source"
                    )
                    
    except Exception as e:
        logger.error(f"Error finding correlations: {str(e)}")
        correlations['error'] = f"Failed to complete correlation analysis: {str(e)}"
        
    return correlations

def generate_recommendations(df, insights):
    """
    Generate actionable recommendations based on the data and insights
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
    insights : dict
        Dictionary of insights generated from the data
        
    Returns:
    --------
    list
        A list of recommendations
    """
    recommendations = []
    
    try:
        # Check air quality trends for recommendations
        if 'pollutant_trends' in insights and not insights['pollutant_trends'].get('error'):
            trends = insights['pollutant_trends']
            
            # Check individual pollutant trends
            for pollutant, trend_data in trends.items():
                if pollutant not in ['daily_patterns', 'weekly_patterns', 'error']:
                    # Respond to increasing trends
                    if trend_data.get('trend') == 'increasing' and trend_data.get('percent_change', 0) > 10:
                        recommendations.append({
                            'priority': 'high',
                            'category': 'trend',
                            'pollutant': pollutant,
                            'message': f"{pollutant} levels are increasing significantly. Consider implementing emission reduction strategies."
                        })
                    
                    # Highlight concerning average levels
                    if pollutant == 'PM2.5' and trend_data.get('recent_average', 0) > 35:
                        recommendations.append({
                            'priority': 'high' if trend_data.get('recent_average', 0) > 50 else 'medium',
                            'category': 'level',
                            'pollutant': pollutant,
                            'message': f"Average PM2.5 levels ({trend_data.get('recent_average', 0):.1f} µg/m³) exceed healthy limits. Consider air filtration and reduced outdoor activities."
                        })
                    elif pollutant == 'PM10' and trend_data.get('recent_average', 0) > 50:
                        recommendations.append({
                            'priority': 'high' if trend_data.get('recent_average', 0) > 75 else 'medium',
                            'category': 'level',
                            'pollutant': pollutant,
                            'message': f"Average PM10 levels ({trend_data.get('recent_average', 0):.1f} µg/m³) exceed recommended limits. Consider dust suppression measures."
                        })
                    elif pollutant == 'NO2' and trend_data.get('recent_average', 0) > 100:
                        recommendations.append({
                            'priority': 'medium',
                            'category': 'level',
                            'pollutant': pollutant,
                            'message': f"Elevated NO2 levels detected. Consider traffic management strategies in affected areas."
                        })
                    elif pollutant == 'O3' and trend_data.get('recent_average', 0) > 70:
                        recommendations.append({
                            'priority': 'medium',
                            'category': 'level',
                            'pollutant': pollutant,
                            'message': f"Ozone levels are elevated. Limit outdoor activities during peak hours."
                        })
            
            # Check daily patterns
            if 'daily_patterns' in trends:
                for pollutant, pattern in trends['daily_patterns'].items():
                    if 'peak_hours' in pattern and pattern['peak_hours']:
                        peak_hour, peak_value = pattern['peak_hours'][0]
                        
                        # Format hour in 12-hour format for recommendation
                        peak_hour_12h = f"{peak_hour if peak_hour < 12 else peak_hour-12}{' AM' if peak_hour < 12 else ' PM'}"
                        
                        recommendations.append({
                            'priority': 'medium',
                            'category': 'pattern',
                            'pollutant': pollutant,
                            'message': f"{pollutant} levels peak around {peak_hour_12h}. Consider adjusting outdoor activities and ventilation schedules accordingly."
                        })
        
        # Check for anomalies that require attention
        if 'anomalies' in insights:
            anomalies = insights['anomalies']
            extreme_anomalies = [a for a in anomalies if a.get('type') == 'extreme_high']
            
            if extreme_anomalies:
                for anomaly in extreme_anomalies:
                    pollutant = anomaly.get('pollutant', 'Unknown')
                    recommendations.append({
                        'priority': 'high',
                        'category': 'anomaly',
                        'pollutant': pollutant,
                        'message': f"Detected extreme {pollutant} spikes. Investigate potential emission sources or sensor malfunctions."
                    })
        
        # Check for correlations that might lead to recommendations
        if 'correlations' in insights and 'insights' in insights['correlations']:
            corr_insights = insights['correlations']['insights']
            
            # Generate recommendations from correlations with weather parameters
            for insight in corr_insights:
                if 'humidity' in insight.lower() and 'higher' in insight.lower():
                    pollutant = next((p for p in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2'] if p in insight), None)
                    if pollutant:
                        recommendations.append({
                            'priority': 'medium',
                            'category': 'correlation',
                            'pollutant': pollutant,
                            'message': f"Consider humidity control measures to help manage {pollutant} levels, as they show strong correlation."
                        })
                        
                if 'temperature' in insight.lower() and 'higher' in insight.lower():
                    pollutant = next((p for p in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2'] if p in insight), None)
                    if pollutant:
                        recommendations.append({
                            'priority': 'medium',
                            'category': 'correlation',
                            'pollutant': pollutant,
                            'message': f"Be especially vigilant about {pollutant} levels during warmer periods, as they show significant temperature correlation."
                        })
        
        # Check for device/location specific recommendations
        if 'Device_ID' in df.columns and df['Device_ID'].nunique() > 1:
            # Calculate average AQI by device if AQI column exists
            if 'AQI' in df.columns:
                device_aqi = df.groupby('Device_ID')['AQI'].mean().sort_values(ascending=False)
                
                # Recommend focusing on highest pollution areas
                if not device_aqi.empty:
                    worst_device = device_aqi.index[0]
                    worst_aqi = device_aqi.iloc[0]
                    
                    if worst_aqi > 100:  # Only recommend if AQI is at least moderate
                        recommendations.append({
                            'priority': 'high' if worst_aqi > 150 else 'medium',
                            'category': 'location',
                            'device': worst_device,
                            'message': f"Location monitored by device {worst_device} shows the highest pollution levels (AQI: {worst_aqi:.1f}). Prioritize interventions in this area."
                        })
        
        # Add general recommendations if we don't have many specific ones
        if len(recommendations) < 3:
            recommendations.append({
                'priority': 'medium',
                'category': 'general',
                'message': "Expand monitoring network to capture more spatial variations in air quality."
            })
            
            recommendations.append({
                'priority': 'medium',
                'category': 'general',
                'message': "Implement an early warning system for pollution episodes based on weather forecasts and historical patterns."
            })
            
        # Ensure we return at least one recommendation
        if not recommendations:
            recommendations.append({
                'priority': 'medium',
                'category': 'general',
                'message': "Continue regular monitoring to establish baseline conditions and detect emerging trends."
            })
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        recommendations.append({
            'priority': 'medium',
            'category': 'error',
            'message': "Unable to generate comprehensive recommendations due to analysis errors."
        })
        
    return recommendations

def generate_forecasts(df):
    """
    Generate simple forecasts for pollutant levels
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
        
    Returns:
    --------
    dict
        Forecast data for various pollutants
    """
    forecasts = {}
    
    try:
        # Check if we have enough data and datetime column
        if df is None or 'Datetime' not in df.columns or len(df) < 24:
            return {'error': 'Insufficient data for forecasting'}
            
        # Get pollutant columns
        pollutant_cols = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'AQI']]
        
        if not pollutant_cols:
            return {'error': 'No pollutant data found for forecasting'}
            
        # Resample to hourly data for forecasting
        df_hourly = df.copy()
        df_hourly['Datetime'] = pd.to_datetime(df['Datetime'])
        df_hourly = df_hourly.set_index('Datetime').sort_index()
        
        # For each pollutant, fit a simple model and predict next 24 hours
        for pollutant in pollutant_cols:
            if pollutant in df_hourly.columns and df_hourly[pollutant].notna().sum() > 24:
                # Create a series of hourly values
                hourly_data = df_hourly[pollutant].resample('H').mean()
                
                # Fill any missing values
                hourly_data = hourly_data.fillna(method='ffill').fillna(method='bfill')
                
                # Get last 7 days of data for training (if available)
                train_data = hourly_data.tail(min(168, len(hourly_data)))
                
                if len(train_data) >= 24:  # Need at least 24 hours for reasonable forecast
                    # Create features: hour of day and day of week
                    X = np.column_stack([
                        train_data.index.hour.values,  # Hour of day
                        train_data.index.dayofweek.values  # Day of week
                    ])
                    y = train_data.values
                    
                    # Fit linear regression model
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Create features for the next 24 hours
                    last_timestamp = train_data.index[-1]
                    forecast_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(24)]
                    forecast_X = np.column_stack([
                        [ts.hour for ts in forecast_timestamps],
                        [ts.dayofweek for ts in forecast_timestamps]
                    ])
                    
                    # Make predictions
                    forecast_values = model.predict(forecast_X)
                    
                    # Ensure no negative values
                    forecast_values = np.maximum(forecast_values, 0)
                    
                    # Format forecast data
                    forecast_data = {
                        'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in forecast_timestamps],
                        'values': [float(val) for val in forecast_values],
                        'model_type': 'linear_regression',
                        'confidence': 'medium'  # Simple model, medium confidence
                    }
                    
                    forecasts[pollutant] = forecast_data
                    
        # Add metadata
        forecasts['metadata'] = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'forecast_hours': 24,
            'model_description': 'Simple linear regression based on hour of day and day of week patterns'
        }
            
    except Exception as e:
        logger.error(f"Error generating forecasts: {str(e)}")
        forecasts['error'] = f"Failed to complete forecasting: {str(e)}"
        
    return forecasts

def assess_health_impact(df):
    """
    Assess the potential health impacts of the air quality data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
        
    Returns:
    --------
    dict
        Health impact assessment information
    """
    health_impact = {}
    
    try:
        # Check if we have necessary data
        required_cols = ['PM2.5', 'PM10', 'NO2', 'O3', 'AQI']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if not available_cols:
            return {'error': 'No air quality parameters available for health impact assessment'}
            
        # Overall health risk based on AQI if available
        if 'AQI' in df.columns:
            avg_aqi = df['AQI'].mean()
            max_aqi = df['AQI'].max()
            
            # Define health risk levels
            if avg_aqi <= 50:
                risk_level = "Low"
                description = "Air quality is satisfactory, and air pollution poses little or no risk."
            elif avg_aqi <= 100:
                risk_level = "Moderate"
                description = "Air quality is acceptable; however, there may be some health concerns for a small number of sensitive individuals."
            elif avg_aqi <= 150:
                risk_level = "Elevated"
                description = "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
            elif avg_aqi <= 200:
                risk_level = "High"
                description = "Some members of the general public may experience health effects; members of sensitive groups may experience more serious effects."
            elif avg_aqi <= 300:
                risk_level = "Very High"
                description = "Health alert: The risk of health effects is increased for everyone."
            else:
                risk_level = "Hazardous"
                description = "Health warning of emergency conditions. The entire population is likely to be affected."
                
            health_impact['overall_risk'] = {
                'level': risk_level,
                'description': description,
                'avg_aqi': float(avg_aqi),
                'max_aqi': float(max_aqi)
            }
            
            # Use Gemini API to generate more detailed health insights
            try:
                # Initialize the Gemini API with the provided key
                import os
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    
                    # Prepare data summary for Gemini
                    pollutant_data = {}
                    for col in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']:
                        if col in df.columns:
                            pollutant_data[col] = {
                                'avg': float(df[col].mean()),
                                'max': float(df[col].max())
                            }
                    
                    # Create the prompt for Gemini
                    prompt = f"""
                    Analyze the following air quality data and provide specific health impact insights:
                    
                    Average AQI: {avg_aqi:.1f} (Risk Level: {risk_level})
                    Maximum AQI: {max_aqi:.1f}
                    
                    Pollutant Averages:
                    {json.dumps(pollutant_data, indent=2)}
                    
                    Please provide:
                    1. Potential health impacts for general population (2-3 sentences)
                    2. Specific impacts for sensitive groups (elderly, children, respiratory conditions) (2-3 sentences)
                    3. Recommended precautions based on these levels (3-4 specific bullet points)
                    4. Long-term health concerns if these levels persist (1-2 sentences)
                    
                    Keep the response concise and focused on actionable information.
                    Format the response with clear section headings.
                    """
                    
                    # Generate content with Gemini
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(prompt)
                    
                    # Process and structure the response
                    if response and hasattr(response, 'text'):
                        # Get the response text
                        gemini_insights = response.text.strip()
                        
                        # Add Gemini-generated insights
                        health_impact['gemini_insights'] = gemini_insights
                        logger.info("Successfully generated AI health insights with Gemini")
                    else:
                        logger.warning("Gemini API returned an empty or invalid response")
                else:
                    logger.warning("No Google API key available for Gemini integration")
            except Exception as e:
                logger.error(f"Error generating Gemini health insights: {str(e)}")
            
        # Specific pollutant health impacts
        pollutant_impacts = {}
        
        # PM2.5 impact
        if 'PM2.5' in df.columns:
            pm25_avg = df['PM2.5'].mean()
            pm25_exceedance = (df['PM2.5'] > 35).mean() * 100  # % of time exceeding WHO guideline
            
            pm25_impact = {
                'average': float(pm25_avg),
                'exceedance_rate': float(pm25_exceedance),
                'health_effects': []
            }
            
            # Add relevant health effects based on levels
            if pm25_avg > 50:
                pm25_impact['health_effects'].append("Increased respiratory symptoms, aggravation of heart and lung diseases, and premature mortality")
            elif pm25_avg > 35:
                pm25_impact['health_effects'].append("Increased aggravation of heart or lung disease and premature mortality in sensitive groups")
            elif pm25_avg > 12:
                pm25_impact['health_effects'].append("Possible respiratory symptoms in sensitive individuals")
                
            pollutant_impacts['PM2.5'] = pm25_impact
            
        # PM10 impact
        if 'PM10' in df.columns:
            pm10_avg = df['PM10'].mean()
            pm10_exceedance = (df['PM10'] > 50).mean() * 100  # % of time exceeding WHO guideline
            
            pm10_impact = {
                'average': float(pm10_avg),
                'exceedance_rate': float(pm10_exceedance),
                'health_effects': []
            }
            
            # Add relevant health effects based on levels
            if pm10_avg > 150:
                pm10_impact['health_effects'].append("Significant aggravation of respiratory and cardiovascular symptoms")
            elif pm10_avg > 50:
                pm10_impact['health_effects'].append("Increased respiratory symptoms and aggravation of lung diseases")
                
            pollutant_impacts['PM10'] = pm10_impact
            
        # O3 impact
        if 'O3' in df.columns:
            o3_avg = df['O3'].mean()
            o3_exceedance = (df['O3'] > 70).mean() * 100  # % of time exceeding EPA standard
            
            o3_impact = {
                'average': float(o3_avg),
                'exceedance_rate': float(o3_exceedance),
                'health_effects': []
            }
            
            # Add relevant health effects based on levels
            if o3_avg > 100:
                o3_impact['health_effects'].append("Significant respiratory effects, especially during outdoor activities")
            elif o3_avg > 70:
                o3_impact['health_effects'].append("Possible respiratory symptoms, especially during extended outdoor activities")
                
            pollutant_impacts['O3'] = o3_impact
            
        # NO2 impact
        if 'NO2' in df.columns:
            no2_avg = df['NO2'].mean()
            no2_exceedance = (df['NO2'] > 100).mean() * 100  # % of time exceeding standard
            
            no2_impact = {
                'average': float(no2_avg),
                'exceedance_rate': float(no2_exceedance),
                'health_effects': []
            }
            
            # Add relevant health effects based on levels
            if no2_avg > 200:
                no2_impact['health_effects'].append("Significant respiratory issues, inflammation of airways")
            elif no2_avg > 100:
                no2_impact['health_effects'].append("Increased respiratory symptoms, especially in sensitive groups")
                
            pollutant_impacts['NO2'] = no2_impact
            
        health_impact['pollutant_impacts'] = pollutant_impacts
        
        # Vulnerable populations impact assessment
        vulnerable_groups = []
        
        # Add relevant vulnerable groups based on pollutant levels
        if ('PM2.5' in df.columns and df['PM2.5'].mean() > 12) or \
           ('PM10' in df.columns and df['PM10'].mean() > 35) or \
           ('O3' in df.columns and df['O3'].mean() > 60) or \
           ('NO2' in df.columns and df['NO2'].mean() > 80):
            vulnerable_groups.extend([
                {
                    'group': 'Children',
                    'concern': 'Developing respiratory systems are more vulnerable to air pollution'
                },
                {
                    'group': 'Elderly',
                    'concern': 'Decreased lung function and potential cardiovascular complications'
                },
                {
                    'group': 'Asthmatics',
                    'concern': 'Increased risk of asthma attacks and respiratory symptoms'
                },
                {
                    'group': 'Individuals with heart/lung disease',
                    'concern': 'Exacerbation of existing conditions'
                }
            ])
            
        health_impact['vulnerable_groups'] = vulnerable_groups
            
    except Exception as e:
        logger.error(f"Error assessing health impact: {str(e)}")
        health_impact['error'] = f"Failed to complete health impact assessment: {str(e)}"
        
    return health_impact

def assess_device_performance(df):
    """
    Assess the performance of monitoring devices in the data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed air quality data
        
    Returns:
    --------
    dict
        Device performance assessment information
    """
    performance = {}
    
    try:
        # Check if we have device information
        if 'Device_ID' not in df.columns:
            return {'error': 'No device information available for performance assessment'}
            
        # Get unique devices
        devices = df['Device_ID'].unique()
        
        # Device-specific metrics
        device_metrics = {}
        
        for device in devices:
            device_data = df[df['Device_ID'] == device]
            
            # Data completeness
            if 'Datetime' in device_data.columns:
                # Calculate expected number of records based on the time range
                start_time = pd.to_datetime(device_data['Datetime']).min()
                end_time = pd.to_datetime(device_data['Datetime']).max()
                
                time_range_hours = (end_time - start_time).total_seconds() / 3600
                
                # Assume readings every hour (adjust as needed)
                expected_records = max(1, int(time_range_hours))
                actual_records = len(device_data)
                
                completeness = min(100, (actual_records / expected_records) * 100)
            else:
                completeness = None
                
            # Data quality
            pollutant_cols = [col for col in device_data.columns if col in 
                             ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']]
            
            # Calculate percentage of non-null values
            if pollutant_cols:
                quality_scores = []
                
                for col in pollutant_cols:
                    if col in device_data.columns:
                        non_null_pct = device_data[col].notna().mean() * 100
                        quality_scores.append(non_null_pct)
                
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None
            else:
                avg_quality = None
                
            # Overall device status
            if completeness is not None and avg_quality is not None:
                if completeness >= 90 and avg_quality >= 95:
                    status = "Excellent"
                elif completeness >= 75 and avg_quality >= 85:
                    status = "Good"
                elif completeness >= 60 and avg_quality >= 70:
                    status = "Fair"
                else:
                    status = "Poor"
            else:
                status = "Unknown"
                
            # Store metrics
            device_metrics[str(device)] = {
                'data_completeness': float(completeness) if completeness is not None else None,
                'data_quality': float(avg_quality) if avg_quality is not None else None,
                'record_count': int(actual_records) if 'actual_records' in locals() else len(device_data),
                'status': status
            }
            
        performance['device_metrics'] = device_metrics
        
        # Overall network assessment
        if device_metrics:
            statuses = [metrics['status'] for metrics in device_metrics.values()]
            
            # Count status frequencies
            status_counts = {
                'Excellent': statuses.count('Excellent'),
                'Good': statuses.count('Good'),
                'Fair': statuses.count('Fair'),
                'Poor': statuses.count('Poor'),
                'Unknown': statuses.count('Unknown')
            }
            
            # Overall network status
            total_devices = len(devices)
            good_or_better = status_counts['Excellent'] + status_counts['Good']
            
            if good_or_better >= total_devices * 0.8:
                network_status = "Excellent"
            elif good_or_better >= total_devices * 0.6:
                network_status = "Good"
            elif good_or_better >= total_devices * 0.4:
                network_status = "Fair"
            else:
                network_status = "Needs Attention"
                
            performance['network_status'] = {
                'status': network_status,
                'device_count': total_devices,
                'status_breakdown': status_counts
            }
            
    except Exception as e:
        logger.error(f"Error assessing device performance: {str(e)}")
        performance['error'] = f"Failed to complete device performance assessment: {str(e)}"
        
    return performance
