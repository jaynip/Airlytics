import streamlit as st

# Define SVG icons as constants for use throughout the app

# Leaf icon for environmental indicators
LEAF_ICON = '''
<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M6.5 17.5L4 15C4 10.7574 7.75736 7 12 7C16.2426 7 20 10.7574 20 15C20 15.5523 19.5523 16 19 16C18.4477 16 18 15.5523 18 15C18 11.8624 15.1376 9 12 9C8.8624 9 6 11.8624 6 15L3.5 12.5" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M12 13C13.6569 13 15 14.3431 15 16C15 17.6569 13.6569 19 12 19C10.3431 19 9 17.6569 9 16" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
'''

# Cloud icon for air quality indicators
CLOUD_ICON = '''
<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M6.5 19C4.01472 19 2 16.9853 2 14.5C2 12.1564 3.79151 10.2313 6.07974 10.0194C6.54781 7.17213 9.02024 5 12 5C14.9798 5 17.4522 7.17213 17.9203 10.0194C20.2085 10.2313 22 12.1564 22 14.5C22 16.9853 19.9853 19 17.5 19C13.5 19 10.5 19 6.5 19Z" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
'''

# Sun icon for outdoor conditions
SUN_ICON = '''
<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="12" cy="12" r="4" stroke="#4bb051" stroke-width="1.5"/>
    <path d="M12 5V3" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M12 21V19" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M16.9498 7.04996L18.364 5.63574" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M5.63608 18.3644L7.05029 16.9502" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M19 12L21 12" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M3 12L5 12" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M16.9498 16.95L18.364 18.3643" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M5.63608 5.63559L7.05029 7.0498" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round"/>
</svg>
'''

# Thermometer icon for temperature indicators
THERMOMETER_ICON = '''
<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M14.5 4.5C14.5 3.11929 13.3807 2 12 2C10.6193 2 9.5 3.11929 9.5 4.5V13.7578C8.29401 14.565 7.5 15.9398 7.5 17.5C7.5 19.9853 9.51472 22 12 22C14.4853 22 16.5 19.9853 16.5 17.5C16.5 15.9398 15.706 14.565 14.5 13.7578V4.5Z" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M12 18.5C12.5523 18.5 13 18.0523 13 17.5C13 16.9477 12.5523 16.5 12 16.5C11.4477 16.5 11 16.9477 11 17.5C11 18.0523 11.4477 18.5 12 18.5Z" fill="#4bb051" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
'''

# Water drop icon for humidity indicators
WATER_DROP_ICON = '''
<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M20 14C20 18.4183 16.4183 22 12 22C7.58172 22 4 18.4183 4 14C4 12.9391 4.20651 11.9074 4.58152 10.9558C5.76829 8.16777 12 2 12 2C12 2 18.2317 8.16777 19.4185 10.9558C19.7935 11.9074 20 12.9391 20 14Z" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M12.5 13C13.3284 13 14 12.3284 14 11.5C14 10.6716 13 10 12 9C11 10 10 10.6716 10 11.5C10 12.3284 10.6716 13 11.5 13" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
'''

# Wind icon for wind speed indicators
WIND_ICON = '''
<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M9.50929 4.99999C9.8755 3.27361 11.41 2 13.25 2C15.3211 2 17 3.67893 17 5.75C17 7.82107 15.3211 9.5 13.25 9.5H2" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M12.5093 19C12.8755 20.7264 14.41 22 16.25 22C18.3211 22 20 20.3211 20 18.25C20 16.1789 18.3211 14.5 16.25 14.5H2" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M6.50048 12C6.50048 13.1046 5.60505 14 4.50048 14C3.3959 14 2.50048 13.1046 2.50048 12C2.50048 10.8954 3.3959 10 4.50048 10C5.60505 10 6.50048 10.8954 6.50048 12Z" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M14.5 12H22" stroke="#4bb051" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
'''

# Pulse animation CSS
PULSE_ANIMATION = '''
@keyframes pulse {
    0% {
        transform: scale(1);
        opacity: 1;
    }
    50% {
        transform: scale(1.1);
        opacity: 0.7;
    }
    100% {
        transform: scale(1);
        opacity: 1;
    }
}
'''

# Floating animation CSS
FLOATING_ANIMATION = '''
@keyframes float {
    0% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-10px);
    }
    100% {
        transform: translateY(0px);
    }
}
'''

def get_banner_html():
    """
    Get HTML for an animated banner with icons.
    
    Returns:
    --------
    str
        HTML string for the banner
    """
    html = f"""
    <style>
        {FLOATING_ANIMATION}
        .banner-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            background: linear-gradient(120deg, rgba(75, 176, 81, 0.1) 0%, rgba(255, 255, 255, 0.8) 100%);
            border-radius: 10px;
            margin: 20px 0;
        }}
        .banner-icon {{
            margin: 0 15px;
            animation: float 3s ease-in-out infinite;
        }}
        .banner-icon:nth-child(2) {{
            animation-delay: 0.5s;
        }}
        .banner-icon:nth-child(3) {{
            animation-delay: 1s;
        }}
        .banner-icon:nth-child(4) {{
            animation-delay: 1.5s;
        }}
    </style>
    <div class="banner-container">
        <div class="banner-icon">{LEAF_ICON}</div>
        <div class="banner-icon">{CLOUD_ICON}</div>
        <div class="banner-icon">{SUN_ICON}</div>
        <div class="banner-icon">{THERMOMETER_ICON}</div>
    </div>
    """
    return html

def get_floating_icons_html():
    """
    Get HTML for floating icons that can be used as decorative elements.
    
    Returns:
    --------
    str
        HTML string for floating icons
    """
    html = f"""
    <style>
        {FLOATING_ANIMATION}
        .floating-icons-container {{
            position: absolute;
            top: 0;
            right: 0;
            width: 100px;
            height: 300px;
            overflow: hidden;
            pointer-events: none;
            z-index: 1;
        }}
        .floating-icon {{
            position: absolute;
            opacity: 0.5;
        }}
        .floating-icon:nth-child(1) {{
            top: 10%;
            right: 20px;
            animation: float 4s ease-in-out infinite;
        }}
        .floating-icon:nth-child(2) {{
            top: 30%;
            right: 5px;
            animation: float 6s ease-in-out infinite;
            animation-delay: 1s;
        }}
        .floating-icon:nth-child(3) {{
            top: 60%;
            right: 25px;
            animation: float 5s ease-in-out infinite;
            animation-delay: 2s;
        }}
    </style>
    <div class="floating-icons-container">
        <div class="floating-icon">{LEAF_ICON}</div>
        <div class="floating-icon">{CLOUD_ICON}</div>
        <div class="floating-icon">{WATER_DROP_ICON}</div>
    </div>
    """
    return html
