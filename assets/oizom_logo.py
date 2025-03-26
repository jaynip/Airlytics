import streamlit as st

def get_oizom_logo_html(width=100):
    """
    Get HTML for the Oizom logo in SVG format.
    
    Parameters:
    -----------
    width : int
        Width of the logo in pixels
        
    Returns:
    --------
    str
        HTML string containing the SVG logo
    """
    # SVG for Oizom logo - using a simple leaf logo design in the company's green color
    svg_logo = f"""
    <svg width="{width}" height="{int(width * 0.6)}" viewBox="0 0 300 180" xmlns="http://www.w3.org/2000/svg">
        <!-- Leaf shape -->
        <path d="M150 20 
                 C 120 40, 80 80, 80 130 
                 C 80 155, 100 175, 125 175 
                 C 150 175, 170 155, 170 130 
                 C 170 80, 130 40, 100 20 
                 Z" 
              fill="#4bb051" />
        
        <!-- Stem -->
        <path d="M125 130 
                 C 135 120, 145 120, 155 130" 
              stroke="#ffffff" stroke-width="5" stroke-linecap="round" fill="none" />
        
        <!-- Leaf veins -->
        <path d="M125 40 
                 C 125 70, 125 100, 125 130" 
              stroke="#ffffff" stroke-width="5" stroke-linecap="round" fill="none" />
        
        <path d="M110 60 
                 C 118 70, 122 80, 125 90" 
              stroke="#ffffff" stroke-width="3" stroke-linecap="round" fill="none" />
        
        <path d="M140 60 
                 C 132 70, 128 80, 125 90" 
              stroke="#ffffff" stroke-width="3" stroke-linecap="round" fill="none" />
        
        <!-- Text "OIZOM" -->
        <text x="185" y="85" font-family="Arial, sans-serif" font-size="36" font-weight="bold" fill="#4bb051">OIZOM</text>
        
        <!-- Tagline -->
        <text x="185" y="110" font-family="Arial, sans-serif" font-size="14" fill="#666666">Smart Monitoring</text>
    </svg>
    """
    
    return svg_logo
