import streamlit as st

def get_oizom_logo_html(width=100):
    """
    Get HTML for the Oizom logo.
    
    Parameters:
    -----------
    width : int
        Width of the logo in pixels
        
    Returns:
    --------
    str
        HTML string containing the logo
    """
    # Use a simple image tag with inline styling instead of SVG
    logo_html = f"""
    <div style="display: flex; align-items: center;">
        <div style="color: #4bb051; font-weight: bold; font-size: 24px; margin-right: 5px;">
            <span style="color: #4bb051; font-size: 28px;">O</span>IZOM
        </div>
        <div style="width: 30px; height: 30px; background-color: #4bb051; border-radius: 50%; margin-left: 5px;"></div>
    </div>
    """
    
    return logo_html
