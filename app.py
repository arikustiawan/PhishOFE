import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Phishing URL Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        background-color: #0056A3;
        padding: 20px;
        text-align: center;
        color: white;
        font-size: 22px;
        font-weight: bold;
    }
    .sub-header {
        color: #0056A3;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .custom-button button {
        background-color: #0056A3 !important;
        color: white !important;
        font-size: 16px !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #0056A3;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
    }
    .result-box {
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        margin-top: 20px;
        padding: 10px;
        border-radius: 5px;
    }
    .secure {
        color: green;
        border: 2px solid green;
        background-color: #eaffea;
    }
    .not-secure {
        color: red;
        border: 2px solid red;
        background-color: #ffeaea;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header section
st.markdown('<div class="main-header">PHISHING URL DETECTION USING MACHINE LEARNING</div>', unsafe_allow_html=True)

# Navigation bar
st.markdown(
    """
    <div style="display: flex; justify-content: center; gap: 20px; margin: 10px 0; font-size: 16px; font-weight: bold;">
        <a href="#" style="color: #0056A3; text-decoration: none;">Upload Dataset</a>
        <a href="#" style="color: #0056A3; text-decoration: none;">Predict URL</a>
        <a href="#" style="color: #0056A3; text-decoration: none;">Performance Analysis</a>
    </div>
    """,
    unsafe_allow_html=True
)

# Input section
st.markdown('<div class="sub-header">ENTER URL:</div>', unsafe_allow_html=True)
url_input = st.text_input("", placeholder="Enter URL here", label_visibility="collapsed")

# Button section
st.markdown('<div class="button-container custom-button">', unsafe_allow_html=True)
if st.button("CHECK"):
    # Example logic for URL detection
    if "facebook.com" in url_input:
        st.markdown('<div class="result-box secure">URL looks secure!<br>and safe to visit.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box not-secure">URL does not look secure!<br>It might be harmful and unsafe to visit.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer section
st.markdown('<div class="footer">Developed by Ari Kustiawan</div>', unsafe_allow_html=True)
