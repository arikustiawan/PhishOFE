import streamlit as st

# Page configuration
st.set_page_config(page_title="Phishing URL Detection", layout="centered")

# Header
st.markdown(
    """
    <style>
        .header {
            background-color: #003399;
            padding: 15px;
            color: white;
            text-align: center;
            font-size: 25px;
        }
        .footer {
            background-color: #003399;
            padding: 10px;
            color: white;
            text-align: center;
            font-size: 12px;
        }
        .button-container {
            margin: 20px 0;
        }
    </style>
    <div class="header">
        <b>PHISHING URL DETECTION USING MACHINE LEARNING</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation Bar
menu_items = ["Upload Dataset", "Predict URL", "Performance Analysis"]
selected_item = st.sidebar.radio("Menu", menu_items)

# Center form for URL input
st.markdown("<h3 style='text-align: center;'>ENTER URL:</h3>", unsafe_allow_html=True)
input_url = st.text_input("", placeholder="Type your URL here...")

# Button container
if st.button("CHECK"):
    if input_url:
        st.success(f"Checking URL: {input_url}")
    else:
        st.error("Please enter a URL!")

# Footer
st.markdown(
    """
    <div class="footer">
        <b>Developed by Ari Kustiawan</b>
    </div>
    """,
    unsafe_allow_html=True,
)
