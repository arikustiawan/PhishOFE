import streamlit as st

# Full-width layout configuration
st.set_page_config(page_title="Phishing URL Detection", layout="wide")

# Header
st.markdown(
    """
    <style>
        .header {
            background-color: #003399;
            padding: 20px;
            color: white;
            text-align: center;
            font-size: 25px;
            margin-bottom: 20px;
        }
        .footer {
            background-color: #003399;
            padding: 10px;
            color: white;
            text-align: center;
            font-size: 12px;
            position: fixed;
            bottom: 0;
            width: 100%;
        }
        .button-container {
            margin: 20px 0;
        }
        .main-content {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .input-box {
            width: 50%;
            margin-bottom: 10px;
        }
        .check-button {
            margin-top: 10px;
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

# Main content container
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

# Center form for URL input
st.markdown("<h3 style='text-align: center;'>ENTER URL:</h3>", unsafe_allow_html=True)
input_url = st.text_input("", placeholder="Type your URL here...", key="input-url")

# Button container
if st.button("CHECK"):
    if input_url:
        st.success(f"Checking URL: {input_url}")
    else:
        st.error("Please enter a URL!")

# Close main-content container
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div class="footer">
        <b>Developed by Ari Kustiawan</b>
    </div>
    """,
    unsafe_allow_html=True,
)
