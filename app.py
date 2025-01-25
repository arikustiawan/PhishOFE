import streamlit as st

# Full-width layout configuration
st.set_page_config(page_title="Phishing URL Detection", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f2f2f2;
        }
        .header {
            background-color: #003399;
            padding: 20px;
            color: white;
            text-align: left;
            font-size: 22px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header img {
            height: 50px;
        }
        .nav-links {
            display: flex;
            gap: 20px;
        }
        .nav-links a {
            color: white;
            text-decoration: none;
            font-size: 16px;
        }
        .main-content {
            text-align: center;
            margin-top: 100px;
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
        .input-box {
            width: 30%;
            margin: 20px auto;
            text-align: center;
        }
        .input-box input {
            width: 100%;
            padding: 10px;
            font-size: 16px;
        }
        .button-container {
            margin-top: 20px;
        }
        .button-container button {
            background-color: #003399;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }
        .button-container button:hover {
            background-color: #002266;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="header">
        <div>
            <img src="https://upload.wikimedia.org/wikipedia/en/1/14/MMU_Logo.png" alt="MMU Logo">
        </div>
        <div style="flex-grow: 1; text-align: center; font-weight: bold;">
            PHISHING URL DETECTION USING MACHINE LEARNING
        </div>
        <div class="nav-links">
            <a href="#">Upload Dataset</a>
            <a href="#">Predict URL</a>
            <a href="#">Performance Analysis</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main content
st.markdown(
    """
    <div class="main-content">
        <h3>ENTER URL:</h3>
        <div class="input-box">
            <input type="text" placeholder="Type your URL here...">
        </div>
        <div class="button-container">
            <button>CHECK</button>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    """
    <div class="footer">
        Developed by Ari Kustiawan
    </div>
    """,
    unsafe_allow_html=True,
)
