import streamlit as st

# Page configuration
st.set_page_config(page_title="Phishing URL Detection", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Header styling */
        .header {
            background-color: #003399;
            padding: 20px;
            color: white;
            text-align: left;
            font-size: 22px;
            font-weight: bold;
        }
        /* MMU logo */
        .logo {
            display: inline-block;
            vertical-align: middle;
            margin-right: 10px;
        }
        /* Navigation bar styling */
        .nav-bar {
            background-color: #003399;
            text-align: right;
            padding: 10px;
            margin-top: -10px;
        }
        .nav-bar a {
            color: white;
            text-decoration: none;
            font-weight: bold;
            margin: 0 15px;
        }
        .nav-bar a:hover {
            text-decoration: underline;
        }
        /* Main content area styling */
        .main-content {
            text-align: center;
            margin-top: 50px;
        }
        .main-content h3 {
            font-size: 20px;
        }
        .main-content input {
            width: 50%;
            padding: 10px;
            margin-top: 20px;
        }
        .main-content button {
            background-color: #003399;
            color: white;
            padding: 10px 20px;
            border: none;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        .main-content button:hover {
            background-color: #0056b3;
        }
        /* Footer styling */
        .footer {
            background-color: #003399;
            color: white;
            text-align: center;
            padding: 10px;
            position: fixed;
            bottom: 0;
            width: 100%;
            font-size: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with logo and title
st.markdown(
    """
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/en/4/42/Multimedia_University_logo.png" alt="MMU Logo" class="logo" width="50" />
        PHISHING URL DETECTION USING MACHINE LEARNING
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation Bar
st.markdown(
    """
    <div class="nav-bar">
        <a href="#">Upload Dataset</a>
        <a href="#">Predict URL</a>
        <a href="#">Performance Analysis</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Content Area
st.markdown(
    """
    <div class="main-content">
        <h3>ENTER URL:</h3>
        <form>
            <input type="text" placeholder="Type your URL here..." id="url_input" />
            <br><br>
            <button type="button" onclick="checkURL()">CHECK</button>
        </form>
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

# Add interactivity with Streamlit
url_input = st.text_input("Enter URL", key="input_key", label_visibility="hidden")

if st.button("CHECK", key="check_button"):
    if url_input:
        st.success(f"Checking URL: {url_input}")
    else:
        st.error("Please enter a URL!")
