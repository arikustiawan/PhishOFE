import streamlit as st
import joblib
import pandas as pd
import numpy as np
from feature import FeatureExtraction
from sklearn.preprocessing import LabelEncoder

# Load the trained model
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.pkl' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Set up the Streamlit app
st.set_page_config(page_title="Phishing URL Detection", layout="wide", page_icon="🌐")

# Custom CSS for full-page design
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
        }
        .reportview-container {
            padding: 0;
        }
        header {
            visibility: hidden;
        }
        .main {
            padding: 0 !important;
        }
        .nav-bar {
            background-color: #003366;
            padding: 15px 0;
            text-align: center;
        }
        .nav-bar img {
            height: 60px;
        }
        .nav-bar h1 {
            display: inline-block;
            color: white;
            font-size: 24px;
            vertical-align: middle;
            margin-left: 10px;
        }
        .nav-links {
            margin-top: 10px;
        }
        .nav-links a {
            color: white;
            text-decoration: none;
            margin: 0 15px;
            font-size: 18px;
            font-weight: bold;
        }
        .nav-links a:hover {
            text-decoration: underline;
        }
        .input-section {
            text-align: center;
            margin-top: 50px;
        }
        .input-section input {
            width: 40%;
            height: 40px;
            font-size: 18px;
            padding: 5px;
        }
        .input-section button {
            background-color: #0066cc;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .input-section button:hover {
            background-color: #004d99;
        }
        .result-section {
            text-align: center;
            margin-top: 20px;
        }
        .success {
            color: green;
            font-size: 20px;
            font-weight: bold;
        }
        .warning {
            color: red;
            font-size: 20px;
            font-weight: bold;
        }
        footer {
            background-color: #003366;
            color: white;
            text-align: center;
            padding: 10px 0;
            position: fixed;
            bottom: 0;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation Bar
st.markdown(
    """
    <div class="nav-bar">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/MMU_Logo.png" alt="MMU Logo">
        <h1>PHISHING URL DETECTION USING MACHINE LEARNING</h1>
        <div class="nav-links">
            <a href="#upload-dataset">Upload Dataset</a>
            <a href="#predict-url">Predict URL</a>
            <a href="#performance-analysis">Performance Analysis</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input Section
st.markdown(
    """
    <div class="input-section">
        <h2>ENTER URL:</h2>
        <form action="" method="get">
            <input type="text" placeholder="https://example.com" id="url-input" name="url-input">
            <button type="submit">CHECK</button>
        </form>
    </div>
    """,
    unsafe_allow_html=True,
)

url_input = st.text_input("", placeholder="Enter a URL here", label_visibility="collapsed")

# Check URL Section
if st.button("CHECK"):
    if url_input:
        try:
            # Extract features using the FeatureExtraction class
            extractor = FeatureExtraction(url_input)
            features = extractor.getFeaturesList()

            # Convert features to a DataFrame
            feature_names = [
                'IsHTTPS', 'TLD', 'URLLength', 'NoOfSubDomain', 'NoOfDots', 'NoOfObfuscatedChar',
                'NoOfEqual', 'NoOfQmark', 'NoOfAmp', 'NoOfDigits', 'LineLength', 'HasTitle',
                'HasMeta', 'HasFavicon', 'HasExternalFormSubmit', 'HasCopyright', 'HasSocialNetworking',
                'HasPasswordField', 'HasSubmitButton', 'HasKeywordBank', 'HasKeywordPay', 'HasKeywordCrypto',
                'NoOfPopup', 'NoOfiFrame', 'NoOfImage', 'NoOfJS', 'NoOfCSS', 'NoOfURLRedirect',
                'NoOfHyperlink', 'SuspiciousCharRatio', 'URLComplexityScore', 'HTMLContentDensity', 'InteractiveElementDensity'
            ]
            obj = np.array(features).reshape(1, len(feature_names))
            df = pd.DataFrame(obj, columns=feature_names)

            # Encode the TLD column
            tld_encoder = LabelEncoder()
            df['TLD'] = tld_encoder.fit_transform(df['TLD'])
            x = df.to_numpy()

            # Predict using the model
            y_prob_phishing = model.predict_proba(x)[0, 1]

            # Display the result
            if y_prob_phishing >= 0.99:
                st.markdown(
                    "<div class='result-section warning'>URL does not look secure! It might be harmful and unsafe to visit.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='result-section success'>URL looks secure and safe to visit.</div>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"An error occurred during feature extraction or prediction: {e}")
    else:
        st.warning("Please enter a URL.")

# Footer
st.markdown(
    """
    <footer>
        <p>Developed by Ari Kustiawan</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
