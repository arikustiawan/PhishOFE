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

# Set up Streamlit page
st.set_page_config(page_title="Phishing URL Detection", layout="wide")

# Custom CSS for the design
st.markdown(
    """
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background-color: #f0f4f5;
        }
        header {
            width: 100%;
            background-color: #004b93;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 20px;
            box-sizing: border-box;
            position: fixed;
            top: 0;
            left: 0;
            z-index: 1000;
        }
        header .logo {
            display: flex;
            align-items: center;
        }
        header .logo img {
            height: 40px;
            margin-right: 10px;
            background-color: white;
            padding: 5px;
            border-radius: 5px;
        }
        header h1 {
            font-size: 1.5rem;
            margin: 0;
        }
        header .menu {
            display: flex;
            gap: 15px;
        }
        header .menu a {
            color: white;
            text-decoration: none;
            font-size: 1rem;
        }
        header .menu a:hover {
            text-decoration: underline;
        }
        .container {
            background-color: #dde5e8;
            width: 60%;
            margin: 100px auto;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .container h2 {
            font-size: 1.25rem;
            color: #004b93;
        }
        .container input {
            width: 80%;
            padding: 10px;
            font-size: 1rem;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin: 20px 0;
        }
        .container button {
            padding: 10px 20px;
            font-size: 1rem;
            background-color: #004b93;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .container button:hover {
            background-color: #003766;
        }
        footer {
            width: 100%;
            background-color: #004b93;
            color: white;
            text-align: center;
            padding: 10px 0;
            position: fixed;
            bottom: 0;
            left: 0;
        }
        footer p {
            margin: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header section
st.markdown(
    """
    <header>
        <div class="logo">
            <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/MMU_Logo.png" alt="MMU Logo">
            <h1>PHISHING URL DETECTION USING MACHINE LEARNING</h1>
        </div>
        <nav class="menu">
            <a href="#">Upload Dataset</a>
            <a href="#">Predict URL</a>
            <a href="#">Performance Analysis</a>
        </nav>
    </header>
    """,
    unsafe_allow_html=True,
)

# Input Section
st.markdown(
    """
    <div class="container">
        <h2>ENTER URL:</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

url_input = st.text_input("", placeholder="Enter URL here", label_visibility="collapsed")

# Check URL Section
if st.button("CHECK"):
    if url_input:
        try:
            # Feature extraction
            extractor = FeatureExtraction(url_input)
            features = extractor.getFeaturesList()

            # Create DataFrame
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

            # Encode TLD column
            tld_encoder = LabelEncoder()
            df['TLD'] = tld_encoder.fit_transform(df['TLD'])
            x = df.to_numpy()

            # Prediction
            y_prob_phishing = model.predict_proba(x)[0, 1]

            # Display result
            if y_prob_phishing >= 0.99:
                st.warning("URL does not look secure! It might be harmful and unsafe to visit.")
            else:
                st.success("URL looks secure and safe to visit.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
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
