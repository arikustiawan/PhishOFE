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
    </div>
    """,
    unsafe_allow_html=True,
)

url_input = st.text_input("", placeholder="Type your URL here...")

# Prediction logic
if st.button("Check URL"):
    if url_input:
        try:
            # Extract features using the FeatureExtraction class
            extractor = FeatureExtraction(url_input)
            features = extractor.getFeaturesList()

            # Convert features to a DataFrame (expected input format for the model)
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
            df_encoded = df.copy()

            # Use the model to predict
            x = df_encoded.to_numpy()
            y = model.predict(x)
            y_prob_phishing = model.predict_proba(x)[0, 1]
            y_prob_non_phishing = model.predict_proba(x)[0, 0]

            # Display the results
            pred_phishing = y_prob_phishing * 100
            pred_legitimate = y_prob_non_phishing * 100
            st.success(f"Phishing Probability: {pred_phishing:.2f}%")
            st.success(f"Legitimate Probability: {pred_legitimate:.2f}%")
            result = "Phishing" if pred_phishing >= 99 else "Legitimate"
            st.success(f"The URL is classified as: **{result}**")
        except Exception as e:
            st.error(f"An error occurred during feature extraction or prediction: {e}")
    else:
        st.warning("Please enter a URL.")

# Footer
st.markdown(
    """
    <div class="footer">
        Developed by Ari Kustiawan
    </div>
    """,
    unsafe_allow_html=True,
)
