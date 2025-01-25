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

# Page configuration
st.set_page_config(page_title="Phishing URL Detection", layout="centered")

# Header
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
            text-align: center;
            font-size: 22px;
            font-weight: bold;
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
            width: 50%;
            margin: 20px auto;
            text-align: center;
        }
        .input-box input {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .button-container {
            margin-top: 10px;
            text-align: center;
        }
        .check-button {
            background-color: #003399;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .check-button:hover {
            background-color: #002266;
        }
    </style>
    <div class="header">
        PHISHING URL DETECTION USING MACHINE LEARNING
    </div>
    """,
    unsafe_allow_html=True,
)

# Main content
st.markdown("<h3 style='text-align: center;'>Enter URL:</h3>", unsafe_allow_html=True)
url_input = st.text_input("")

# Check button
if st.button("Check URL"):
    if url_input:
        try:
            # Feature extraction
            extractor = FeatureExtraction(url_input)
            features = extractor.getFeaturesList()

            # Data preparation
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

            # TLD encoding
            tld_encoder = LabelEncoder()
            df['TLD'] = tld_encoder.fit_transform(df['TLD'])
            x = df.to_numpy()

            # Prediction
            y = model.predict(x)
            y_prob_phishing = model.predict_proba(x)[0, 1] * 100
            y_prob_legitimate = model.predict_proba(x)[0, 0] * 100
            result = "Phishing" if y_prob_phishing > 99 else "Legitimate"

            # Display results
            st.success(f"Phishing Probability: {y_prob_phishing:.2f}%")
            st.success(f"Legitimate Probability: {y_prob_legitimate:.2f}%")
            st.success(f"The URL is classified as: **{result}**")
        except Exception as e:
            st.error(f"An error occurred: {e}")
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
