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
st.set_page_config(page_title="Phishing URL Detection", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .header {
        background-color: #002F6C;
        color: white;
        padding: 10px;
        text-align: center;
    }
    .footer {
        background-color: #002F6C;
        color: white;
        padding: 10px;
        text-align: center;
        position: fixed;
        bottom: 0;
        width: 100%;
    }
    .input-section {
        margin: 20px auto;
        text-align: center;
    }
    .output-section {
        margin: 20px auto;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="header"><h2>PHISHING URL DETECTION USING MACHINE LEARNING</h2></div>', unsafe_allow_html=True)

# Navigation menu
st.markdown(
    """
    <div style="display: flex; justify-content: space-around; padding: 10px; background-color: #002F6C; color: white;">
        <a href="#" style="color: white; text-decoration: none;">Upload Dataset</a>
        <a href="#" style="color: white; text-decoration: none;">Predict URL</a>
        <a href="#" style="color: white; text-decoration: none;">Performance Analysis</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input Section
st.markdown('<div class="input-section"><h3>Enter URL:</h3></div>', unsafe_allow_html=True)
url_input = st.text_input("", placeholder="Enter URL here", key="url_input")

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

            # Use the model to predict
            x = df.to_numpy()
            y = model.predict(x)

            y_prob_phishing = model.predict_proba(x)[0, 1]

            # Display the result
            if y_prob_phishing >= 0.99:
                st.markdown(
                    '<div class="output-section" style="color: red;">URL does not look secure! It might be harmful and unsafe to visit.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="output-section" style="color: green;">URL looks secure and safe to visit.</div>',
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"An error occurred during feature extraction or prediction: {e}")
    else:
        st.warning("Please enter a URL.")

# Footer
st.markdown('<div class="footer">Developed by Ari Kustiawan</div>', unsafe_allow_html=True)
