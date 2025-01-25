import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from feature import FeatureExtraction

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
st.set_page_config(page_title="Phishing URL Detection", layout="wide")
st.markdown(
    """
    <style>
    header {
        background-color: #004b93 !important;
        color: white;
    }
    footer {
        background-color: #004b93 !important;
        color: white;
    }
    h1 {
        color: #004b93;
    }
    .stTextInput > div > label {
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom header for consistency with HTML
st.markdown(
    """
    <header style="display: flex; justify-content: space-between; align-items: center; background-color: #004b93; padding: 10px;">
        <div style="display: flex; align-items: center;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/7/77/MMU_Malaysia_logo.png" alt="MMU Logo" style="height: 50px; margin-right: 10px; background-color: white; padding: 5px; border-radius: 5px;">
            <h1 style="margin: 0; font-size: 1.5rem; color: white;">PHISHING URL DETECTION USING MACHINE LEARNING</h1>
        </div>
        <nav style="display: flex; gap: 15px;">
            <a href="#" style="color: white; text-decoration: none;">Upload Dataset</a>
            <a href="#" style="color: white; text-decoration: none;">Predict URL</a>
            <a href="#" style="color: white; text-decoration: none;">Performance Analysis</a>
        </nav>
    </header>
    """,
    unsafe_allow_html=True,
)

# Input section
st.header("Enter URL:")
url_input = st.text_input("URL", placeholder="Enter the URL for classification")

if st.button("Check URL"):
    if url_input:
        try:
            # Feature extraction
            extractor = FeatureExtraction(url_input)
            features = extractor.getFeaturesList()
            feature_names = [
                'IsHTTPS', 'TLD', 'URLLength', 'NoOfSubDomain', 'NoOfDots', 'NoOfObfuscatedChar', 
                'NoOfEqual', 'NoOfQmark', 'NoOfAmp', 'NoOfDigits', 'LineLength', 'HasTitle',
                'HasMeta', 'HasFavicon', 'HasExternalFormSubmit', 'HasCopyright', 'HasSocialNetworking',
                'HasPasswordField', 'HasSubmitButton', 'HasKeywordBank', 'HasKeywordPay', 'HasKeywordCrypto',
                'NoOfPopup', 'NoOfiFrame', 'NoOfImage', 'NoOfJS', 'NoOfCSS', 'NoOfURLRedirect',
                'NoOfHyperlink', 'SuspiciousCharRatio', 'URLComplexityScore', 'HTMLContentDensity', 'InteractiveElementDensity'
            ]
            feature_array = np.array(features).reshape(1, len(feature_names))
            df = pd.DataFrame(feature_array, columns=feature_names)
            df['TLD'] = LabelEncoder().fit_transform(df['TLD'])

            # Model prediction
            prediction = model.predict(df)
            prediction_proba = model.predict_proba(df)
            phishing_prob = prediction_proba[0][1] * 100
            legitimate_prob = prediction_proba[0][0] * 100

            # Display result
            if phishing_prob >= 50:
                st.error(f"URL does not look secure! It might be harmful and unsafe to visit.")
            else:
                st.success("URL looks secure and safe to visit.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a URL.")

# Footer
st.markdown(
    """
    <footer style="background-color: #004b93; color: white; text-align: center; padding: 10px;">
        <p>Developed by Ari Kustiawan</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
