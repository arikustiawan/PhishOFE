import streamlit as st
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
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

# Set up the Streamlit app with custom page layout
st.set_page_config(page_title="Phishing URL Detection", layout="wide")

st.markdown(
    """
    <style>
    .center-button {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="background-color:#0044cc; padding:10px; border-radius:5px;">
        <h1 style="color:white; text-align:center;">PHISHING URL DETECTION USING MACHINE LEARNING</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation menu (placeholder for options)
st.markdown(
    """
    <div style="text-align:right; margin-bottom:20px;">
        <a href="#" style="color:#0044cc; margin-right:20px; text-decoration:none; font-weight:bold;">Upload Dataset</a>
        <a href="#" style="color:#0044cc; margin-right:20px; text-decoration:none; font-weight:bold;">Predict URL</a>
        <a href="#" style="color:#0044cc; text-decoration:none; font-weight:bold;">Performance Analysis</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input Section
st.markdown(
    """
    <div style="text-align:center; margin-top:50px;">
        <h3>ENTER URL:</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

url_input = st.text_input(
    "",
    placeholder="Enter URL here",
    label_visibility="collapsed",
    key="url_input",
    help="This is a custom input box"
)
st.markdown('<div class="center-button">', unsafe_allow_html=True)
if st.button("CHECK", type="primary"):
    st.markdown('</div>', unsafe_allow_html=True)
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
            df_encoded = df.copy()

            x = df_encoded.to_numpy()

            # Use the model to predict
            y = model.predict(x)
            y_prob_phishing = model.predict_proba(x)[0, 1]
            y_prob_non_phishing = model.predict_proba(x)[0, 0]

            # Display the result
            pred = y_prob_phishing * 100
            pred2 = y_prob_non_phishing * 100
            result = "Phishing" if pred >= 99 else "Legitimate"
            st.markdown(f"<h3 style='text-align:center;'>The URL is classified as: <b>{result}</b></h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during feature extraction or prediction: {e}")
    else:
        st.warning("Please enter a URL.")

# Footer
st.markdown(
    """
    <div style="text-align:center; margin-top:50px; background-color:#0044cc; padding:10px; border-radius:5px;">
        <p style="color:white;">Developed by Ari Kustiawan</p>
    </div>
    """,
    unsafe_allow_html=True,
)
