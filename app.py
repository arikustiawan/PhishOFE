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

# Set up the Streamlit app
st.set_page_config(page_title="Phishing URL Detection", layout="centered")
st.markdown(
    """
    <style>
    header {
        background-color: #004b93;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 1.5rem;
    }
    .stTextInput > div > label {
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<header>PHISHING URL DETECTION USING MACHINE LEARNING</header>", unsafe_allow_html=True)

# Input Section
st.write("Enter a URL to classify as Legitimate or Phishing:")
url_input = st.text_input("URL", placeholder="Enter the URL here")

if st.button("Check URL"):
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
            feature_array = np.array(features).reshape(1, len(feature_names))
            df = pd.DataFrame(feature_array, columns=feature_names)

            # Label Encoding
            tld_encoder = LabelEncoder()
            df['TLD'] = tld_encoder.fit_transform(df['TLD'])
            df_encoded = df.copy()

            # Predict using the model
            x = df_encoded.to_numpy()
            y = model.predict(x)
            y_prob_phishing = model.predict_proba(x)[0, 1]
            y_prob_non_phishing = model.predict_proba(x)[0, 0]

            # Display the result
            pred_phishing = y_prob_phishing * 100
            pred_legitimate = y_prob_non_phishing * 100
            st.write(f"Phishing Probability: {pred_phishing:.2f}%")
            st.write(f"Legitimate Probability: {pred_legitimate:.2f}%")
            result = "Phishing" if pred_phishing >= 50 else "Legitimate"
            
            if result == "Phishing":
                st.error(f"The URL is classified as: **{result}**")
            else:
                st.success(f"The URL is classified as: **{result}**")
        except Exception as e:
            st.error(f"An error occurred during feature extraction or prediction: {e}")
    else:
        st.warning("Please enter a URL.")

# Footer
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    </style>
    <footer>
    <p>Developed by Ari Kustiawan</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
