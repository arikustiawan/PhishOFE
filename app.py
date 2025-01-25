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
st.markdown(
    """
    <style>
        header {visibility: hidden;}
        .reportview-container {
            background: #f5f5f5;
        }
        .css-18e3th9 {
            padding-top: 2rem;
        }
        .stButton button {
            background-color: #0066cc;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #004d99;
        }
        .stTextInput {
            margin-bottom: 1rem;
        }
        .message-success {
            color: green;
            font-size: 18px;
        }
        .message-warning {
            color: red;
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top navigation bar
st.markdown(
    """
    <div style="background-color:#003366; padding:10px;">
        <h2 style="color:white; text-align:center; margin: 0;">PHISHING URL DETECTION USING MACHINE LEARNING</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input Section
st.markdown("<h3 style='text-align:center;'>ENTER URL:</h3>", unsafe_allow_html=True)
url_input = st.text_input("", placeholder="https://example.com", key="url_input", label_visibility="collapsed")

if st.button("CHECK", key="check_url", help="Check if the URL is safe or phishing"):
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
            y_prob_legitimate = model.predict_proba(x)[0, 0]

            # Display the result
            if y_prob_phishing >= 0.99:
                st.markdown(
                    f"<p class='message-warning'>URL does not look secure! It might be harmful and unsafe to visit.</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<p class='message-success'>URL looks secure and safe to visit.</p>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"An error occurred during feature extraction or prediction: {e}")
    else:
        st.warning("Please enter a URL.")

# Footer
st.markdown(
    """
    <footer style="background-color:#003366; padding:10px; text-align:center; color:white;">
        <p>Developed by Ari Kustiawan</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
