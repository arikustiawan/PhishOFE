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
st.set_page_config(page_title="Phishing URL Detection", layout="centered")

# Custom CSS styling to make the header and footer full width, center the content
st.markdown(
    """
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background-color: #f0f4f5;
            display: flex;
            flex-direction: column;
            align-items: center;
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
        }

        header h1 {
            margin: 0;
            font-size: 1.5rem;
        }

        header .menu a {
            color: white;
            text-decoration: none;
            font-size: 1rem;
            margin-right: 15px;
        }

        .container {
            text-align: center;
            background-color: #dde5e8;
            width: 60%;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: auto;
            margin-top: 100px;
        }

        .container h2 {
            font-size: 1.25rem;
            color: #004b93;
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
    </style>
    """,
    unsafe_allow_html=True,
)

# Header section
st.markdown(
    """
    <header>
        <div class="logo">
            <h3>PHISHING URL DETECTION USING MACHINE LEARNING</h3>
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
    """,
    unsafe_allow_html=True,
)

url_input = st.text_input("Enter URL", key="url_input")

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
            y_prob = model.predict_proba(df.to_numpy())[0]

            # Display the result
            phishing_prob = y_prob[1] * 100
            legitimate_prob = y_prob[0] * 100
            st.markdown(f"<div id='result-message'><strong>Phishing Probability:</strong> {phishing_prob:.2f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div id='result-message'><strong>Legitimate Probability:</strong> {legitimate_prob:.2f}%</div>", unsafe_allow_html=True)

            result = "Phishing" if phishing_prob >= 99 else "Legitimate"
            st.success(f"The URL is classified as: **{result}**")

        except Exception as e:
            st.error(f"An error occurred: {e}")
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
