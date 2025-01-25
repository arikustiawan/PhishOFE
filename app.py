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

# HTML styling and structure
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phishing URL Detection</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background-color: #f0f4f5;
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100vh;
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
        header .logo img {
            height: 40px;
            margin-right: 10px;
            background-color: white;
            padding: 5px;
            border-radius: 5px;
        }
        header .menu a {
            color: white;
            text-decoration: none;
            font-size: 1rem;
        }
        .container {
            text-align: center;
            background-color: #dde5e8;
            width: 60%;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 80px auto 0 auto;
        }
        .container input {
            width: 80%;
            padding: 10px;
            font-size: 1rem;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 20px;
            margin-bottom: 20px;
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
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <img src="logo.jpg" alt="Logo">
            <h1>Phishing URL Detection Using Machine Learning</h1>
        </div>
    </header>
    <div class="container">
        <h2>Enter a URL Below:</h2>
    </div>
</body>
</html>
"""

# Render HTML content
st.markdown(html_template, unsafe_allow_html=True)

# Input Section
url_input = st.text_input("Enter URL")

if st.button("Check URL"):
    if url_input:
        try:
            # Extract features using the FeatureExtraction class
            extractor = FeatureExtraction(url_input)
            features = extractor.getFeaturesList()

            # Define feature names and create DataFrame
            feature_names = [
                'IsHTTPS', 'TLD', 'URLLength', 'NoOfSubDomain', 'NoOfDots', 'NoOfObfuscatedChar',
                'NoOfEqual', 'NoOfQmark', 'NoOfAmp', 'NoOfDigits', 'LineLength', 'HasTitle',
                'HasMeta', 'HasFavicon', 'HasExternalFormSubmit', 'HasCopyright', 'HasSocialNetworking',
                'HasPasswordField', 'HasSubmitButton', 'HasKeywordBank', 'HasKeywordPay', 'HasKeywordCrypto',
                'NoOfPopup', 'NoOfiFrame', 'NoOfImage', 'NoOfJS', 'NoOfCSS', 'NoOfURLRedirect',
                'NoOfHyperlink', 'SuspiciousCharRatio', 'URLComplexityScore', 'HTMLContentDensity', 'InteractiveElementDensity'
            ]

            df = pd.DataFrame([features], columns=feature_names)

            # Encode categorical columns
            tld_encoder = LabelEncoder()
            df['TLD'] = tld_encoder.fit_transform(df['TLD'])

            # Predict using the loaded model
            prediction = model.predict(df)
            prediction_prob = model.predict_proba(df)[0]

            # Display results
            st.write(f"Phishing Probability: {prediction_prob[1] * 100:.2f}%")
            st.write(f"Legitimate Probability: {prediction_prob[0] * 100:.2f}%")
            result = "Phishing" if prediction[0] == 1 else "Legitimate"
            st.success(f"The URL is classified as: **{result}**")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a URL.")
