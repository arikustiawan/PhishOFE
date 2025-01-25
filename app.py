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
            position: absolute;
            top: 0;
            left: 0;
        }
        
        header h1 {
            margin: 0;
            font-size: 1.5rem;
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
        header .menu {
            display: flex;
            gap: 15px;
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
            margin: auto;
            margin-top: calc(50vh - 150px);
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
        
        footer {
            width: 100%;
            background-color: #004b93;
            color: white;
            text-align: center;
            padding: 10px 0;
            position: absolute;
            bottom: 0;
            left: 0;
        }
        
        footer p {
            margin: 0;
        }

    </style>
</head>
    """,
    unsafe_allow_html=True,
)

# Header section
st.markdown(
    """
    <body>
    <header>
        <div class="logo">
            <img src="logo.jpg" alt="MMU Logo">
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
    </body>
    """,
    unsafe_allow_html=True,
)
