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

# Set up Streamlit page configuration
st.set_page_config(page_title="Phishing URL Detection", layout="wide")

# Custom CSS to match the design
st.markdown(
    """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #e6e6e6;
        }
        header {
            width: 100%;
            background-color: #004b93;
            padding: 10px 20px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: fixed; 
            top: 5; 
            left: 0; 
            z-index: 1000; 
            height: 100px;
        }
        header .logo {
            display: flex;
            align-items: center;
        }
        header .logo img {
            height: 50px;
            margin-right: 10px;
        }
        header h1 {
            font-size: 1.5rem;
            margin: 0;
        }
        nav {
            display: flex;
            gap: 20px;
        }
        nav a {
            color: white;
            text-decoration: none;
            font-size: 1rem;
        }
        nav a:hover {
            text-decoration: underline;
        }
        .container {
            display: flex; /* Activate Flexbox */
            justify-content: center; /* Center items horizontally */
            align-items: center; /* Center items vertically */
            text-align: center; /* Optional: Center-align text inside */
            margin-top: 150px;
         
        }
        .container h2 {
            font-size: 1.25rem;
            color: #004b93;
            margin-bottom: 20px;
            text-align: center;
        }
        .container input {
            width: 80%;
            padding: 10px;
            font-size: 1rem;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .container button {
            background-color: #004b93;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 1rem;
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
            position: fixed; 
            bottom: 0; 
            left: 0; 
            z-index: 1000; 
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation bar
st.markdown(
    """
    <header>
        <div class="logo">
            <h1>PHISHING URL DETECTION USING MACHINE LEARNING</h1>
        </div>
        <nav>
            <a href="#">Upload Dataset</a>
            <a href="#">Predict URL</a>
            <a href="#">Performance Analysis</a>
        </nav>
    </header>
    """,
    unsafe_allow_html=True,
)

# Main container for URL input and results
st.markdown(
    """
    <div class="container">
        <h2>ENTER URL:</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

url_input = st.text_input("", placeholder="Enter URL here", label_visibility="collapsed")

# URL checking button
if st.button("CHECK"):
    if url_input:
        try:
            # Feature extraction
            extractor = FeatureExtraction(url_input)
            features = extractor.getFeaturesList()

            # Create DataFrame
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

            # Encode TLD column
            tld_encoder = LabelEncoder()
            df['TLD'] = tld_encoder.fit_transform(df['TLD'])
            x = df.to_numpy()

            # Prediction
            y_prob_phishing = model.predict_proba(x)[0, 1]

            # Display result
            #if y_prob_phishing >= 0.99:
             #   st.warning("URL does not look secure! It might be harmful and unsafe to visit.")
            #else:
            #    st.success("URL looks secure and safe to visit.")

            if y_prob_phishing >= 0.99:
                st.markdown(
                    "<p style='color: red; font-size: 18px; font-weight: bold;'>URL does not look secure! It might be harmful and unsafe to visit.</p>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<p style='color: green; font-size: 18px; font-weight: bold;'>URL looks secure and safe to visit.</p>",
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
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
