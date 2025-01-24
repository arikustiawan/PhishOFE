import streamlit as st
import joblib
from feature import FeatureExtraction
import pandas as pd

# Load the trained model
model = joblib.load("model.pkl")

# Page Config
st.set_page_config(page_title="Phishing URL Detection", layout="centered")
st.title("Phishing URL Detection Using Machine Learning")

# Input Section
st.write("Enter a URL to classify as Legitimate or Phishing:")
url_input = st.text_input("URL")

if st.button("Check URL"):
    if url_input:
        try:
            # Extract features using the FeatureExtraction class
            extractor = FeatureExtraction(url_input)
            features = extractor.getFeaturesList()
            
            # Convert features to a DataFrame (expected input format for the model)
            feature_names = [
                "isHttps", "isDomainIp", "tld", "URLlength", "NoOfSubdomain",
                "NoOfDots", "NoOfObfuscatedChar", "NoOfEqual", "NoOfQmark", "NoOfAmp", 
                "NoOfDigits", "LineLength", "hasTitle", "hasMeta", "hasFavicon", 
                "hasExternalFormSubmit", "hasCopyright", "hasSocialNetworking", 
                "hasPasswordField", "hasSubmitButton", "hasKeywordBank", 
                "hasKeywordPay", "hasKeywordCrypto", "NoOfPopup", "NoOfiframe", 
                "NoOfImage", "NoOfJS", "NoOfCSS", "NoOfURLRedirect", 
                "NoOfHyperlink", "SuspiciousCharRatio", "URLComplexityScore", 
                "HTMLContentDensity", "InteractiveElementDensity"
            ]
            features_df = pd.DataFrame([features], columns=feature_names)
            
            # Use the model to predict
            prediction = model.predict(features_df.iloc[:, 1:])[0]  # Skip the URL itself
            
            # Display the result
            result = "Legitimate" if prediction == 0 else "Phishing"
            st.success(f"The URL is classified as: **{result}**")
        except Exception as e:
            st.error(f"An error occurred: {e}")
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
