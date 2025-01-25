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
st.title("Phishing URL Detection Using Machine Learning")

# Input Section
st.write("Enter a URL to classify as Legitimate or Phishing:")
url_input = st.text_input("URL")

if st.button("Check URL"):
    if url_input:
        try:
            # Extract features using the FeatureExtraction class
            extractor = FeatureExtraction(url_input)
            st.write("extractor ok")
            features = extractor.getFeaturesList()
            st.write("feature ok")
            # Convert features to a DataFrame (expected input format for the model)
            #feature_names = [
                #'IsHTTPS', 'TLD', 'URLLength', 'NoOfSubDomain', 'NoOfDots', 'NoOfObfuscatedChar', 
                
                #'NoOfEqual', 'NoOfQmark', 'NoOfAmp', 'NoOfDigits', 'LineLength', 'HasTitle',
                #'HasMeta', 'HasFavicon', 'HasExternalFormSubmit', 'HasCopyright', 'HasSocialNetworking',
               # 'HasPasswordField', 'HasSubmitButton', 'HasKeywordBank', 'HasKeywordPay', 'HasKeywordCrypto',
              #  'NoOfPopup', 'NoOfiFrame', 'NoOfImage', 'NoOfJS', 'NoOfCSS', 'NoOfURLRedirect',
             #   'NoOfHyperlink', 'SuspiciousCharRatio', 'URLComplexityScore', 'HTMLContentDensity', 'InteractiveElementDensity'
            #]
            #features_df = pd.DataFrame([features])

            obj = np.array(extractor.getFeaturesList()).reshape(1,33) 
            df = pd.DataFrame(obj)
            
            d = defaultdict(LabelEncoder)
            df = df.apply(lambda x: d[x.name].fit_transform(x))

            x = df.to_numpy()
            # Use the model to predict
            y = model.predict(x)[0] 
            st.write("predict ok")

            y_pro_phishing = model.predict_proba(x)[0,0]
            y_pro_non_phishing = model.predict_proba(x)[0,1]
            st.write(y_pro_phishing)
            st.write(y_pro_non_phishing)
            
            # Display the result
            pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
            st.success({pred})
            
            result = "Legitimate" if y_pro_non_phishing == 1 else "Phishing"
            #st.success(f"The URL is classified as: **{result}**")
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
