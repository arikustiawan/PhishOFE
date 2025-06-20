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



# Streamlit page configuration
st.set_page_config(
    page_title="Predict URL",
    layout="wide",
    page_icon="🔗",
    initial_sidebar_state="expanded"  
)
#st.sidebar.title("Predict URL")
# Add a logo to the top of the sidebar
st.sidebar.image("logo.jpg", use_container_width=True)

# Add other sidebar elements

# Main page content
st.title("Phishing URL Detection Using Machine Learning")

# Input Section
#st.write("Enter a URL to classify as Legitimate or Phishing:")
url_input = st.text_input("Enter a URL to classify as Legitimate or Phishing:")

if st.button("Check URL"):
    if url_input:
        try:
            # Extract features using the FeatureExtraction class
            extractor = FeatureExtraction(url_input)
            #st.write("extractor ok")
            features = extractor.getFeaturesList()
            #st.write("feature ok")
            # Convert features to a DataFrame (expected input format for the model)
            feature_names = [
                'IsHTTPS', 'TLD', 'URLLength', 'NoOfSubDomain', 'NoOfDots', 'NoOfObfuscatedChar',   
                'NoOfEqual', 'NoOfQmark', 'NoOfAmp', 'NoOfDigits', 'LineLength', 'HasTitle',
                'HasMeta', 'HasFavicon', 'HasExternalFormSubmit', 'HasCopyright', 'HasSocialNetworking',
                'HasPasswordField', 'HasSubmitButton', 'HasKeywordBank', 'HasKeywordPay', 'HasKeywordCrypto',
                'NoOfPopup', 'NoOfiFrame', 'NoOfImage', 'NoOfJS', 'NoOfCSS', 'NoOfURLRedirect',
                 'NoOfHyperlink', 'SuspiciousCharRatio', 'URLComplexityScore', 'HTMLContentDensity', 'InteractiveElementDensity'
            ]
            #features_df = pd.DataFrame([features])

            obj = np.array(extractor.getFeaturesList()).reshape(1,len(feature_names)) 
            df = pd.DataFrame(obj,columns=feature_names)
            #st.dataframe(df)

            #st.write("Label Encoder")
            tld_encoder = LabelEncoder()

            # Encode the TLD column
            df['TLD'] = tld_encoder.fit_transform(df['TLD'])
            df_encoded = df.copy()

            x = df_encoded.to_numpy()
            #st.dataframe(df_encoded)
            
            # Use the model to predict
            y = model.predict(x)
            #st.write("predict: ",y)

            y_prob_phishing = model.predict_proba(x)[0,1]
            y_prob_non_phishing = model.predict_proba(x)[0,0]
            #st.write(y_pro_phishing)
            #st.write(y_pro_non_phishing)
            
            # Display the result
            pred = y_prob_phishing*100
            pred2 = y_prob_non_phishing*100
            #st.success({pred})
            #st.success({pred2})
            result = "Phishing" if pred == pred else "Legitimate"
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
