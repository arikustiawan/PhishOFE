import streamlit as st
import pandas as pd
import os

st.title("Upload Dataset")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV file only)", type=["csv"])

if uploaded_file is not None:
    # Save the file to the 'dataset' folder
    file_path = os.path.join("dataset", uploaded_file.name)
    try:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        
        # Save the file in the folder
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' saved to 'dataset' folder.")
        
        # Display dataset preview
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        # Show dataset information
        st.write("Dataset Information:")
        st.text(data.info())  # Using st.text for info()
    except Exception as e:
        st.error(f"Error loading the file: {e}")
else:
    st.info("Please upload a dataset to get started.")
