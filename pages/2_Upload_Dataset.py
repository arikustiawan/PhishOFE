import streamlit as st
import pandas as pd
import os

st.title("Upload Dataset")
st.sidebar.image("logo.jpg", use_container_width=True)

if not os.path.exists("dataset"):
    os.makedirs("dataset")
    
# Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV file only)", type=["csv"])

if uploaded_file is not None:
    # Save the file to the 'dataset' folder
    file_path = os.path.join("dataset", uploaded_file.name)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
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
        st.text(data.info())
    except Exception as e:
        st.error(f"Error loading the file: {e}")
else:
    st.info("Please upload a dataset to get started.")
