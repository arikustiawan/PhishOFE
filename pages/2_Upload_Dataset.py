import streamlit as st
import pandas as pd

st.title("Upload Dataset")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV file only)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        # Show dataset information
        st.write("Dataset Information:")
        st.write(data.info())
    except Exception as e:
        st.error(f"Error loading the file: {e}")
else:
    st.info("Please upload a dataset to get started.")
