import streamlit as st
import pandas as pd
import os
from training import TrainingPipeline 

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

    except Exception as e:
        st.error(f"Error loading the file: {e}")

    # Add a button to train the model
    if st.button("Train Dataset"):
        try:
            try:
                # Initialize the TrainingPipeline class
                pipeline = TrainingPipeline(dataset_path=file_path)
                
                # Execute the training steps
                pipeline.load_data()
                pipeline.add_features()
                pipeline.preprocess_data()
                pipeline.feature_selection()
                pipeline.train_model()
                
                # Display training success
                st.success("Model training completed!")
                
                # Display model performance metrics
                st.write("### Training Metrics:")
                st.write(f"- **Accuracy**: {pipeline.model.score(pipeline.features, pipeline.target):.2f}")
                #st.write("### Visualization:")
                #st.pyplot(pipeline.plot_metrics())  # Plot ROC and Precision-Recall curves
                
            except Exception as e:
                st.error(f"An error occurred during training: {e}")
else:
    st.info("Please upload a dataset to get started.")
