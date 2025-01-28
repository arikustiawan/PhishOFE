import streamlit as st
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
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
            # Initialize the TrainingPipeline class
            pipeline = TrainingPipeline(file_path)

            # Run the full pipeline
            results = pipeline.run_pipeline()

            # Display training metrics
            st.success("Model training completed successfully!")
            st.write("### Training Metrics:")
            st.write(f"- **Accuracy**: {results['test_accuracy']:.3f}")
            st.write(f"- **Precision**: {results['precision']:.3f}")
            st.write(f"- **Recall**: {results['recall']:.3f}")
            st.write(f"- **F1 Score**: {results['f1_score']:.3f}")
                
        except Exception as e:
            st.error(f"An error occurred during training: {e}")
        except Exception as e:
            st.error(f"Error loading the file: {e}")
else:
    st.info("Please upload a dataset to get started.")
