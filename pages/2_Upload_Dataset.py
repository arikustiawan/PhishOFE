import streamlit as st
import pandas as pd
import os
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn import metrics
from lightgbm import LGBMClassifier

# Streamlit App Title and Sidebar
st.title("Upload Dataset & Model Training")
st.sidebar.image("logo.jpg", use_container_width=True)

# Ensure the 'dataset' folder exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")
    
if "model_results" not in st.session_state:
    st.session_state.model_results = None
    
# File uploader for CSV files
uploaded_file = st.file_uploader("Upload your dataset (CSV file only)", type=["csv"])


# Class Definition Inside Streamlit App
class TrainingPipeline:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.model = None
        self.selected_features = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        """Load dataset from the given path."""
        self.data = pd.read_csv(self.dataset_path)
        print("Dataset loaded successfully!")

    def SuspiciousCharRatio(self, data):
        """Calculate SuspiciousCharRatio."""
        data['SuspiciousCharRatio'] = (
            data['NoOfObfuscatedChar'] +
            data['NoOfEqual'] +
            data['NoOfQmark'] +
            data['NoOfAmp']
        ) / data['URLLength']
        return data

    def URLComplexityScore(self, data):
        """Calculate URL Complexity Score."""
        first_term = (
            data['URLLength'] + 
            data['NoOfSubDomain'] + 
            data['NoOfObfuscatedChar']
        ) / data['URLLength']

        second_term = (
            data['NoOfEqual'] + 
            data['NoOfAmp']
        ) / (data['NoOfQmark'] + 1)

        data['URLComplexityScore'] = first_term + second_term
        return data

    def HTMLContentDensity(self, data):
        """Calculate HTML Content Density."""
        data['HTMLContentDensity'] = (
            data['LineLength'] + data['NoOfImage']
        ) / (
            data['NoOfJS'] + data['NoOfCSS'] + data['NoOfiFrame'] + 1
        )
        return data

    def InteractiveElementDensity(self, data):
        """Calculate Interactive Element Density."""
        data['InteractiveElementDensity'] = (
            data['HasSubmitButton'] +
            data['HasPasswordField'] +
            data['NoOfPopup']
        ) / (
            data['LineLength'] + data['NoOfImage']
        )
        return data

    def add_features(self):
        """Apply all feature engineering functions."""
        self.data = self.SuspiciousCharRatio(self.data)
        self.data = self.URLComplexityScore(self.data)
        self.data = self.HTMLContentDensity(self.data)
        self.data = self.InteractiveElementDensity(self.data)

    def preprocess_data(self):
        """Preprocess the data by encoding and scaling."""
        self.add_features()
        d = defaultdict(LabelEncoder)
        self.data = self.data.apply(lambda x: d[x.name].fit_transform(x))
        print("Data preprocessing complete.")

    def feature_selection(self):
        """Perform feature selection using Boruta."""
        y = self.data['label']
        X = self.data.drop(columns=['label', 'URL'])

        rf = RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_depth=5,
            n_estimators=100  # Use a specific value since 'auto' is invalid
        )
        feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)
        feat_selector.fit(X.values, y.values.ravel())

        self.selected_features = X.columns[feat_selector.support_].tolist()
        print("Feature selection complete. Selected features:", self.selected_features)

        return X[self.selected_features], y

    def split_train_test(self, X, y):
        """Split the data into training and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data split into training and test sets.")

    def train_model(self):
        """Train the LightGBM model."""
        self.model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=-1, random_state=42)
        self.model.fit(self.X_train, self.y_train)

        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        train_acc = metrics.accuracy_score(self.y_train, y_train_pred)
        test_acc = metrics.accuracy_score(self.y_test, y_test_pred)

        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "f1_score": metrics.f1_score(self.y_test, y_test_pred),
            "recall": metrics.recall_score(self.y_test, y_test_pred),
            "precision": metrics.precision_score(self.y_test, y_test_pred)
        }

    def run_training(self):
        """Run the entire training pipeline."""
        self.load_data()
        self.preprocess_data()
        X, y = self.feature_selection()
        self.split_train_test(X, y)
        metrics_result = self.train_model()
        return metrics_result


# File Upload in Streamlit
if uploaded_file is not None:
    # Save the uploaded file in the 'dataset' folder
    file_path = os.path.join("dataset", uploaded_file.name)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File '{uploaded_file.name}' saved to 'dataset' folder.")

        # Read and display dataset preview
        data = pd.read_csv(file_path)
        st.write("### Dataset Preview:")
        st.dataframe(data.head())

        # Train Model
        if st.button("Train Dataset"):
            try:
                # Initialize TrainingPipeline and Train the Model
                pipeline = TrainingPipeline(dataset_path=file_path)
                results = pipeline.run_training()
                st.session_state.model_results = results

                # Display Model Metrics
                st.success("Model training completed successfully!")
                st.write("### Training Metrics:")
                st.write(f"- **Accuracy**: {results['test_accuracy']:.3f}")
                st.write(f"- **Precision**: {results['precision']:.3f}")
                st.write(f"- **Recall**: {results['recall']:.3f}")
                st.write(f"- **F1 Score**: {results['f1_score']:.3f}")

            except Exception as e:
                st.error(f"An error occurred during training: {e}")

    except Exception as e:
        st.error(f"Error saving or processing the file: {e}")

else:
    st.info("Please upload a dataset to get started.")
