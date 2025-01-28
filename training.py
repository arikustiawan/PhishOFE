import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from lightgbm import LGBMClassifier
from collections import defaultdict


class TrainingPipeline:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.model = None
        self.features = None
        self.target = None

    def load_data(self):
        """Load the dataset from the provided path."""
        self.data = pd.read_csv(self.dataset_path)
        print("Dataset loaded successfully!")

    def add_features(self):
        """Add derived features to the dataset."""
        self.data['SuspiciousCharRatio'] = (
            self.data['NoOfObfuscatedChar']
            + self.data['NoOfEqual']
            + self.data['NoOfQmark']
            + self.data['NoOfAmp']
        ) / self.data['URLLength']

        print("Derived features added.")

    def preprocess_data(self):
        """Preprocess the dataset by encoding labels and scaling features."""
        label_encoder = LabelEncoder()
        self.data['Label'] = label_encoder.fit_transform(self.data['Label'])
        self.features = self.data.drop('Label', axis=1)
        self.target = self.data['Label']

        scaler = MinMaxScaler()
        self.features = pd.DataFrame(
            scaler.fit_transform(self.features), columns=self.features.columns
        )

        print("Data preprocessing complete.")

    def feature_selection(self):
        """Perform feature selection using Boruta."""
        rf = RandomForestClassifier(n_jobs=-1, max_depth=5)
        boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42)
        boruta_selector.fit(self.features.values, self.target.values)

        # Update features with selected ones
        self.features = self.features.loc[:, boruta_selector.support_]
        print("Feature selection complete.")

    def train_model(self):
        """Train a LightGBM model on the dataset."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        self.model = LGBMClassifier()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print(f"Model training complete! Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
