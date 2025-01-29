import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn import metrics
from lightgbm import LGBMClassifier


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

    def SuspiciousCharRatio(data):
    data['SuspiciousCharRatio'] = (
            data['NoOfObfuscatedChar'] +
            data['NoOfEqual'] +
            data['NoOfQmark'] +
            data['NoOfAmp']
        ) / data['URLLength']
    return data

    def URLComplexityScore(data):
        # Calculate the first term: (URLLength + NoOfSubDomain + NoOfObfuscatedChar) / URLLength
        first_term = (
            data['URLLength'] + 
            data['NoOfSubDomain'] + 
            data['NoOfObfuscatedChar']
        ) / data['URLLength']   
    
        # Calculate the second term: (NoOfEqual + NoOfAmp) / (NoOfQmark + 1)
        second_term = (
            data['NoOfEqual'] + 
            data['NoOfAmp']
        ) / (data['NoOfQmark'] + 1)
    
        data['URLComplexityScore'] = first_term + second_term
        
        return data

    def HTMLContentDensity(data):
        data['HTMLContentDensity'] = (
                data['LineLength'] + data['NoOfImage']
            ) / (
                data['NoOfJS'] + data['NoOfCSS'] + data['NoOfiFrame'] + 1
            )    
        return data

    def InteractiveElementDensity(data):
        data['InteractiveElementDensity'] = (
                data['HasSubmitButton'] +
                data['HasPasswordField'] +
                data['NoOfPopup']
            ) / (
                data['LineLength'] + data['NoOfImage']
            )
        return data

    def add_features(self):
        self.data = self.SuspiciousCharRatio(self.data)
        self.data = self.URLComplexityScore(self.data)
        self.data = self.HTMLContentDensity(self.data)
        self.data = self.InteractiveElementDensity(self.data)

    def preprocess_data(self):
        """Preprocess the data by encoding and scaling."""
        self.data = self.add_features(self.data)
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

    def train_test_split(self, X, y):
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

        # Save the trained model using pickle
        with open('model.pkl', 'wb') as model_file:
            pickle.dump(self.model, model_file)
        #print("Model has been saved as 'model.pkl'")

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "f1_score": metrics.f1_score(self.y_test, y_test_pred),
            "recall": metrics.recall_score(self.y_test, y_test_pred),
            "precision": metrics.precision_score(self.y_test, y_test_pred)
        }

    def runTraining(self):
        """Run the entire training pipeline."""
        self.load_data()
        self.add_features()
        self.preprocess_data()
        X, y = self.feature_selection()
        self.train_test_split(X, y)
        metrics = self.train_model()
        return metrics
