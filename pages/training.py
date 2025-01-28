import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics 
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier


df = pd.read_csv("dataset/Legitimate_Phishing_Dataset.csv")

def SuspiciousCharRatio(data):
    data['SuspiciousCharRatio'] = (
            data['NoOfObfuscatedChar'] +
            data['NoOfEqual'] +
            data['NoOfQmark'] +
            data['NoOfAmp']
        ) / data['URLLength']
    return data

def SuspiciousCharRatio(data):
    data['SuspiciousCharRatio'] = (
            data['NoOfObfuscatedChar'] +
            data['NoOfEqual'] +
            data['NoOfQmark'] +
            data['NoOfAmp']
        ) / data['URLLength']
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

df = SuspiciousCharRatio(df)
df = URLComplexityScore(df)
df = HTMLContentDensity(df)
df = InteractiveElementDensity(df)

d = defaultdict(LabelEncoder)
df = df.apply(lambda x: d[x.name].fit_transform(x))
df_FS = df.copy()

# Define 'y' as the target variable (label column)
y = df_FS['label']
X = df_FS.drop(columns=['label','URL'])

# Define a ranking function
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))

# Initialize RandomForestClassifier for Boruta
rf = RandomForestClassifier(
    n_jobs=-1, 
    class_weight="balanced_subsample", 
    max_depth=5, 
    n_estimators="auto"  # Adjust n_estimators to a specific value since 'auto' is not valid for this parameter
)

# Initialize Boruta Feature Selector
feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)

# Fit Boruta to the dataset
feat_selector.fit(X.values, y.values.ravel())

# Generate feature rankings
boruta_score = ranking(
    list(map(float, feat_selector.ranking_)), 
    X.columns,  # Use X.columns to provide feature names
    order=-1
)

# Convert rankings into a DataFrame
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
boruta_score = boruta_score.sort_values("Score", ascending=False)

selected_features = X.columns[feat_selector.support_].tolist()

# Use only the selected features
X_selected = X[selected_features]

# Split for model training
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Instantiate the model
lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=-1, random_state=42)

# Fit the model
lgbm.fit(X_train, y_train)

# Predicting the target value from the model for the samples
y_train_lgbm = lgbm.predict(X_train)
y_test_lgbm = lgbm.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance

acc_test_lgbm = metrics.accuracy_score(y_test,y_test_lgbm)
f1_score_test_lgbm = metrics.f1_score(y_test,y_test_lgbm)
recall_score_test_lgbm = metrics.recall_score(y_test,y_test_lgbm)
precision_score_test_lgbm = metrics.precision_score(y_test,y_test_lgbm)

print("Accuracy: {:.3f}".format(acc_test_lgbm))
print("Precision: {:.3f}".format(precision_score_test_lgbm))
print("Recall: {:.3f}".format(recall_score_test_lgbm))
print("F1 Score: {:.3f}".format(f1_score_test_lgbm))
