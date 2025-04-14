# Customer Churn Prediction Project

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset
df = pd.read_csv(r'C:\Users\sakth\Customer churn ML\customer_churn.csv')  # Replace with your dataset path

# Basic EDA
print(df.head())
print(df.info())
print(df.describe())

# Preprocessing
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])  # Yes/No -> 1/0

# Drop unnecessary columns
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Encode categorical variables
df = pd.get_dummies(df)

# Scale numerical data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Churn', axis=1))
X = pd.DataFrame(scaled_features, columns=df.drop('Churn', axis=1).columns)
y = df['Churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model training
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Visualization examples
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.show()

# Export for Tableau
df['Prediction'] = models['Random Forest'].predict(X)
df.to_csv('churn_predictions.csv', index=False)
