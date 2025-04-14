Customer Churn Prediction

This project focuses on predicting customer churn using machine learning techniques. It includes data preprocessing, feature selection, model training using Random Forest, XGBoost, and Neural Networks, followed by evaluation and visualization.

🔍 Problem Statement

Customer churn is a significant challenge for businesses. The objective of this project is to build a model that accurately predicts whether a customer will leave or stay, using features like age, purchase behavior, and account activity.

📊 Dataset

Contains 900 customer records with the following features:

Names, Age, Total_Purchase, Account_Manager, Years, Num_Sites, Onboard_date, Location, Company, Churn

Target: Churn (1 = churned, 0 = retained)

🛠️ Tech Stack

Python Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Keras/TensorFlow

Visualization: Seaborn, Matplotlib, Tableau

🤖 ML Models Used

Random Forest

High accuracy for majority class

Low recall for churned class

XGBoost

Better balance between precision and recall

Higher F1 and AUC scores

Neural Network

High recall for churned class

Lower overall accuracy due to imbalance

📉 Results

Model

Accuracy

Precision (1)

Recall (1)

F1 Score (1)

AUC

Random Forest

86%

1.00

0.13

0.24

0.57

XGBoost

87%

0.64

0.53

0.58

0.74

Neural Network

19%

0.16

0.90

0.27

0.47

Note: Performance is limited by class imbalance (only ~17% of customers churned).

📁 Project Structure

├── churn_prediction.py         # Model building and evaluation script
├── churn_data.csv              # Dataset
├── requirements.txt            # Python packages
├── README.md                   # Project overview
└── visualizations/             # Tableau and plot images

📊 Tableau Dashboard

Includes visualizations of:

Churn distribution

Feature correlation

Purchase behavior segmented by churn

🚀 Future Work

Address class imbalance using SMOTE or ADASYN

Improve models with hyperparameter tuning

Try ensemble models for performance boost


✨ Author

Sakthi Pandi — www.linkedin.com/in/sakthi-pandi-320a31268