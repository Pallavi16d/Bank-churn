import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV

# Step 1: Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV

# Step 2: Import Dataset
df = pd.read_csv("Bank_churn_modelling.csv")

# Step 3: Analyze Data
print(df.head())  # Display the first few rows of the dataset
print(df.info())  # Get information about the dataset (data types, missing values, etc.)
print(df.describe())  # Summary statistics of the dataset

# Step 4: Data Encoding (one-hot encoding for categorical variables)
df_encoded = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Step 5: Define Label and Features
X = df_encoded.drop(columns=['Churn'])  # Use 'Churn' instead of 'Exited'
y = df_encoded['Churn']  # Use 'Churn' instead of 'Exited'

# Step 6: Handling Imbalanced Data
# You can choose either undersampling or oversampling, or a combination of both.

# Random Under Sampling
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X, y)

# Random Over Sampling
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)

# Step 7: Train-Test Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 8: Standardize Features
X_train = X_train.drop(columns=['Surname'])  # Remove the 'Surname' column
X_test = X_test.drop(columns=['Surname'])    # Remove the 'Surname' column

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Support Vector Machine Classifier with Raw Data
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)

# Step 10: Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with raw data:", accuracy)

# Step 11: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'sigmoid']
}
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# Step 12: Model with Random Under Sampling
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

# Step 13: Model Accuracy after Hyperparameter Tuning
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with random under sampling and hyperparameter tuning:", accuracy)

# Step 14: Model with Random Over Sampling
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

# Step 15: Model Accuracy after Hyperparameter Tuning
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with random over sampling and hyperparameter tuning:", accuracy)
