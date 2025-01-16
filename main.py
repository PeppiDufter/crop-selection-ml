import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('soil_measures.csv')

# Features and target variable
X = data[['N', 'P', 'K', 'pH']]
y = data['crop']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

best_feature = None
best_score = 0

# Evaluate each feature using Logistic Regression
for feature in X.columns:
    # Use only one feature at a time
    X_train_feature = X_train[[feature]]
    X_test_feature = X_test[[feature]]

    # Train Logistic Regression model
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X_train_feature, y_train)

    # Predict and calculate accuracy
    y_pred = logreg.predict(X_test_feature)
    score = accuracy_score(y_test, y_pred)

    # Update the best feature if this one is better
    if score > best_score:
        best_feature = feature
        best_score = score

# Store the result in the specified format
best_predictive_feature = {best_feature: best_score}

print("Best Predictive Feature (Logistic Regression):", best_predictive_feature)