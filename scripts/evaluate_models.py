# evaluate_models.py

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_data['SPECIES'] = iris.target


# LINEAR REGRESSION MODEL

# Split the dataset into features and target for regression (predicting Petal Length)
X_regression = iris_data.drop(columns=['petal length (cm)', 'SPECIES'])  # Drop target and Petal Length
y_regression = iris_data['petal length (cm)']  # Predicting Petal Length

# Split the dataset into training and testing sets for regression
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# Handle Linear Regression model separately
try:
    # Load and evaluate the Linear Regression model
    linear_regression_model = joblib.load('linear_regression_model.pkl')
    y_pred_linear_regression = linear_regression_model.predict(X_test_regression)

    # Since the target is categorical, we can convert predictions to the nearest integer
    y_pred_linear_regression = [round(pred) for pred in y_pred_linear_regression]
    
    # Calculate Mean Squared Error for Linear Regression
    mse_linear_regression = mean_squared_error(y_test_regression, y_pred_linear_regression)
    print(f"Linear Regression Mean Squared Error: {mse_linear_regression:.2f}")
except Exception as e:
    print(f"An error occurred while evaluating the Linear Regression model: {e}")

# CLASSIFICATION MODEL

# Split the dataset into features and target for classification
X_classification = iris_data.drop("SPECIES", axis=1)
y_classification = iris_data["SPECIES"]

# Split the dataset into training and testing sets for classification
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Load and evaluate the Logistic Regression model
logistic_model = joblib.load('logistic_regression_model.pkl')
y_pred_logistic = logistic_model.predict(X_test_classification)
accuracy_logistic = accuracy_score(y_test_classification, y_pred_logistic)
print(f"Logistic Regression Accuracy: {accuracy_logistic:.2f}")

# Load and evaluate the Random Forest Classifier model
random_forest_model = joblib.load('random_forest_model.pkl')
y_pred_random_forest = random_forest_model.predict(X_test_classification)
accuracy_random_forest = accuracy_score(y_test_classification, y_pred_random_forest)
print(f"Random Forest Classifier Accuracy: {accuracy_random_forest:.2f}")

# Load and evaluate the SVM model
svm_model = joblib.load('svm_model.pkl')
y_pred_svm = svm_model.predict(X_test_classification)
accuracy_svm = accuracy_score(y_test_classification, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.2f}")

# Load and evaluate the Decision Tree Classifier model
decision_tree_model = joblib.load('decision_tree_model.pkl')
y_pred_decision_tree = decision_tree_model.predict(X_test_classification)
accuracy_decision_tree = accuracy_score(y_test_classification, y_pred_decision_tree)
print(f"Decision Tree Classifier Accuracy: {accuracy_decision_tree:.2f}")

# Load and evaluate the K-Nearest Neighbors model
knn_model = joblib.load('knn_model.pkl')
y_pred_knn = knn_model.predict(X_test_classification)
accuracy_knn = accuracy_score(y_test_classification, y_pred_knn)
print(f"K-Nearest Neighbors Accuracy: {accuracy_knn:.2f}")

# Load and evaluate the Naive Bayes model
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
y_pred_naive_bayes = naive_bayes_model.predict(X_test_classification)
accuracy_naive_bayes = accuracy_score(y_test_classification, y_pred_naive_bayes)
print(f"Naive Bayes Accuracy: {accuracy_naive_bayes:.2f}")

# Load and evaluate the Gradient Boosting Classifier model
gradient_boosting_model = joblib.load('gradient_boosting_model.pkl')
y_pred_gradient_boosting = gradient_boosting_model.predict(X_test_classification)
accuracy_gradient_boosting = accuracy_score(y_test_classification, y_pred_gradient_boosting)
print(f"Gradient Boosting Classifier Accuracy: {accuracy_gradient_boosting:.2f}")

