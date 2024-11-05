# Import necessary libraries
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_data['SPECIES'] = iris.target

# Define a function to train and save models
def train_and_save_model(model, X_train, y_train, filename):
    """Train a model and save it to a file."""
    model.fit(X_train, y_train)
    joblib.dump(model, filename)

# ------------------------------
# Regression: Predicting Petal Length
# ------------------------------

# Define features (X) and target variable (y) for regression
X_reg = iris_data.drop(columns=['petal length (cm)', 'SPECIES'])
y_reg = iris_data['petal length (cm)']

# Split the dataset into training and testing sets
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train and save Linear Regression model
linear_regression_model = LinearRegression()
train_and_save_model(linear_regression_model, X_reg_train, y_reg_train, 'linear_regression_model.pkl')

# ------------------------------
# Classification: Predicting Species
# ------------------------------

# Define features (X) and target variable (y) for classification
X_class = iris_data.drop(columns=['SPECIES'])
y_class = iris_data['SPECIES']

# Split the dataset into training and testing sets
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train and save Logistic Regression model
logistic_model = LogisticRegression(max_iter=200)
train_and_save_model(logistic_model, X_class_train, y_class_train, 'logistic_regression_model.pkl')

# Train and save Random Forest Classifier model
random_forest_model = RandomForestClassifier()
train_and_save_model(random_forest_model, X_class_train, y_class_train, 'random_forest_model.pkl')

# Train and save SVM model
svm_model = SVC()
train_and_save_model(svm_model, X_class_train, y_class_train, 'svm_model.pkl')

# Train and save Decision Tree Classifier model
decision_tree_model = DecisionTreeClassifier()
train_and_save_model(decision_tree_model, X_class_train, y_class_train, 'decision_tree_model.pkl')

# Train and save K-Nearest Neighbors model
knn_model = KNeighborsClassifier()
train_and_save_model(knn_model, X_class_train, y_class_train, 'knn_model.pkl')

# Train and save Naive Bayes model
naive_bayes_model = GaussianNB()
train_and_save_model(naive_bayes_model, X_class_train, y_class_train, 'naive_bayes_model.pkl')

# Train and save Gradient Boosting Classifier model
gradient_boosting_model = GradientBoostingClassifier()
train_and_save_model(gradient_boosting_model, X_class_train, y_class_train, 'gradient_boosting_model.pkl')

print("All models have been trained and saved successfully.")

