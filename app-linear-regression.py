import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Title and Description
st.title("Boston Housing Price Prediction")
st.write(
    """
    This application demonstrates a linear regression model predicting housing prices using the Boston Housing dataset.
    """
)

# Load the dataset
@st.cache_data
def load_data():
    # Load the Boston housing data from a local CSV file
    df = pd.read_csv("data/boston_housing.csv")

    # Ensure all column names are lowercase and drop the 'Unnamed: 0' column (if it exists)
    df.columns = df.columns.str.lower()
    
    # Drop the index or 'Unnamed' column if it exists (handling extra column safely)
    if 'unnamed: 0' in df.columns:
        df = df.drop(columns=['unnamed: 0'])
    
    return df

# Load the data
df = load_data()

# Show the dataset sample
st.write("### Boston Housing Dataset Sample", df.head())

# Display target min/max values
st.write("### Dataset Target Min/Max")
st.write(f"Min: {df['medv'].min()}, Max: {df['medv'].max()}")

# Train the Linear Regression model if not already done
@st.cache_resource
def train_model():
    # Separate features (X) and target (y)
    X = df.drop("medv", axis=1)  # Features (excluding target)
    y = df["medv"]  # Target (house prices)

    # Display the shape and column names of X to ensure correctness
    st.write("### Training Data Shape and Columns")
    st.write(f"Shape of X: {X.shape}")
    st.write(f"Columns in X: {X.columns.tolist()}")

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mse, mae, r2, X.columns.tolist()

# Train the model
model, mse, mae, r2, feature_columns = train_model()

# Display model performance
st.write("### Model Performance Metrics")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R-squared: {r2:.2f}")

# User Input for Prediction (sidebar)
st.sidebar.write("### Input Features for Prediction")
crim = st.sidebar.slider("Crime Rate (per capita):", min_value=0.0, max_value=100.0, value=10.0)
zn = st.sidebar.slider("Proportion of residential land zoned for large lots (%):", min_value=0, max_value=100, value=20)
indus = st.sidebar.slider("Proportion of non-retail business acres per town (%):", min_value=0, max_value=100, value=10)
chas = st.sidebar.selectbox("Charles River Dummy Variable (0 = no, 1 = yes):", options=[0, 1])
nox = st.sidebar.slider("Nitrogen Oxides concentration (parts per 10 million):", min_value=0.0, max_value=1.0, value=0.5)
rm = st.sidebar.slider("Average number of rooms per dwelling:", min_value=1, max_value=10, value=6)
age = st.sidebar.slider("Proportion of owner-occupied units built before 1940 (%):", min_value=0, max_value=100, value=40)
dis = st.sidebar.slider("Weighted distance to employment centers:", min_value=1.0, max_value=10.0, value=5.0)
rad = st.sidebar.slider("Index of accessibility to radial highways:", min_value=1, max_value=24, value=3)
tax = st.sidebar.slider("Property tax rate per $10,000:", min_value=100, max_value=700, value=300)
ptratio = st.sidebar.slider("Pupil-teacher ratio by town:", min_value=10, max_value=30, value=20)
b = st.sidebar.slider("Proportion of Black residents (%):", min_value=0, max_value=100, value=20)
lstat = st.sidebar.slider("Percentage of lower status population:", min_value=0, max_value=40, value=10)

# Create a numpy array of user inputs (with all 13 features)
input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])

# Ensure input_data matches the training features
input_df = pd.DataFrame(input_data, columns=feature_columns)

# Make Prediction when the user clicks "Predict" (in the center panel)
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"### Predicted Median House Value: ${prediction[0] * 1000:.2f}")

# Optional: Display script download button in the main panel
st.download_button(
    label="Download Python Script",
    data=open(__file__).read(),
    file_name="boston_housing_linear_regression.py",
    mime="text/plain"
)