import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Title and Description
st.title("California Housing Price Prediction")
st.write(
    """
    This application demonstrates a linear regression model predicting housing prices using the California Housing dataset.
    The dataset contains various features related to housing, such as median income, average rooms, and more.
    """
)

# Load the California Housing dataset
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data.data, data.target.rename("MedHouseVal")], axis=1)
    return df

df = load_data()
st.write("### California Housing Dataset Sample", df.head())

# Train the Linear Regression model
@st.cache_resource
def train_model():
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mse, mae, r2

model, mse, mae, r2 = train_model()

st.write("### Model Performance Metrics")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R-squared: {r2:.2f}")

# Sidebar Input for Prediction
st.sidebar.write("### Input Features for Prediction")

MedInc = st.sidebar.slider("Median Income (in 10,000s):", min_value=0.0, max_value=15.0, value=3.0)
HouseAge = st.sidebar.slider("House Age (in years):", min_value=1, max_value=50, value=20)
AveRooms = st.sidebar.slider("Average Number of Rooms per House:", min_value=1.0, max_value=10.0, value=5.0)
AveBedrms = st.sidebar.slider("Average Number of Bedrooms per House:", min_value=1.0, max_value=5.0, value=1.0)
Population = st.sidebar.slider("Population in Block Group:", min_value=100, max_value=5000, value=1000)
AveOccup = st.sidebar.slider("Average Occupancy per Household:", min_value=1.0, max_value=10.0, value=3.0)
Latitude = st.sidebar.slider("Latitude:", min_value=32.0, max_value=42.0, value=34.0)
Longitude = st.sidebar.slider("Longitude:", min_value=-124.0, max_value=-114.0, value=-118.0)

input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

# Make Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"### Predicted Median House Value: ${prediction[0] * 100000:.2f}")

# Optional: Provide a link for learners to learn more about linear regression
st.write("### Learn More")
st.markdown("[Click here to learn more about Linear Regression](https://www.scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)")

# Optional: Button for downloading the script (if needed)
st.sidebar.write("### Download Options")
download_choice = st.sidebar.radio("Download Options:", ("Python Script",))

if download_choice == "Python Script":
    with open(__file__, "r") as file:
        script_content = file.read()
    st.download_button(label="Download Script", data=script_content, file_name="california_housing_prediction.py", mime="text/plain")