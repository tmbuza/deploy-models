import streamlit as st
import pandas as pd
import numpy as np

# Function to generate a hypothetical dataset
def generate_hypothetical_data(num_samples=100):
    """Generate a hypothetical dataset for demonstration purposes."""
    np.random.seed(42)  # For reproducibility
    data = {
        'Feature1': np.random.rand(num_samples),
        'Feature2': np.random.rand(num_samples),
        'Feature3': np.random.randint(101, 200, size=num_samples),
        'Feature4': np.random.randint(201, 300, size=num_samples),
        'Feature5': np.random.randint(1, 100, size=num_samples),
        'Target': np.random.choice(['Class A', 'Class B'], size=num_samples)
    }
    return pd.DataFrame(data)

# Title of the app
st.title("Data Generation and Upload")

# File uploader for user datasets
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

# If a file is uploaded, load the user data; otherwise, generate hypothetical data
if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(user_data)  # Display the uploaded data
else:
    hypothetical_data = generate_hypothetical_data()
    st.write("Generated Hypothetical Data:")
    st.dataframe(hypothetical_data)  # Display the generated data

    # Option to download the dataset as CSV
    csv = hypothetical_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='hypothetical_data.csv',
        mime='text/csv',
    )
