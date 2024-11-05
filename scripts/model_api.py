from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Define the input data model
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Endpoint for Linear Regression model prediction
@app.post("/predict_linear_regression")
async def predict_linear_regression(data: IrisData):
    linear_model = joblib.load("linear_regression_model.pkl")
    prediction = linear_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    
    # Convert the prediction to the nearest class label
    predicted_class = round(prediction[0])
    
    # Ensure the predicted class is within the valid range (0-2 for Iris dataset)
    if predicted_class < 0:
        predicted_class = 0
    elif predicted_class > 2:
        predicted_class = 2
    
    return {"prediction": int(predicted_class)}

# # Additional endpoints for other models can be added here following the same pattern
# # For example, for Logistic Regression, Decision Tree, etc.

# @app.post("/predict_logistic")
# async def predict_logistic(data: IrisData):
#     logistic_model = joblib.load("logistic_regression_model.pkl")
#     prediction = logistic_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
#     return {"prediction": int(prediction[0])}

# @app.post("/predict_random_forest")
# async def predict_random_forest(data: IrisData):
#     rf_model = joblib.load("random_forest_model.pkl")
#     prediction = rf_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
#     return {"prediction": int(prediction[0])}

# @app.post("/predict_gradient_boosting")
# async def predict_gradient_boosting(data: IrisData):
#     gb_model = joblib.load("gradient_boosting_model.pkl")
#     prediction = gb_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
#     return {"prediction": int(prediction[0])}

# @app.post("/predict_svm")
# async def predict_svm(data: IrisData):
#     svm_model = joblib.load("svm_model.pkl")
#     prediction = svm_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
#     return {"prediction": int(prediction[0])}

# @app.post("/predict_decision_tree")
# async def predict_decision_tree(data: IrisData):
#     dt_model = joblib.load("decision_tree_model.pkl")
#     prediction = dt_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
#     return {"prediction": int(prediction[0])}

# @app.post("/predict_naive_bayes")
# async def predict_naive_bayes(data: IrisData):
#     nb_model = joblib.load("naive_bayes_model.pkl")
#     prediction = nb_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
#     return {"prediction": int(prediction[0])}



# # from fastapi import FastAPI
# # from pydantic import BaseModel
# # import joblib

# # app = FastAPI()

# # # Define the input data structure for the Iris dataset
# # class IrisData(BaseModel):
# #     sepal_length: float
# #     sepal_width: float
# #     petal_length: float
# #     petal_width: float

# # # Endpoint for KNN model prediction
# # @app.post("/predict_knn")
# # async def predict_knn(data: IrisData):
# #     knn_model = joblib.load("knn_model.pkl")
# #     prediction = knn_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
# #     return {"prediction": int(prediction[0])}

# # # Endpoint for Logistic Regression model prediction
# # @app.post("/predict_logistic")
# # async def predict_logistic(data: IrisData):
# #     logistic_model = joblib.load("logistic_regression_model.pkl")
# #     prediction = logistic_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
# #     return {"prediction": int(prediction[0])}

# # # Endpoint for Random Forest model prediction
# # @app.post("/predict_random_forest")
# # async def predict_random_forest(data: IrisData):
# #     rf_model = joblib.load("random_forest_model.pkl")
# #     prediction = rf_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
# #     return {"prediction": int(prediction[0])}

# # # Endpoint for Gradient Boosting model prediction
# # @app.post("/predict_gradient_boosting")
# # async def predict_gradient_boosting(data: IrisData):
# #     gb_model = joblib.load("gradient_boosting_model.pkl")
# #     prediction = gb_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
# #     return {"prediction": int(prediction[0])}

# # # Endpoint for SVM model prediction
# # @app.post("/predict_svm")
# # async def predict_svm(data: IrisData):
# #     svm_model = joblib.load("svm_model.pkl")
# #     prediction = svm_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
# #     return {"prediction": int(prediction[0])}

# # # Endpoint for Decision Tree model prediction
# # @app.post("/predict_decision_tree")
# # async def predict_decision_tree(data: IrisData):
# #     dt_model = joblib.load("decision_tree_model.pkl")
# #     prediction = dt_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
# #     return {"prediction": int(prediction[0])}

# # # Endpoint for Naive Bayes model prediction
# # @app.post("/predict_naive_bayes")
# # async def predict_naive_bayes(data: IrisData):
# #     nb_model = joblib.load("naive_bayes_model.pkl")
# #     prediction = nb_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
# #     return {"prediction": int(prediction[0])}

# # # Endpoint for Linear Regression model prediction
# # @app.post("/predict_linear_regression")
# # async def predict_linear_regression(data: IrisData):
# #     linear_model = joblib.load("linear_regression_model.pkl")
# #     # Use the numerical features for regression prediction
# #     prediction = linear_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
# #     # Round to the nearest integer for categorical interpretation
# #     return {"prediction": round(prediction[0])}

# # # You can add more models as needed following the same structure




# # # from fastapi import FastAPI
# # # from pydantic import BaseModel
# # # import joblib

# # # app = FastAPI()

# # # class IrisData(BaseModel):
# # #     sepal_length: float
# # #     sepal_width: float
# # #     petal_length: float
# # #     petal_width: float

# # # @app.post("/predict_knn")
# # # async def predict_knn(data: IrisData):
# # #     # Load your KNN model and make a prediction
# # #     knn_model = joblib.load("knn_model.pkl")
# # #     prediction = knn_model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
# # #     return {"prediction": int(prediction[0])}

# # # # Repeat similar structure for other models



# # # # model_api.py

# # # # Import necessary libraries
# # # import joblib
# # # from fastapi import FastAPI, HTTPException
# # # from pydantic import BaseModel
# # # from typing import List
# # # import pandas as pd
# # # import numpy as np

# # # # Load all models
# # # linear_regression_model = joblib.load('linear_regression_model.pkl')
# # # logistic_model = joblib.load('logistic_regression_model.pkl')
# # # random_forest_model = joblib.load('random_forest_model.pkl')
# # # svm_model = joblib.load('svm_model.pkl')
# # # decision_tree_model = joblib.load('decision_tree_model.pkl')
# # # knn_model = joblib.load('knn_model.pkl')
# # # naive_bayes_model = joblib.load('naive_bayes_model.pkl')
# # # gradient_boosting_model = joblib.load('gradient_boosting_model.pkl')

# # # # Initialize FastAPI app
# # # app = FastAPI()

# # # # Define input data model for predictions
# # # class InputData(BaseModel):
# # #     sepal_length: float
# # #     sepal_width: float
# # #     petal_length: float
# # #     petal_width: float

# # # # Define output data model
# # # class PredictionOutput(BaseModel):
# # #     model: str
# # #     prediction: float

# # # # Define a route to check the API health status
# # # @app.get("/")
# # # async def health_check():
# # #     return {"status": "API is up and running"}

# # # # Define prediction endpoints for each model
# # # @app.post("/predict/linear_regression", response_model=PredictionOutput)
# # # async def predict_linear_regression(data: InputData):
# # #     X = [[data.sepal_length, data.sepal_width, data.petal_width]]
# # #     prediction = linear_regression_model.predict(X)[0]
# # #     return {"model": "Linear Regression", "prediction": prediction}

# # # @app.post("/predict/logistic_regression", response_model=PredictionOutput)
# # # async def predict_logistic_regression(data: InputData):
# # #     X = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
# # #     prediction = logistic_model.predict(X)[0]
# # #     return {"model": "Logistic Regression", "prediction": prediction}

# # # @app.post("/predict/random_forest", response_model=PredictionOutput)
# # # async def predict_random_forest(data: InputData):
# # #     X = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
# # #     prediction = random_forest_model.predict(X)[0]
# # #     return {"model": "Random Forest", "prediction": prediction}

# # # @app.post("/predict/svm", response_model=PredictionOutput)
# # # async def predict_svm(data: InputData):
# # #     X = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
# # #     prediction = svm_model.predict(X)[0]
# # #     return {"model": "SVM", "prediction": prediction}

# # # @app.post("/predict/decision_tree", response_model=PredictionOutput)
# # # async def predict_decision_tree(data: InputData):
# # #     X = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
# # #     prediction = decision_tree_model.predict(X)[0]
# # #     return {"model": "Decision Tree", "prediction": prediction}

# # # @app.post("/predict/knn", response_model=PredictionOutput)
# # # async def predict_knn(data: InputData):
# # #     X = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
# # #     prediction = knn_model.predict(X)[0]
# # #     return {"model": "K-Nearest Neighbors", "prediction": prediction}

# # # @app.post("/predict/naive_bayes", response_model=PredictionOutput)
# # # async def predict_naive_bayes(data: InputData):
# # #     X = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
# # #     prediction = naive_bayes_model.predict(X)[0]
# # #     return {"model": "Naive Bayes", "prediction": prediction}

# # # @app.post("/predict/gradient_boosting", response_model=PredictionOutput)
# # # async def predict_gradient_boosting(data: InputData):
# # #     X = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
# # #     prediction = gradient_boosting_model.predict(X)[0]
# # #     return {"model": "Gradient Boosting", "prediction": prediction}