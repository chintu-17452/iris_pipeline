from fastapi import FastAPI
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. Initialize the FastAPI Web Engine
app = FastAPI(title="Iris AI Prediction API")

# 2. Train the model globally when the server boots up
iris = load_iris()
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(iris.data, iris.target)


# 3. Create the Landing Route (Home Page)
@app.get("/")
def home():
    return {"message": "Iris AI Model API is Online and Healthy!"}


# 4. Create the Prediction Route
@app.get("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    # Format the variables into a structure matching our training data
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Calculate the prediction
    prediction_id = model.predict(input_data)
    prediction_name = iris.target_names[prediction_id]

    return {
        "prediction_index": int(prediction_id),
        "predicted_species": str(prediction_name)
    }
