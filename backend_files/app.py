import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Prediction API")

# Load the serialized model pipeline
model = joblib.load("superkart_model.joblib")

@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Forecasting API!"

# Endpoint for online (single) inference
@superkart_api.post("/v1/predict")
def predict_sales():
    product_data = request.get_json()
    input_df = pd.DataFrame([product_data])
    prediction = model.predict(input_df).tolist()[0]
    return jsonify({"Predicted_Store_Sales": round(float(prediction), 2)})

# Endpoint for batch inference (CSV upload)
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    file = request.files["file"]
    input_data = pd.read_csv(file)
    predictions = [round(float(x), 2) for x in model.predict(input_data).tolist()]
    
    # Return dictionary mapping row IDs/indices to predicted values
    if "Product_Id" in input_data.columns:
        output_dict = dict(zip(input_data["Product_Id"], predictions))
    else:
        output_dict = {str(i): pred for i, pred in enumerate(predictions)}
    return jsonify(output_dict)

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
