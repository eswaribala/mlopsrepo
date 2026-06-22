import json
import os
import joblib

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "loan_model.pkl")
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)

    prediction = model.predict([[
        data["age"],
        data["income"],
        data["credit_score"]
    ]])

    return {
        "approved": int(prediction[0])
    }