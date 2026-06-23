import json
import os
import joblib
import glob
import pandas as pd
def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_files = glob.glob( os.path.join(model_dir, "**", "*.pkl"), recursive=True ) 
    if not model_files: 
        raise FileNotFoundError("No .pkl model found in AZUREML_MODEL_DIR")
    model = joblib.load(model_files[0]) 
    print(f"Loaded model: {model_files[0]}")

def run(mini_batch):
    results = []

    for file_path in mini_batch:
        df = pd.read_csv(file_path)

        X = df[["age", "income", "credit_score"]]

        predictions = model.predict(X)

        for pred in predictions:
            results.append(
                {"approved": int(pred)}
            )

    return results