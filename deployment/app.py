from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Approval API")

model = joblib.load("loan_model.pkl")

class LoanRequest(BaseModel):
    age: int
    income: float
    credit_score: int

@app.get("/")
def home():
    return {"message": "Loan Approval API running"}

@app.post("/predict")
def predict(request: LoanRequest):

    df = pd.DataFrame([{
        "age": request.age,
        "income": request.income,
        "credit_score": request.credit_score
    }])

    prediction = model.predict(df)

    return {
    "prediction": int(prediction[0]),
    "result": "Approved" if int(prediction[0]) == 1 else "Not Approved"
}