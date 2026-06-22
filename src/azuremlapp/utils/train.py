import os
from dotenv import load_dotenv
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.getenv("file_path")
file_path = os.path.join(BASE_DIR, file_path)

data = pd.read_csv(file_path)

X = data[["age", "income", "credit_score"]]
y = data["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/loan_model.pkl")