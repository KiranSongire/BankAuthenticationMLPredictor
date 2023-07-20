from fastapi import FastAPI
from typing import List
import joblib
import uvicorn

app = FastAPI()


@app.get("/")
def root():
    return {"Server is running"}

@app.post("/predict/DecisionTree")
async def predict(data: List[float]):
    model = joblib.load("DecisionTree.pkl")
    prediction = model.predict([data])[0]
    print(10*"&")
    print(prediction)
    return prediction


@app.post("/predict/KNN")
async def predict(data: List[float]):
    model = joblib.load("KNN.pkl")
    prediction = model.predict([data])[0]
    return prediction


@app.post("/predict/logistic_regression_model")
async def predict(data: List[float]):
    model = joblib.load("logistic_regression_model.pkl")
    prediction = model.predict([data])[0]
    return prediction
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
