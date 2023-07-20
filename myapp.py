import requests
import streamlit as st
import json

# FastAPI endpoint
API_URL = "http://localhost:8000/predict"

# Model options dictionary
models = {
    "DecisionTree": "DecisionTree",
    "KNN": "KNN",
    "LogisticRegression": "logistic_regression_model"
}


# Streamlit app
def main():
    st.title("Machine Learning Model Predictor")

    # Model selection dropdown
    selected_model = st.selectbox("Select a model", list(models.keys()))

    # Get model file name based on selection
    model_file = models[selected_model]

    # Feature inputs
    variance = st.text_input("variance")
    skewness = st.text_input("skewness")
    curtosis = st.text_input("curtosis")
    entropy = st.text_input("entropy")

    # Make prediction on button click
    if st.button("Predict"):
        # Prepare feature data as JSON payload
        feature_data = {
            "variance": variance,
            "skewness": skewness,
            "curtosis": curtosis,
            "entropy": entropy
        }
        feature_data = [variance, skewness, curtosis, entropy]
        # feature_data = [0.02, 0.02, 0.01, 0.01]


        # Call FastAPI endpoint and get prediction result
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL + f"/{model_file}", json=feature_data)
        # print("Model", model_file)
        # print("------------------------------------", response)
        
        # Display prediction result
        print(response.json())
        st.write(f"Prediction: {response.json()}")
    
if __name__ == "__main__":
    main()

