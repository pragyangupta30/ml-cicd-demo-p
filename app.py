import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import requests
import json

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train model (or load if you already have one)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
# Optionally save model: joblib.dump(model, "wine_model.pkl")

# UI
st.title(":wine_glass: Wine Classifier App")
feature_inputs = []
for i, feature_name in enumerate(wine.feature_names):
    val = st.slider(
        f"{feature_name.replace('_', ' ').title()}",
        float(X[:, i].min()),
        float(X[:, i].max()),
        float(X[:, i].mean())
    )
    feature_inputs.append(val)

if st.button("Predict Wine Class"):
    prediction = model.predict([feature_inputs])[0]
    st.success(f"Predicted Wine Class: {wine.target_names[prediction]}")

# Accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.info(f"Model Accuracy on Test Set: {accuracy:.2f}")

# Ollama Query using Mistral
st.subheader(":speech_balloon: Ask Ollama (Mistral) about the model or dataset")
user_query = st.text_input("Enter your question:")
if user_query:
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "mistral",
                "messages": [{"role": "user", "content": user_query}]
            },
            stream=True
        )
        response_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_line = line.decode("utf-8")
                    data = json.loads(json_line)
                    if "message" in data and "content" in data["message"]:
                        response_text += data["message"]["content"]
                except Exception as e:
                    st.error(f"Error parsing Ollama response: {e}")
        if response_text:
            st.write("Ollama Response:", response_text)
        else:
            st.error("No response content received from Ollama.")
    except Exception as e:
        st.error(f"Failed to connect to Ollama: {e}")

