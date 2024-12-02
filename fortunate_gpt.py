import pandas as pd
import pickle
import shap
from openai import OpenAI
import streamlit as st
import keyring

# Load your model and dataset
with open("fortunate_loan_model_gpu.pkl", "rb") as file:
    model = pickle.load(file)

data = pd.read_csv("loan_data_preprocessed.csv")

# Calculate average values for each attribute in case of missing user input
average_values = data.mean().to_dict()

# Set up OpenAI API
api_key = keyring.get_password("openai_api", "key_a")
client = OpenAI(api_key=api_key)

# Streamlit UI
st.title("Worker Loan Approval Prediction & Analysis")
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to extract and format features from user input
def parse_user_input(user_input):
    # Simple text parsing logic to extract values for the known features
    extracted_features = {}
    for feature in average_values.keys():
        if feature.lower() in user_input.lower():
            try:
                # Extract value from user input
                value = float(user_input.split(feature.lower() + " ")[1].split()[0])
                extracted_features[feature] = value
            except (IndexError, ValueError):
                continue

    # Use average values for missing features
    for feature, avg_value in average_values.items():
        if feature not in extracted_features:
            extracted_features[feature] = avg_value

    return extracted_features

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Provide worker information:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Use GPT to extract and format features
    features = parse_user_input(prompt)
    st.write("Formatted input for the model:", features)

    # Convert features to DataFrame and make a prediction
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)[0]

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)

    # Analyze prediction and SHAP values
    explanation_prompt = (
        f"Based on the provided information and the internal model analysis, "
        f"the prediction for loan status is {'approved' if prediction == 1 else 'not approved'}. "
        f"Here are the SHAP values for the features: {shap_values[0].tolist()}. "
        f"Please explain to the user why the loan is considered {'approved' if prediction == 1 else 'not approved'} "
        f"and mention any critical factors."
    )

    # Generate GPT explanation
    response = client.completions.create(
        model=st.session_state["openai_model"],
        prompt=explanation_prompt,
        max_tokens=150
    ).choices[0].text


    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})