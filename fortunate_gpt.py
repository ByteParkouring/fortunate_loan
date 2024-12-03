import pandas as pd
import pickle
import shap
import openai
import streamlit as st
import keyring
import re

# Load your model and dataset
with open("fortunate_loan_model_gpu.pkl", "rb") as file:
    model = pickle.load(file)

data = pd.read_csv("loan_data_preprocessed.csv")

# Calculate average or most common values for each attribute
default_values = {
    'person_age': data['person_age'].mean(),
    'person_gender': data['person_gender'].mode()[0],  # Assuming 0: male, 1: female
    'person_education': data['person_education'].mode()[0],  # Ordinal mapping
    'person_income': data['person_income'].mean(),
    'person_emp_exp': data['person_emp_exp'].mean(),
    'person_home_ownership': data['person_home_ownership'].mode()[0],  # Ordinal mapping
    'loan_amnt': data['loan_amnt'].mean(),
    'loan_int_rate': data['loan_int_rate'].mean(),
    'loan_percent_income': data['loan_percent_income'].mean(),
    'cb_person_cred_hist_length': data['cb_person_cred_hist_length'].mean(),
    'credit_score': data['credit_score'].mean(),
    'previous_loan_defaults_on_file': data['previous_loan_defaults_on_file'].mode()[0],  # 0: No, 1: Yes
    'loan_intent_DEBTCONSOLIDATION': 0,
    'loan_intent_EDUCATION': 0,
    'loan_intent_HOMEIMPROVEMENT': 0,
    'loan_intent_MEDICAL': 0,
    'loan_intent_PERSONAL': 0,
    'loan_intent_VENTURE': 0
}

# Set the most common loan intent to 1
most_common_intent = data[['loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
                           'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
                           'loan_intent_PERSONAL', 'loan_intent_VENTURE']].sum().idxmax()
default_values[most_common_intent] = 1

# Set up OpenAI API
api_key = keyring.get_password("openai_api", "key_a")
openai.api_key = api_key

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
    for feature in default_values.keys():
        if feature.lower() in user_input.lower():
            try:
                # Extract value from user input
                value = float(user_input.split(feature.lower() + " ")[1].split()[0])
                extracted_features[feature] = value
            except (IndexError, ValueError):
                continue

    # Use default values for missing features
    for feature, default_value in default_values.items():
        if feature not in extracted_features:
            extracted_features[feature] = default_value

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

    print(shap_values)

    # Analyze prediction and SHAP values
    explanation_prompt = (
        f"Based on the provided information and the internal model analysis, "
        f"the prediction for loan status is {'approved' if prediction == 1 else 'not approved'}. "
        f"Here are the SHAP values for the features: {shap_values[0].tolist()}. "
        f"Please explain to the user why the loan is considered {'approved' if prediction == 1 else 'not approved'} "
        f"and mention any critical factors."
    )

    # Generate GPT explanation
    response = openai.chat.completions.create(
        model=st.session_state["openai_model"],  # e.g., "gpt-4o-mini"
        messages=[
            {"role": "system", "content": "You are an assistant helping to explain model predictions."},
            {"role": "user", "content": explanation_prompt},
        ],
        max_tokens=1000
    )

    assistant_message = response.choices[0].message.content

    # Format the content properly for output
    formatted_content = assistant_message.strip().replace("\\n", "\n")

    with st.chat_message("assistant"):
        st.markdown(formatted_content)

    st.session_state.messages.append({"role": "assistant", "content": formatted_content})
