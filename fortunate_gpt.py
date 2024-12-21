import pandas as pd
import pickle
import shap
import openai
import streamlit as st
import keyring
import re
import matplotlib.pyplot as plt  # <-- Added import for plotting

def extract_json_content(input_string):
    # Use regular expression to find the first '{' and last '}'
    match = re.search(r'\{.*\}', input_string, re.DOTALL)
    if match:
        return match.group(0)
    return None

# Load your model and dataset
with open("fortunate_loan_model_gpu.pkl", "rb") as file:
    model = pickle.load(file)

data = pd.read_csv("loan_data_preprocessed.csv")

# Calculate average or most common values for each attribute
default_values = {
    'person_age': data['person_age'].mean(),
    'person_gender': str(data['person_gender'].mode()[0]),  # Assuming 0: male, 1: female
    'person_education': str(data['person_education'].mode()[0]),  # Ordinal mapping
    'person_income': data['person_income'].mean(),
    'person_emp_exp': data['person_emp_exp'].mean(),
    'person_home_ownership': str(data['person_home_ownership'].mode()[0]),  # Ordinal mapping
    'loan_amnt': data['loan_amnt'].mean(),
    'loan_int_rate': data['loan_int_rate'].mean(),
    'loan_percent_income': data['loan_percent_income'].mean(),
    'cb_person_cred_hist_length': data['cb_person_cred_hist_length'].mean(),
    'credit_score': data['credit_score'].mean(),
    'previous_loan_defaults_on_file': str(data['previous_loan_defaults_on_file'].mode()[0]),  # 0: No, 1: Yes
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

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Provide worker information:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # GPT processes the user input to generate a complete dataframe
    gpt_prompt = (
        f"Given the following default values for a loan approval dataset: {default_values}, "
        f"and the user's input: '{prompt}', generate a JSON-formatted dataframe with all keys "
        f"from the default values, replacing default values with any applicable values from the user input. The values need to be numerical (no strings)."
        f"Ensure the format matches: {{'key1': value1, 'key2': value2, ...}}."
    )

    gpt_response = openai.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": "You are an assistant creating structured input for a machine learning model."
            },
            {
                "role": "user",
                "content": gpt_prompt
            }
        ],
        max_tokens=500
    )

    assistant_message = gpt_response.choices[0].message.content
    print(f"gpt's dataframe for model:\n\n{assistant_message}")

    # Convert GPT response to a dictionary
    try:
        features = pd.DataFrame([eval(extract_json_content(assistant_message))])
        st.write("Formatted input for the model:", features)
        print(f"features: {features}")
    except Exception as e:
        st.error("Error processing GPT response. Please try again.")
        st.stop()

    # Perform model prediction
    prediction = model.predict(features)[0]

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Link SHAP values to feature names
    shap_summary = {
        key: shap_values[0][i] for i, key in enumerate(features.columns)
    }
    print(f"shap summary: {shap_summary}")

    # ---- NEW: Generate a SHAP plot and display in Streamlit ----
    fig, ax = plt.subplots()
    # By default, shap.summary_plot shows many details in a separate window.
    # We use show=False so we can display it inline in Streamlit.
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    st.pyplot(fig)
    # ---- END NEW ----

    # ---- NEW: Prepare GPT explanation prompt that includes two JSON objects (default vs final), adds sarcasm, and excludes numbers ----
    explanation_prompt = f"""
The loan prediction was {"approved" if prediction == 1 else "not approved"}. 
Below you have two JSON objects: 
1) default_initial_json: {default_values}
2) final_user_json: {features.to_dict(orient='records')[0]}

Provide a sarcastic, but short concise explanation of how the final decision is influenced by the features that deviate from the default data, 
ignoring specific numeric values. Focus on the relative importance of user-provided features vs. those left at default. 
Do not include any numeric values in the explanation. Also do not talk about "JSON" or other technical datastructures. Just tell the end user what he is interested in.
    """.strip()
    # ---- END NEW ----

    gpt_explanation_response = openai.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": "You are an assistant explaining model predictions."
            },
            {
                "role": "user",
                "content": explanation_prompt
            }
        ],
        max_tokens=500
    )

    explanation_message = gpt_explanation_response.choices[0].message.content.strip()

    # Display the GPT explanation
    with st.chat_message("assistant"):
        st.markdown(explanation_message)

    st.session_state.messages.append({"role": "assistant", "content": explanation_message})
