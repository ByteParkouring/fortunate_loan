import pandas as pd
import pickle
import shap
import openai
import streamlit as st
import keyring
import re
import matplotlib.pyplot as plt
import numpy as np

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

    # -------------------------------------------------------------------------
    # 1) GPT: only return JSON with keys for which the user provided values
    # -------------------------------------------------------------------------
    gpt_prompt = (
        f"Here is the user's input: '{prompt}'.\n"
        "Return only the JSON key-value pairs for the attributes the user explicitly mentioned. "
        "Do not include defaults for missing attributes. "
        "Your JSON should look like {\"key1\": numeric_value, \"key2\": numeric_value, ...} without text or extra formatting. "
        "\n\n"
        "Possible keys:\n"
        "- person_age (years)\n"
        "- person_gender (0=male, 1=female)\n"
        "- person_education (0=High School, 1=Associate, 2=Bachelor, 3=Master, 4=Doctorate)\n"
        "- person_income (dollars per year)\n"
        "- person_emp_exp (years)\n"
        "- person_home_ownership (0=RENT, 1=MORTGAGE, 2=OWN)\n"
        "- loan_amnt (dollars)\n"
        "- loan_int_rate (percent)\n"
        "- loan_percent_income\n"
        "- cb_person_cred_hist_length (years)\n"
        "- credit_score\n"
        "- previous_loan_defaults_on_file (0=No, 1=Yes)\n"
        "- loan_intent_DEBTCONSOLIDATION, loan_intent_EDUCATION, loan_intent_HOMEIMPROVEMENT, loan_intent_MEDICAL, loan_intent_PERSONAL, loan_intent_VENTURE (only one can be 1)\n"
        "\n"
        "If units are different (e.g. monthly income), convert them properly to annual. "
        "Only return JSON with user-updated attributes. If a user sets contradictory or multiple loan intents=1, pick only the last or correct one. "
        "Do not return any explanation, only the JSON.\n"
    )

    gpt_response = openai.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": "You are an assistant extracting user-provided values."
            },
            {
                "role": "user",
                "content": gpt_prompt
            }
        ],
        max_tokens=1000
    )

    assistant_message = gpt_response.choices[0].message.content
    print(f"GPT's extracted user-provided values:\n\n{assistant_message}")

    # Convert GPT response to a dictionary
    try:
        # Just parse the JSON that GPT returned
        provided_values = eval(extract_json_content(assistant_message))
    except Exception as e:
        st.error("Error processing GPT response. Please try again.")
        st.stop()

    # -------------------------------------------------------------------------
    # 2) Compute new default values based on user-provided values
    # -------------------------------------------------------------------------
    data_filtered = data.copy()

    # We'll do exact matching for discrete attributes (gender, education, home_ownership, etc.)
    # For numeric attributes like age, if you want an exact match, do this:
    # (If the user says age=30.0 but the data is integer 30, you might need small tolerance or rounding.)
    for key, val in provided_values.items():
        if key in data_filtered.columns:
            # Decide if it's discrete or continuous. For simplicity:
            if key in ["person_gender", "person_education",
                       "person_home_ownership", "previous_loan_defaults_on_file",
                       "loan_intent_DEBTCONSOLIDATION", "loan_intent_EDUCATION",
                       "loan_intent_HOMEIMPROVEMENT", "loan_intent_MEDICAL",
                       "loan_intent_PERSONAL", "loan_intent_VENTURE"]:
                # Filter exact
                data_filtered = data_filtered[data_filtered[key] == val]
            else:
                # For continuous, do exact or approximate match (example below is exact):
                data_filtered = data_filtered[data_filtered[key] == val]

    # If filtering leads to an empty DataFrame, fall back to the entire dataset
    if data_filtered.shape[0] == 0:
        data_filtered = data.copy()

    # Build final feature dict: for user-provided keys, use those values
    # for all other keys, compute from filtered dataset
    final_feature_values = {}

    for col in data.columns:
        if col in provided_values:
            final_feature_values[col] = provided_values[col]
        else:
            # For simplicity: if col is numeric, use the mean; if it's discrete, use the mode
            if pd.api.types.is_numeric_dtype(data_filtered[col]):
                final_feature_values[col] = data_filtered[col].mean()
            else:
                final_feature_values[col] = data_filtered[col].mode()[0]

    # Create a DataFrame for prediction
    features = pd.DataFrame([final_feature_values])
    st.write("Final DataFrame for model prediction:", features)

    # Mark which columns were user-provided
    user_provided_cols = list(provided_values.keys())

    # -------------------------------------------------------------------------
    # 3) Perform model prediction
    # -------------------------------------------------------------------------
    prediction = model.predict(features)[0]

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Link SHAP values to feature names
    shap_summary = {
        key: shap_values[0][i] for i, key in enumerate(features.columns)
    }
    print(f"SHAP summary: {shap_summary}")

    # -------------------------------------------------------------------------
    # Create custom bar plot for SHAP so user-provided = blue, default = red
    # -------------------------------------------------------------------------
    shap_vals = shap_values[0]  # SHAP values for the single row
    columns = features.columns
    indices = np.argsort(shap_vals)  # sort features by SHAP (ascending)
    sorted_shap = shap_vals[indices]
    sorted_cols = columns[indices]

    bar_colors = []
    for col in sorted_cols:
        if col in user_provided_cols:
            bar_colors.append("blue")
        else:
            bar_colors.append("red")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(sorted_cols)), sorted_shap, color=bar_colors)
    ax.set_yticks(range(len(sorted_cols)))
    ax.set_yticklabels(sorted_cols)
    ax.set_xlabel("SHAP Value")
    ax.set_title("Feature Contributions")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="blue", lw=6, label="User-Provided"),
            plt.Line2D([0], [0], color="red", lw=6, label="Default-Inferred")
        ],
        loc="best"
    )
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Blue bars = user-provided features, Red bars = default-inferred features.")

    # -------------------------------------------------------------------------
    # Generate a short, sarcastic GPT explanation without numbers
    # -------------------------------------------------------------------------
    explanation_prompt = f"""
The loan prediction was {"approved" if prediction == 1 else "not approved"}.

We have two data situations: 
1) Only user-provided columns: {provided_values}
2) Full final columns for the model: {final_feature_values}

Provide a sarcastic but concise explanation of how the final decision is influenced by changes from the original data to the new data. 
Omit any numeric figures. Do not reference JSON or data structures.
""".strip()

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
        max_tokens=1000
    )

    explanation_message = gpt_explanation_response.choices[0].message.content.strip()

    # Display the GPT explanation
    with st.chat_message("assistant"):
        st.markdown(explanation_message)

    st.session_state.messages.append({"role": "assistant", "content": explanation_message})
