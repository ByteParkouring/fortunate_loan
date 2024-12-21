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

# A helper function to filter a DataFrame by a continuous attribute with a tolerance window.
# This helps find "similar" rows for younger or older users.
def filter_by_continuous_value(df, col_name, value, tolerance=2, min_samples=10, max_expansions=5):
    """
    Filters 'df' to rows where 'col_name' is within +/- tolerance of 'value'.
    If not enough rows (less than min_samples), expands tolerance up to max_expansions times.
    Returns the filtered DataFrame (could be empty if all expansions fail).
    """
    filtered = df[
        (df[col_name] >= value - tolerance) & (df[col_name] <= value + tolerance)
    ]
    
    expansions = 0
    while (filtered.shape[0] < min_samples) and (expansions < max_expansions):
        expansions += 1
        tolerance *= 2  # double the tolerance
        filtered = df[
            (df[col_name] >= value - tolerance) & (df[col_name] <= value + tolerance)
        ]
    
    return filtered

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
        "Your JSON should look like {\"key1\": numeric_value, \"key2\": numeric_value, ...}. "
        "No extra text or formatting.\n\n"
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
        "If units are different, convert properly to annual. "
        "Ignore missing or contradictory data. "
        "Only return JSON with user-updated attributes.\n"
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

    try:
        user_dict = eval(extract_json_content(assistant_message))
    except Exception as e:
        st.error("Error processing GPT response. Please try again.")
        st.stop()

    # -------------------------------------------------------------------------
    # 2) Compute new default values based on user-provided values
    # -------------------------------------------------------------------------
    # We'll do this in two steps:
    #   a) Filter the dataset for discrete columns exactly.
    #   b) For continuous columns the user provided, also do a tolerance-based filter.
    #   c) Then compute means/modes for the remaining columns from that filtered subset.

    data_filtered = data.copy()

    # We'll collect the user-provided columns so we know what we must NOT fill
    user_provided_cols = list(user_dict.keys())

    # Step (a) + (b): filter data for each user-provided column
    for key, val in user_dict.items():
        if key not in data_filtered.columns:
            # skip if not recognized
            continue
        if pd.api.types.is_numeric_dtype(data_filtered[key]):
            # For example, if user provided person_age=18, we do tolerance-based filtering
            if key == "person_age":
                data_filtered = filter_by_continuous_value(
                    data_filtered, key, val, tolerance=2, min_samples=10, max_expansions=5
                )
            else:
                # For other numeric columns (like income if the user provided it):
                # you could do a tolerance-based filter, but let's do direct match for demonstration
                data_filtered = data_filtered[data_filtered[key] == val]
        else:
            # For discrete attributes (gender, education, etc.) do exact match
            data_filtered = data_filtered[data_filtered[key] == val]

    # If filtering leads to an empty DataFrame, fallback to entire dataset
    if data_filtered.shape[0] == 0:
        data_filtered = data.copy()

    # Step (c) Build final_feature_values:
    # For user-provided keys, use them directly;
    # for missing ones, compute from the filtered subset
    final_feature_values = {}
    for col in data.columns:
        if col in user_dict:  # user-provided value
            final_feature_values[col] = user_dict[col]
        else:
            if pd.api.types.is_numeric_dtype(data_filtered[col]):
                # e.g. age, income
                final_feature_values[col] = data_filtered[col].mean()
            else:
                # e.g. gender, home ownership, etc.
                final_feature_values[col] = data_filtered[col].mode()[0]

    features = pd.DataFrame([final_feature_values])
    st.write("Final DataFrame for model prediction:", features)

    # -------------------------------------------------------------------------
    # 3) Perform model prediction
    # -------------------------------------------------------------------------
    prediction = model.predict(features)[0]

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Create a SHAP summary (optional for debugging)
    shap_summary = {
        key: shap_values[0][i] for i, key in enumerate(features.columns)
    }
    print(f"SHAP summary: {shap_summary}")

    # -------------------------------------------------------------------------
    # Custom bar plot: user-provided = blue, default-inferred = red
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
    # Generate a short GPT explanation without numbers
    # -------------------------------------------------------------------------
    explanation_prompt = f"""
The loan prediction was {"approved" if prediction == 1 else "not approved"}.

User provided columns: {user_dict}
Final model columns: {final_feature_values}

Give a sarcastic explanation of the influence of these changes, but do not include any numeric values.
"""
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
