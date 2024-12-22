import pandas as pd
import pickle
import shap
import openai
import streamlit as st
import keyring
import re
import matplotlib.pyplot as plt
import numpy as np  
import matplotlib.patches as mpatches

def extract_json_content(input_string):
    # Use regular expression to find the first '{' and last '}'
    match = re.search(r'\{.*\}', input_string, re.DOTALL)
    if match:
        return match.group(0)
    return None

# Load model and dataset
with open("fortunate_loan_model_gpu.pkl", "rb") as file:
    model = pickle.load(file)

data = pd.read_csv("loan_data_preprocessed.csv")

# Calculate average or most common values for each attribute
default_values = {
    'person_age': data['person_age'].mean(),
    'person_gender': str(data['person_gender'].mode()[0]),  
    'person_education': str(data['person_education'].mode()[0]),
    'person_income': data['person_income'].mean(),
    'person_emp_exp': data['person_emp_exp'].mean(),
    'person_home_ownership': str(data['person_home_ownership'].mode()[0]),
    'loan_amnt': data['loan_amnt'].mean(),
    'loan_int_rate': data['loan_int_rate'].mean(),
    'loan_percent_income': data['loan_percent_income'].mean(),
    'cb_person_cred_hist_length': data['cb_person_cred_hist_length'].mean(),
    'credit_score': data['credit_score'].mean(),
    'previous_loan_defaults_on_file': str(data['previous_loan_defaults_on_file'].mode()[0]),
    'loan_intent_DEBTCONSOLIDATION': 0,
    'loan_intent_EDUCATION': 0,
    'loan_intent_HOMEIMPROVEMENT': 0,
    'loan_intent_MEDICAL': 0,
    'loan_intent_PERSONAL': 0,
    'loan_intent_VENTURE': 0
}

# one-hot management: set the most common loan intent to 1 and keep the rest at 0
most_common_intent = data[
    [
        'loan_intent_DEBTCONSOLIDATION',
        'loan_intent_EDUCATION',
        'loan_intent_HOMEIMPROVEMENT',
        'loan_intent_MEDICAL',
        'loan_intent_PERSONAL',
        'loan_intent_VENTURE'
    ]
].sum().idxmax()
default_values[most_common_intent] = 1

# Set up OpenAI API
api_key = keyring.get_password("openai_api", "key_a")
openai.api_key = api_key

# Streamlit UI
st.title("Worker Loan Approval Prediction & Analysis")
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey, I can predict whether your bank loan application will be approved! :)\n\nPlease provide me with data relevant for this purpose. Our model can deal with data about the interest rate, your credit score, work experience and many more features. Simply paste your data in the form you wish. It does not have to be complete. We will take care of the rest!"}
    ]

# display greeting message
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
        f"Given the following default values for a loan approval dataset: {default_values}, and the user's input: '{prompt}', "
        f"generate a JSON-formatted dataframe with all keys from the default values, replacing default values with any applicable values from the user input. "
        f"The values need to be numerical (no strings). "
        f"For 'person_gender': 0 means 'male' and 1 means 'female'. "
        f"For 'person_education': 0 means 'High School', 1 means 'Associate', 2 means 'Bachelor', 3 means 'Master', 4 means 'Doctorate'. "
        f"For 'person_home_ownership': 0 means 'RENT', 1 means 'MORTGAGE', 2 means 'OWN'. "
        f"For 'previous_loan_defaults_on_file': 0 means 'No' and 1 means 'Yes'. "
        f"The following columns are one-hot encoded: 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE'. "
        f"If the user attempts to set more than one loan intent key to 1 at once, correct it so that only one key is set to 1. If the user provides information on education or gender and the like which are out of scope of the possible values specified above, then don't take the user-specified value, but instead take the closest value possible (e.g. High School is the closest possible value if the user claims to have Middle School degree.) or keep the initial default value if there is no closest alternative (e.g. neither male nor female are closer to user-provided input 'agender')"
        f"Also note that 'person_age' is measured in years, 'person_income' in dollars per year, 'person_emp_exp' in years, 'loan_amnt' in dollars, 'loan_int_rate' in percent, "
        f"'cb_person_cred_hist_length' in years. If the user provides any of these in different units, convert them to the correct units before placing them in the JSON."
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
        max_tokens=1000
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

    # perform model prediction
    prediction = model.predict(features)[0]

    # calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # link SHAP values to feature names
    shap_summary = {
        key: shap_values[0][i] for i, key in enumerate(features.columns)
    }
    print(f"shap summary: {shap_summary}")

    # shap plot: sort shap values descendingly (strongest impact as first bar on top)
    sorted_cols = sorted(shap_summary, key=lambda x: abs(shap_summary[x]), reverse=True)
    sorted_shap_vals = [shap_summary[col] for col in sorted_cols]

    # Determine color: blue if user-provided (differs from default), red if default
    bar_colors = []
    for col in sorted_cols:
        default_val = default_values[col]
        user_val = features[col].iloc[0]

        # Attempt float comparison if both are numeric; fallback to string comparison otherwise
        try:
            if float(user_val) == float(default_val):
                bar_colors.append('red')
            else:
                bar_colors.append('blue')
        except:
            if str(user_val) == str(default_val):
                bar_colors.append('red')
            else:
                bar_colors.append('blue')

    # create plot
    fig, ax = plt.subplots()
    y_pos = np.arange(len(sorted_cols))
    ax.barh(y_pos, sorted_shap_vals, color=bar_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_cols)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value")
    ax.set_title("Feature Importance (Single Prediction)")

    # legend
    blue_patch = mpatches.Patch(color='blue', label='User-provided (changed)')
    red_patch = mpatches.Patch(color='red', label='Default values')
    ax.legend(handles=[blue_patch, red_patch], loc='best')

    # streamlit rendering of plot
    st.pyplot(fig)

    # GPT explanation prompt
    explanation_prompt = f"""
The loan prediction was {"approved" if prediction == 1 else "not approved"}. 
Below you have two JSON objects: 
1) default_initial_json: {default_values}
2) final_user_json: {features.to_dict(orient='records')[0]}

The final_user_json may contain values deviating from the original user message: "{prompt}". The final_user_json is more important.

The summary of SHAP values for the final_user_json is as follows: {shap_summary}

You are talking directly to the user. Provide a simple, short and clear explanation of how the final decision is influenced by the features, 
ignoring specific numeric values. Focus on three things: 1. Focus on positive vs negative impacts of features. 2. Focus on user-provided vs default values for features. 3. Focus on the most impactful features (i.e. those with the most extreme SHAP values). Do not write separate paragraphs for these three things, but follow a combined approach. 
Do not include any numeric values in the explanation. You may use phrases like "... half as influential ..." or "... three times as significant as ..." and the like though. Also do not talk about "JSON", "SHAP" or other technical datastructures. Just tell the end user what he is interested in. Do not speak in past tense, but in future.

Additional context:
- 'person_gender': 0 means 'male' and 1 means 'female'
- 'person_education': 0 means 'High School', 1 means 'Associate', 2 means 'Bachelor', 3 means 'Master', 4 means 'Doctorate'
- 'person_home_ownership': 0 means 'RENT', 1 means 'MORTGAGE', 2 means 'OWN'
- 'previous_loan_defaults_on_file': 0 means 'No' and 1 means 'Yes'
- The following columns are one-hot encoded: 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE'
- 'person_age' is measured in years, 'person_income' in dollars per year, 'person_emp_exp' in years, 'loan_amnt' in dollars, 'loan_int_rate' in percent, 'cb_person_cred_hist_length' in years.

Do not write sentences like "Even though you shared that you have no previous defaults, the system defaults to assuming a negative past due to the misalignment with your overall profile.". Your answer must be fully based on the final_user_json. default_initial_json only exists so you can find out which features were user provided and which not.

Finish your explanation with a sarcastic line. Here are two example responses which you should orient yourself towards while keeping in mind that your results might deviate:

1. I sadly have to tell you that your loan application will not be approved. The data you provided (credit score, gender, etc.) had a positive influence on the decision making process. However, these attributes are way less important than the history of previous loan defaults. Since you did not provide any information for this particular feature, we have no choice but to refer to our data from the internal database, meaning that we assume that you had previous loan defaults. This single feature was enough to assume that your loan application will most likely not be approved. Mind you provide more good news about yourself the next time, will ya?

2. Well done, you managed to pack lots of horrible indicators in your loan application, as you lack any kind of work experience with a credit score hitting the rock bottom. The bank is probably laughing about the fact that someone like you has a Master's degree. Nonetheless, the immense interest rate of 68.3% which you are willing to pay managed to weigh out all those negative indicators you provided. Furthermore, our database assumes your age to be 41, adding a slight bonus in the decision making process, since people with life experience, but young enough to have the drive to achieve much in the coming years, are generally speaking more trustworthy. Have fun with your loan, but stay responsible. You don't want to ruin your reputation any further than already, do you?

Use the rules, context data and text examples I provided you with to write a loan approval status application to the end user. Keep in mind which features the user actually provided and which not. Also keep in mind that positive SHAP values influence the loan approval decision positively and negative values negatively. The more extreme the number, the greater the impact. Your explanation reasoning must not conflict with the reality of these SHAP values.
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
        max_tokens=2000
    )

    explanation_message = gpt_explanation_response.choices[0].message.content.strip()

    # display the GPT explanation
    with st.chat_message("assistant"):
        st.markdown(explanation_message)

    st.session_state.messages.append({"role": "assistant", "content": explanation_message})
