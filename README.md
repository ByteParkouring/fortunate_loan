# Fortunate Loan

You want to find out whether the bank will approve your loan application? No big deal! Just provide (relevant) information about yourself in free text form to our Chatbot and it will deliver an answer with 90% accuracy.

But the key point is: The bot also provides explanations on the final decision based on **SHAP**!

## Quickstart

```streamlit run fortunate_gpt.py```

Each prompt represents a new loan request with no memory inbetween. 

![PNG Image](presentation/chat_example.png)

(Install the entries of `pip_libraries.txt` and `conda_libraries.txt` recursively if you have troubles with getting it to run)

Also make sure to provide your own openai API key.

No powerful GPU is required. The training was performed on an Intel CPU with Intel Iris Xe GPU acceleration.

## Overview

The most relevant files of this repository are the following:

- `loan_data.csv`: downloaded initial data
- `fortunate_training.py`: processes initial data to create `loan_data_preprocessed.csv` and train a model `fortunate_loan_model_gpu.plk`
- `fortunate_gpt.py`: the final application for the end user
- `fortunate_experiments.py`: more plots and analysis on SHAP itself including background info and theoretical explanations

## General concept

![SVG Image](presentation/aml_sketch.drawio.svg)

The user provides data which is then processed in a GPT request to create a JSON structure that can be extracted by the program to pass corresponding parameters to the loan model. The model performs a prediction. The result is then numerically explained with SHAP. These informations are then passed in another GPT request to be reformulated in natural language with an own interpretation of the situation. 

## Bugs

- Often the explanation states that a user failed to provide a feature value even though he did
- Sometimes GPT creates a JSON structure with wrong values from the user provided prompt data, also leading to flawed waterfall plots.

The instructions I gave to GPT in `fortunate_gpt.py` appear clear and complete to me though. I also don't want to switch to a model which costs more than gpt-4o-mini. 

## Sources

- Loan dataset: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data 
- Model trained from scratch: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
- SHAP analysis: https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html  
- Streamlit UI: https://docs.streamlit.io/develop/tutorials
- OpenAI Chat: https://platform.openai.com/docs/guides/text-generation

