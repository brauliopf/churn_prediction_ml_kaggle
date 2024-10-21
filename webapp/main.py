'''
About Streamlit
Streamlit reruns your script from top to bottom every time you interact with your app.
Each reruns takes place in a blank slate: no variables are shared between runs.
Use Session State and Callbacks to manage data through various run-throughs.
Reference: https://docs.streamlit.io/develop/concepts/architecture/session-state
'''

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
   base_url='https://api.groq.com/openai/v1',
   api_key=os.environ["GROQ_API_KEY"]
)

# FUNCTIONS
def load_model(filename):
  # open file as binary file and load with pickle
  with open(filename, "rb") as file:
    return pickle.load(file)

def load_models():
  '''Create global variables with specific models loaded from pkl files'''
  models = ['xgb_model', 'gnb_model', 'rfc_model', 'tre_model', 'svm_model',
          'knn_model', 'xgb_featureengineering', 'xgb_smote', 'xgb_ensemble_hard',
          'xgb_ensemble_soft']
  for model_name in models:
    globals()[model_name] = load_model(f"../models/{model_name}.pkl")

def prepare_input(credit_score, location, gender, age, tenure, balance,
                 num_products, has_credit_card, is_active_member,
                 estimated_salary):
    '''Prepare data on input fields to be used by the models for prediction'''
    
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

import utils as ut
def make_predictions(input_df):
    '''Prints 3 predictions and the average of 3 models: XGBoost, Random Forest, KNN.
    Returns the average of the predictions of the 3 models'''
    probabilities = {
        'XGBoost': xgb_model.predict_proba(input_df)[0][1],
        'Random Forest': rfc_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
    }
    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    with col1:
       fig = ut.create_gauge_chart(avg_probability)
       st.plotly_chart(fig, use_contaioner_width=True)
       st.write(f"The customer has {avg_probability:.2%} probability of churning")

    with col2:
       fig_probs = ut.create_model_probability_chart(probabilities)
       st.plotly_chart(fig_probs, use_container_width=True)

    return avg_probability 

def explain_prediction(probability, input_dict, surname, df, model="llama-3.2-3b-preview"):
    '''Engage with an LLM to explain the prediction of the machine learning model based on
    feature importances and customer data, and summary statistics of churned and non-churned customers'''

    prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.
    You are given the probability of churn and must explain the model's prediction based on your assessment of the context: customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.

    - Do not mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction.
    - Use the model's prediction. Do not make up your own assessment. Use this criteria:
      - [Riosk of churning > 30%] Custoner is at risk ofhcuirning. Generate a 3 sentence explanation of why they are at risk of churning.
      - [Risk of churning < 30%] Customer might not be at risk of churning. Generate a 2 sentence explanation of why they might not be at risk of churning.

    Customer information is shared below.
    
    Customer information:
    Name: {surname}
    Risk of churning: {round(probability * 100, 1)}% (decimal: {probability})
    General customer data: {input_dict}

    Here are the machine learning model's top 10 most important features for predicting churn:
          Feature             | Importance
          --------------------------------
          NumOfProducts       | 0.323888
          IsActiveMember      | 0.164146
          Age                 | 0.109550
          Geography_Germany   | 0.091373
          Balance             | 0.052786
          Geography_France    | 0.046463
          Gender_Female       | 0.045283
          Geography_Spain     | 0.036855
          CreditScore         | 0.035005
          EstimatedSalary     | 0.032655
          HasCrCard           | 0.031940
          Tenure              | 0.030054
          Gender_Male         | 0.000000
"""

    print("EXPLANATION PROMPT", prompt)

    raw_response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    return raw_response.choices[0].message.content
   
def build_UI(selected_customer):
      '''
      Use 2 columns.
      Declaring an st (Streamlit) object adds it to the UI and session.
      Returns: output of prepare_input(...): (input_df, input_dict)
      '''
      
      col1, col2 = st.columns(2)

      with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer['CreditScore']),
            # disabled=True
            )
        
        location = st.selectbox(
          "Location", ["Spain", "France", "Germany"],
          index=["Spain", "France", "Germany"].index(selected_customer['Geography']))
        
        gender = st.radio("Gender", ["Male", "Female"], index=0 if selected_customer['Gender'] == 'Male' else 1)
        
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(selected_customer['Age']) if 'Age' in selected_customer else None)

        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer['Tenure']))
        
      with col2:
        balance = st.number_input(
          "Balance",
          min_value=0.0,
          value=float(selected_customer['Balance']))

        num_products = st.number_input(
          "Number of Products",
          min_value=1,
          max_value=10,
          value=int(selected_customer['NumOfProducts']))

        has_credit_card = st.checkbox(
          "Has Credit Card",
          value=bool(selected_customer['HasCrCard']))

        is_active_member = st.checkbox(
          "Is Active Member",
          value=bool(selected_customer['IsActiveMember']))

        estimated_salary = st.number_input(
          "Estimated Salary",
          min_value=0.0,
          value=float(selected_customer['EstimatedSalary']))

      return prepare_input(credit_score, location, gender, age,
                                          tenure, balance, num_products, has_credit_card,
                                          is_active_member, estimated_salary)

def compose_email(probability, input_dict, explanation, surname):
  """
  Generates a personalized email to a customer based on their churn probability.

  Args:
      probability (float): The probability of the customer churning.
      input_dict (dict): A dictionary containing the customer's information.
      explanation (str): An explanation of why the customer might be at risk of churning.
      surname (str): The customer's surname.

  Returns:
      str: The generated email.
  """

  prompt = f"""You are a manager at HS Bank. You are responsible for
  ensuring customers stay with the bank and are incentivized with
  various offers.

  You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

  Here is the customer's information:
  {input_dict}

  Here is some explanation as to why the customer might be at risk
  of churning:
  {explanation}

  Generate an email to the customer based on their information,
  asking them to stay if they are at risk of churning, or offering them
  incentives so that they become more loyal to the bank.

  Make sure to list out a set of incentives to stay based on their
  information, in bullet point format. Don't ever mention the
  probability of churning, or the machine learning model to the
  customer.
  """

  raw_response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{
       "role": "user",
       "content": prompt
    }],

  )

  print("\n\nEMAIL PROMPT", prompt)

  return raw_response.choices[0].message.content   

def display_customer_percentile(customer, everyone):
  # get metrics
  display_metrics = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

  # calculate percentiles
  percentiles = {}
  for metric in display_metrics:
    # sort context
    values = np.sort(everyone[metric])
    # get index
    index = np.searchsorted(values, customer[metric])
    # Calculate the percentile rank
    percentiles[metric] = (index / len(values))

  # plot chat
  fig_percents = ut.create_percentiles_chart(percentiles)
  st.plotly_chart(fig_percents, use_container_width=True)

  return percentiles
  

def main():
  st.title("Customer Churn Prediction")
  
  df = pd.read_csv("../data/churn.csv")

  customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

  selected_customer_option = st.selectbox("Select a customer", options=customers, index=None) # index == default value

  if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]
    input_df, input_dict = build_UI(selected_customer)
    display_customer_percentile(selected_customer, df)

    load_models()
    avg_probability = make_predictions(input_df)
    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'], df, "llama-3.1-70b-versatile")
    st.markdown("---")
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)

    email = compose_email(avg_probability, input_dict, explanation, selected_customer['Surname'])
    st.markdown("---")
    st.subheader("Personalized Email")
    st.markdown(email)

  else:
    st.write("Please select a customer to proceed.")

  return None;

main()