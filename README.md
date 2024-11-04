# Project 1: Customer Churn Prediction with ML

In this repository, you'll find my solution to Project 1 from the Headstarter Acceleration Program.

## Overview

This project's objective is to predict customer churn using a prediction ML model, trained and evaluated on data from a Kaggle dataset (View on Kaggle: [Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)). All data is structured and labeled, so this is an exercise on supervised learning with classical prediction models. On top of that, I use generative AI to elaborate content that is based on parameters of the data and on the specific prediction.

This solution has 3 components: (1) a Jupyter notebook, used to explore the dataset and to train and evaluate the prediction models; (2) a streamlit application, used to visualize the data, the predictions and to review the generated content; (3) a FastAPI to serve the prediction functionality.

## Components

### Model training playgroud (Jupyter notebook)

On main.ipynb, I perform an EDA, split and scale data, and train several different models. Following, I pick the best performing model (XGBoost) and apply feature engineering and tune the models's learning rate. All models are saved with pickle and made available to make predictions.

### WebApp (Streamlit application)

Builds UI with Streamlit, loads models and gets predictions. Then it makes 2 completions with different models from GROQ to generate objective messages.

### API (FastAPI application)

A FastAPI app with a single method (/predict) that loads a model from a pickle file, manipulates the input for scaling and preprocessing and makes the prediction.
The API is hosted in render.com ([API URL](https://kaggle-churn-prediction-ml.onrender.com))

## Future Work

- Improve model performance with hyperparameter tuning to addess class imbalance
- Adapt webapp to use API
- Deploy API to GCP or AWS
