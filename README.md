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

# Deploy API to AWS

## install pyenv

`sudo yum install git`

`curl https://pyenv.run | bash`

`vi ~/.bashrc`

```
export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
```

`source ~/.bashrc #restart terminal`

## Prepare environment before python (add dependencies)

sudo yum update
sudo yum install gcc make patch zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl11-devel tk-devel libffi-devel xz-devel

\*\*\* This will break on openssl11-devel

## Install Python

pyenv install 3.11.6
pyenv global 3.11.6

## Install pip

\*\*\* pip will be installed when you set up the python version as global (after the global method)

## Run the server

if **name** == "**main**":
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=4000)

## on Postman

http://{server_public_ip}:{PORT}/predict

Example request:

POST: http://3.133.102.24:4000/predict

Request body:

```
{
  "CreditScore": 619,
  "Age": 25,
  "Tenure": 2,
  "Balance": 0,
  "NumOfProducts": 4,
  "HasCrCard": 0,
  "IsActiveMember": 0,
  "EstimatedSalary": 101348.88,
  "Geography": "France",
  "Gender": "Female"
}
```

Output:

```
{
    "prediction": [
        1
    ],
    "probability": [
        [
            0.36,
            0.64
        ]
    ]
}
```

## Keep the server alive

`tmux new -s head_wk1 :create new session head_wk1`

`tmux ls :list all active sessions`

`tmux kill-session -t head_wk1 :kill session head_wk1`

`(Ctrl + B) + D :leave and keep session alive`

---

** The free EC2 instance does not fit the XGBoost module. I used a Random Forest model instead, just for completeness of the project. The performance is not good. The project is ok on render.com: https://kaggle-churn-prediction-ml.onrender.com.**

---

References:

- https://github.com/pyenv/pyenv/wiki#suggested-build-environment
