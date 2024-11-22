from fastapi import FastAPI, Form
from typing import Annotated
from pydantic import BaseModel
import pickle
import pandas as pd

# initialize FastAPI app
app = FastAPI()

# functions
def load_object(object_path):
  with open(object_path, "rb") as file:
    return pickle.load(file)   

# load prediction model and scaler
model = load_object('./model/xgb_model_feate.pkl')
feateng = True
scaler = load_object('./model/scaler_feate.pkl')

if not feateng:
    scaler = load_object('./model/scaler.pkl')


def digest_input(customer):
    '''Prepare data on input fields to be used by the models for prediction'''

    input_dict = {
        'CreditScore': int(customer['CreditScore']),
        'Age': int(customer['Age']),
        'Tenure': int(customer['Tenure']),
        'Balance': float(customer['Balance']),
        'NumOfProducts': int(customer['NumOfProducts']),
        'HasCrCard': int(customer['HasCrCard']),
        'IsActiveMember': int(customer['IsActiveMember']),
        'EstimatedSalary': float(customer['EstimatedSalary']),
        'Geography_France': 1 if customer['Geography'] == 'France' else 0,
        'Geography_Germany': 1 if customer['Geography'] == 'Germany' else 0,
        'Geography_Spain': 1 if customer['Geography'] == 'Spain' else 0,
        'Gender_Female': 1 if customer['Gender'] == 'Female' else 0,
        'Gender_Male': 1 if customer['Gender'] == 'Male' else 0
    }
    
    # include engineered features
    if(feateng == True):
        # CLV = balance * salary / 100000
        input_dict['CLV'] = input_dict['Balance'] * input_dict['EstimatedSalary'] / 100000
        # tenure / age
        input_dict['TenureByAge'] = input_dict['Tenure'] / input_dict['Age']
        # age group -> create bins 0,30,45,60,100
        # initialize values
        input_dict['AgeGroup_Adult'] = 0
        input_dict['AgeGroup_Senior'] = 0
        input_dict['AgeGroup_Elderly'] = 0
        # set correct value
        if(input_dict['Age'] < 30):
            None
        elif input_dict['Age'] < 45:
            input_dict['AgeGroup_Adult'] = 1
        elif input_dict['Age'] < 60:
            input_dict['AgeGroup_Senior'] = 1
        else:
            input_dict['AgeGroup_Elderly'] = 1

    input_df = pd.DataFrame([input_dict])

    input_df_scaled = scaler.transform(input_df)

    return input_df_scaled

def get_predictions(customer):
    customer_scaled = digest_input(customer)

    prediction = model.predict(customer_scaled)
    probability = model.predict_proba(customer_scaled)

    return prediction, probability

app = FastAPI()

class Customer(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography: str
    Gender: str

@app.post("/predict")
async def predict(customer: Customer):

    prediction, probability = get_predictions(customer.model_dump())

    return {
        "prediction": prediction.tolist(),
        "probability": probability.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)