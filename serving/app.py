from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
from serving.model_loader import get_model

'''
Right now startup loads:

model = mlflow.pyfunc.load_model(...)

at import time.

For production we usually do:

@app.on_event("startup")

or use FastAPI lifespan management.

We'll improve that later.

For learning purposes, this is fine.

'''


app = FastAPI(
    title="Income Prediction API",
    version="1.0"
)

# # --------------
# # Load Model Once at startup
# # --------------

# model = mlflow.pyfunc.load_model(
#     model_uri="models:/IncomeModel@Champion"
# )


# --------------------
# Request Schema 
# ------------------

class IncomeRequest(BaseModel):

    age: int
    workclass: str
    fnlwgt: int
    education: str
    educational_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    gender: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    
# -------------------
# Health Endpoint
# -------------------

@app.get("/health")
def health():
    return {"status": "healthy"}


# -----------------------------------
# Prediction Endpoint
# -----------------------------------

@app.post("/predict")
def predict(request: IncomeRequest):

    df = pd.DataFrame([{
        "age": request.age,
        "workclass": request.workclass,
        "fnlwgt": request.fnlwgt,
        "education": request.education,
        "educational-num": request.educational_num,
        "marital-status": request.marital_status,
        "occupation": request.occupation,
        "relationship": request.relationship,
        "race": request.race,
        "gender": request.gender,
        "capital-gain": request.capital_gain,
        "capital-loss": request.capital_loss,
        "hours-per-week": request.hours_per_week,
        "native-country": request.native_country
    }])

    model = get_model()
    pred = model.predict(df)

    return {
        "prediction": int(pred[0])
    }