from fastapi import FastAPI
import joblib
from typing import Literal
from pydantic import BaseModel
import pandas as pd
from fastapi.middlewares.cors import CROSMiddleware

model=joblib.load('tuned_xgboost_pipeline.pkl')

app=FastAPI()


app.add_middleware(
    CROSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Features(BaseModel):
    Carat:float
    Cut:Literal['Ideal', 'Premium', 'Good', 'Very Good', 'Fair']
    Color:Literal['E', 'I', 'J', 'H', 'F', 'G', 'D']
    Clarity:Literal['SI2', 'SI1', 'VS2', 'VVS2', 'VVS1', 'VS1', 'I1', 'IF']
    Depth:float
    Table:float
    X:float
    Y:float
    Z:float


class Predict(BaseModel):
    Price:float


@app.post('/predict',response_model=Predict)
def predict(data:Features):
    input_row=pd.DataFrame([{
        'Carat(Weight of Daimond)':data.Carat,
        'Cut(Quality)':data.Cut,
        'Color':data.Color,
        'Clarity':data.Clarity,
        'Depth':data.Depth,
        'Table':data.Table,
        'X(length)':data.X,
        'Y(width)':data.Y,
        'Z(Depth)':data.Z
    }])

    pred=model.predict(input_row)[0]
    return Predict(Price=pred)