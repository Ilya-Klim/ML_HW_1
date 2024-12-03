
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from joblib import load
import pandas as pd
from io import StringIO
import io
import re
import numpy as np
from fastapi.responses import StreamingResponse

app = FastAPI()
model = load('ridge.joblib')["model"]

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

class DataPreprocessing:
    def __init__(self, data):
        self.data = pd.DataFrame([data]) if isinstance(data, dict) else data

    def data_cleaning(self):

        if "torque" in self.data.columns:
            self.data = self.data.drop(columns=["torque"], axis=1)

        for col in ["mileage", "engine", "max_power"]:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(lambda x: re.sub(r"[^\d.]+", "", str(x)))

        for col in ["mileage", "engine", "max_power"]:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(lambda x: np.nan if len(str(x)) <= 1 else float(x))

        for col in ["mileage", "engine", "max_power", "seats"]:
            if col in self.data.columns:
                median = self.data[col].median()
                self.data[col] = self.data[col].fillna(median)

        for col in ["engine", "seats"]:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(int)

        return self

    def ohe(self):    
        categorical_columns = ["fuel", "seller_type", "transmission", "owner"]
        for col in categorical_columns:
            if col in self.data.columns:
                self.data = pd.get_dummies(self.data, columns=[col], drop_first=True)
        return self.data

def create_csv_response(dataframe: pd.DataFrame, filename: str = "output.csv"): 
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    response = StreamingResponse(csv_buffer, media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


@app.post("/predict_item")
def predict_item(item: Item):
    data = item.dict()
    preprocessing = DataPreprocessing(data)
    formatted_data = preprocessing.data_cleaning().ohe()
    prediction = model.predict(formatted_data)
    return {"Predicted car price": prediction.tolist()}


@app.post("/predict_items")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    preprocessing = DataPreprocessing(df)
    formatted_data = preprocessing.data_cleaning().ohe()
    predictions = model.predict(formatted_data)
    df["predicted_price"] = predictions
    return create_csv_response(df, filename="predictions_output.csv")
