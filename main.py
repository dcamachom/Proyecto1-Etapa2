from typing import Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
from joblib import load, dump
from DataModel import DataModel
from fastapi.staticfiles import StaticFiles

import logging

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()
app.mount("/static", StaticFiles(directory="templates"), name="static") 

@app.get("/")
def read_root():
   with open('templates/index.html','r') as file:
      conten= file.read()
   return HTMLResponse(conten)


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

@app.post("/predict")
def make_predictions(dataModel: DataModel):
    try:
        df = pd.DataFrame(dataModel.dict(), columns=dataModel.columns(), index=[0])
        print(df)
    except Exception as e:
        return {"error": f"Error al crear DataFrame: {e}"}

    model = load("assets/modelo.joblib")
    try:
        result = model.predict(df)
        return {"result": result.tolist()}  # Convertir a lista si es necesario
    except Exception as e:
        return {"error": f"Error al hacer la predicción: {e}"}


@app.post("/retrain")
def retrain_model(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.columns(), index=[0])
    X = df.drop(columns=["classification"])
    y = df["classification"]

    model = load("assets/modelo.joblib")
    model.fit(X, y)
    
    dump(model, "assets/modelo_reentrenado.joblib")
    
    return {"message": "Modelo reentrenado con éxito."}