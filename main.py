from typing import Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
from joblib import load
from DataModel import DataModel
from fastapi.staticfiles import StaticFiles

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
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    return result

@app.post("/retrain")
def retrain_model(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = load("assets/modelo.joblib")
    result = model.fit(df)
    return result