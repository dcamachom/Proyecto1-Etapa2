import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pandas import DataFrame
from joblib import load, dump
import pandas as pd
from DataModel import DataModel

logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="templates"), name="static")

@app.get("/")
def read_root():
    with open('templates/index.html', 'r') as file:
        content = file.read()
    return HTMLResponse(content)

@app.post("/uploadfile/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        decoded = contents.decode("utf-8")
        lines = decoded.split("\n")
        lines = lines[1:]
        data = [line.split(",") for line in lines if line.strip()]
        df = DataFrame(data, columns=["review", "classification"])
        return df.to_dict()
    except Exception as e:
        logger.error(f"Error al procesar el archivo CSV: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo CSV: {str(e)}")

@app.post("/predict")
def make_predictions(data: dict):
    try:
        logger.debug(f"Datos recibidos para predicción: {data}")
        df = pd.DataFrame(data)
        logger.debug(f"DataFrame recibido: {df}")
    except Exception as e:
        logger.error(f"Error al crear DataFrame: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error al crear DataFrame: {str(e)}")

    try:
        model = load("assets/model.joblib")
        logger.debug("Modelo cargado exitosamente")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {str(e)}")

    try:
        result = model.predict(df)
        return {"result": result.tolist()}
    except Exception as e:
        logger.error(f"Error al hacer la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al hacer la predicción: {str(e)}")

@app.post("/retrain")
def retrain_model(dataModel: DataModel):
    try:
        df = pd.DataFrame(dataModel.dict(), columns=dataModel.columns(), index=[0])
        X = df.drop(columns=["classification"])
        y = df["classification"]

        model = load("assets/model.joblib")
        model.fit(X, y)

        dump(model, "assets/modelo_reentrenado.joblib")

        return {"message": "Modelo reentrenado con éxito."}
    except Exception as e:
        logger.error(f"Error al reentrenar el modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al reentrenar el modelo: {str(e)}")
