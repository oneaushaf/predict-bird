from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel
from threading import Thread
import predict
import train

app = FastAPI()

@app.post("/predict/")
async def predict_image(
    image: UploadFile = File(...),
    version : str = Form(...),
):

    result,confidence = await predict.predict(image, version)
    return {
        "result": result[0],
        "confidence":{
            result[0]:confidence[0],
            result[1]:confidence[1],
            result[2]:confidence[2],
        } 
    }

class NewTrainRequest(BaseModel):
    epochs : int = 100
    layers : int = 64
    patience : int = 10

@app.post("/train/new")
async def train_new_model(
    request : NewTrainRequest,
    background_tasks: BackgroundTasks
):
    layers = request.layers
    patience = request.patience
    epochs = request.epochs

    background_tasks.add_task(train.train_new,layers, patience,epochs)
    return {
        "message":"training started, please wait until the training is done",
    }


class BasedTrainRequest(BaseModel):
    patience : int = 10
    epochs : int = 100

@app.post("/train/based/{base_model}")
async def train_based_model(
    request : BasedTrainRequest,
    base_model = "latest",
    background_tasks = BackgroundTasks,
):
    patience = request.patience
    epochs = request.epochs

    background_tasks.add_task(train.train_based,patience, base_model,epochs)
    return {
        "message":"training started, please wait until the training is done",
    }
