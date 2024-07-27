from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel
import predict
import train
import test

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
    base_model : str = "temp"
    epochs : int = 100

@app.post("/train/based")
async def train_based_model(
    request : BasedTrainRequest,
    background_tasks : BackgroundTasks,
):
    patience = request.patience
    epochs = request.epochs
    base_model = request.base_model
    # return {
    #     "message":"training started, please wait until the training is done",
    #     "patience" : request.patience,
    #     "epochs" : request.epochs,
    #     "base_model" : request.base_model,
    # }
    background_tasks.add_task(train.train_based, base_model,patience,epochs)
    return {
        "message":"training started, please wait until the training is done",
    }

@app.get("/test/{model}")
async def train_based_model(
    model:str = "temp"
):
    # return {
    #     "message":"training started, please wait until the training is done",
    #     "patience" : request.patience,
    #     "epochs" : request.epochs,
    #     "base_model" : request.base_model,
    # }
    result = test.test(model)
    return {
        "result" : result,
    }
