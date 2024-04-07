from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel
import predict
import train

app = FastAPI()
is_training = False

@app.post("/predict/")
async def predict_image(
    image: UploadFile = File(...),
    version : str = Form(...),
):

    result,confidence = await predict.predict(image, version)
    return {
        result[0]:confidence[0],
        result[1]:confidence[1],
        result[2]:confidence[2],
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

    global is_training
    if (is_training):
        return {
        "message":"a training is in progress, please wait until the training is done",
    }
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

    global is_training
    if (is_training):
        return {
        "message":"a training is in progress, please wait until the training is done",
    }
    background_tasks.add_task(train.train_based,patience, base_model,epochs)
    return {
        "message":"training started, please wait until the training is done",
    }
