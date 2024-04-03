from fastapi import FastAPI, File, UploadFile,Form
import numpy as np
import predict
import train
from io import BytesIO
from PIL import Image

app = FastAPI()

working_dir = './' 
dataset_dir = working_dir + '/Dataset'

@app.post("/predict/")
async def predict_image(
    image: UploadFile = File(...),
    model_version : str = Form(...),
):
    result,confidence = await predict.predict(image, model_version)
    return {
        "species":result,
        "confidence":confidence
    }

@app.post("/train/new")
async def train_new_species(
    layers: int
):
    result = await train.train_new(64)
    return {
        "report":result,
    }
