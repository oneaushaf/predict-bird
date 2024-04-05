from fastapi import FastAPI, File, UploadFile, Form
import predict
import train

app = FastAPI()
is_training = False

@app.post("/predict/")
async def predict_image(
    image: UploadFile = File(...),
    model_version : str = Form(...),
):
    result,confidence = await predict.predict(image, model_version)
    list = {
        result[0]:confidence[0],
        result[1]:confidence[1],
        result[2]:confidence[2],
    }
    return {
        "result":list
    }

@app.post("/train/new")
async def train_new_species(
    layers: int
):
    global is_training
    if (is_training):
        return {
        "message":"a training is in progress, please wait until the training is done",
    }
    is_training = True
    result = await train.train_new(layers)
    is_training = False
    return {
        "report":result,
    }
