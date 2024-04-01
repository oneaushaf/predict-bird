import tensorflow as tf
import numpy as np
import json
from io import BytesIO
from typing import Tuple
from PIL import Image
from fastapi import UploadFile

async def convert_image(image) -> np.ndarray:
    img = Image.open(BytesIO(await image.read()))
    img = img.resize(size=(224,224))
    result = np.array(img)
    result = result / 255.0
    return result

async def predict(image : UploadFile, model_version : str) -> Tuple[str,float]:
    species_list_path = './models/'+model_version+'/species.json'
    model_path = './models/'+model_version+'/SavedModel.h5'

    try:
        model = tf.keras.models.load_model(model_path)
    except OSError as e:
        return "Error: Failed to load the model", 0.0
    
    img = await convert_image(image)
    with open(species_list_path,"r") as f:
        species = json.load(f)

    predictions = model.predict(img[None,:,:])
    result = species[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    return result,confidence


