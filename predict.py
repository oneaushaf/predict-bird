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

async def predict(image : UploadFile, version : str = "latest") -> Tuple[list[str],list[float]]:
    models_dir = './../models/'
    species_list_path = models_dir + version + '/species.json'
    model_path = models_dir + version + '/SavedModel.h5'

    try:
        model = tf.keras.models.load_model(model_path)
    except OSError as e:
        return "Error: Failed to load the model " + model_path, 0.0
    
    img = await convert_image(image)
    with open(species_list_path,"r") as f:
        species = json.load(f)

    predictions = model.predict(img[None,:,:])
    sorted = np.argsort(predictions[0])
    result = [
                species[sorted[-1]],
                species[sorted[-2]],
                species[sorted[-3]],
    ]
    confidence = [
                predictions[0][sorted[-1]] * 100,
                predictions[0][sorted[-2]] * 100,
                predictions[0][sorted[-3]] * 100,
    ]

    return result,confidence



