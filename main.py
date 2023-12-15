import io
from fastapi import FastAPI, HTTPException, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
from pydantic import BaseModel


app = FastAPI()


def get_prediction(model, x):
    """
    Gets the predicted digit
    :param model: MNIST Handwritten Digit Classification keras model instance
    :param x: input array
    :type: np.ndarray
    :return: predicted digit
    :rtype: int
    """
    y_hat = np.argmax(model.predict(x), axis=1)
    return int(y_hat[0])


def convert_image_data_to_input_array(image_io):
    """
    Converts image data to input array
    :param image_io: image io object
    :return: array
    :rtype: np.ndarray
    """
    image = Image.open(image_io).convert('L').resize((28, 28), reducing_gap=2.0)
    return np.asarray(image).reshape(1, 784)


class PredictionResult(BaseModel):
    result: int


@app.post("/", response_model=PredictionResult, summary='Predicts digit from an input image')
async def predict(file: UploadFile):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image.")
    contents = await file.read()
    matrix = convert_image_data_to_input_array(io.BytesIO(contents))
    prediction = get_prediction(tf.keras.models.load_model('model.h5'), matrix)
    return {"result": prediction}

