import io
from fastapi import FastAPI, File
from typing_extensions import Annotated
import tensorflow as tf
import numpy as np
from PIL import Image
from numpy import asarray


app = FastAPI()


def load_model(file_name):
    return tf.keras.models.load_model(file_name)


def get_predict(model, X):
    y_hat = np.argmax(model.predict(X), axis=1)
    print(np.shape(y_hat), y_hat)
    return int(y_hat[0])


def get_matrix(img):
    image = Image.open(img).convert('L').resize((28, 28), reducing_gap=2.0)
    as_array = asarray(image)
    matrix = as_array.reshape(1, 784)
    return matrix


@app.post("/predict")
async def create_file(file:  Annotated[bytes, File()]):
    bytes_io = io.BytesIO(file)
    matrix = get_matrix(bytes_io)
    predict = get_predict(load_model('model.h5'), matrix)
    return {"result": predict}
