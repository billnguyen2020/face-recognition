# HELPER FUNCTIONS
import cv2
import dlib
import numpy as np

from tensorflow import expand_dims
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize

progress = 0


def align_face(fa, img, x, y, w, h):
    """
    Align face to the center
    Arguments:
        fa: Defined Face Aligner class
        img: Input face image
        x, y, w, h: Image Coordinates and Size
    Return:
        Aligned image as numpy array
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fa_img = fa.align(img, gray, dlib.rectangle(
        left=x, top=y, right=x + w, bottom=y + h))

    return fa_img


def get_embedding(model, img):
    """
    Get embedding of input image
    Arguments:
        model: Embedding model
        img: Input image
    Return:
        Embedding of input image

    """
    img = resize(img, (160, 160))  # Resize
    img /= 255  # Normalize
    img = expand_dims(img, axis=0)  # Expand dimension
    embedding = model.predict(img)[0]  # Get embedding

    return embedding


def generate_data(model, path):
    """
    Generate data from an image path
    Arguments:
        model: Embedding model
        path: path to an image
    Return:
        embedding: Embedding of input image
        label: label of input image
    """
    global progress

    img = img_to_array(load_img(path))

    embedding = get_embedding(model, img)
    label = path.split('_')[0].split('/')[-1]

    progress += 1
    if progress % 10 == 0:
        print("Imaged processed: {}".format(progress))

    return embedding, label
